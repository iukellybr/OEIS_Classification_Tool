import os
import random
import warnings
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.ensemble import IsolationForest
import umap
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────────────────
# Configuration and Setup
# ───────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*n_jobs value.*overridden.*")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    "batch_size": 8,                 # images per batch
    "ssl_epochs": 5,                 # SSL pretrain epochs
    "ssl_lr": 1e-3,                  # SSL learning rate
    "ssl_temp": 0.5,                 # contrastive loss temperature
    "num_workers": 0,                # DataLoader workers
    "base_dir": os.path.expanduser("~/Documents/OEIS_Sequence_Repository"),
    "cache_template": "./features_{model_name}.h5",  # HDF5 cache pattern
    "seed": 42,                      # RNG seed
    "contamination": 0.02,           # IsolationForest contamination
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA device")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS device")
else:
    device = torch.device("cpu")
    logger.info("Using CPU device")

linear_dir = os.path.join(CONFIG["base_dir"], "Linear_Scatterplots")
log_dir    = os.path.join(CONFIG["base_dir"], "Logarithmic_Scatterplots")

# ───────────────────────────────────────────────────────────────────────────────
# Dataset Definitions
# ───────────────────────────────────────────────────────────────────────────────
class OEISDataset(Dataset):
    """Loads PNG scatterplots and returns (image_tensor, id)."""
    def __init__(self, dirs, transform=None):
        self.transform = transform
        self.images, self.ids = [], []
        for d in dirs:
            for f in os.listdir(d):
                if f.endswith('.png'):
                    self.images.append(os.path.join(d, f))
                    self.ids.append(os.path.splitext(f)[0])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.ids[idx]

class SSLDataset(Dataset):
    """Yields two different augmentations of the same image."""
    def __init__(self, dirs, transform):
        self.base = OEISDataset(dirs, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.transform(img), self.transform(img)

# ───────────────────────────────────────────────────────────────────────────────
# Transforms
# ───────────────────────────────────────────────────────────────────────────────
ssl_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

infer_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ───────────────────────────────────────────────────────────────────────────────
# SimCLR Components
# ───────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """MLP head that maps embeddings to contrastive space."""
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class CombinedEmbedder(nn.Module):
    """Concatenates intermediate + classifier pre-logit features."""
    def __init__(self, backbone, proj_head):
        super().__init__()
        self.features     = backbone.features
        self.avgpool      = backbone.avgpool
        self.classifier0  = backbone.classifier[0]
        self.proj         = proj_head

    def forward(self, x):
        f = self.features(x)
        f = self.avgpool(f).flatten(1)   # intermediate features
        c = self.classifier0(f)          # final-layer pre-logits
        combined = torch.cat([f, c], dim=1)
        z = self.proj(combined)          # projection for contrastive
        return combined, F.normalize(z, dim=1)

def nt_xent_loss(z1, z2, temperature):
    """Normalized Temperature-Scaled Cross Entropy (SimCLR) loss."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                 # (2B, D)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    mask = torch.eye(2*B, device=sim.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    targets = torch.arange(B, device=sim.device)
    targets = torch.cat([targets + B, targets], dim=0)
    return F.cross_entropy(sim, targets)

def self_supervised_pretrain(embedder, loader, epochs, lr, temp):
    """Fine-tunes projection head with contrastive loss; backbone frozen."""
    optimizer = torch.optim.Adam(embedder.proj.parameters(), lr=lr)
    embedder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for v1, v2 in tqdm(loader, desc=f"SSL Epoch {epoch+1}/{epochs}"):
            v1, v2 = v1.to(device), v2.to(device)
            _, z1 = embedder(v1)
            _, z2 = embedder(v2)
            loss = nt_xent_loss(z1, z2, temp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1} SSL loss: {total_loss/len(loader):.4f}")

# ───────────────────────────────────────────────────────────────────────────────
# Embedding Extraction & Caching
# ───────────────────────────────────────────────────────────────────────────────
def extract_and_cache_embeddings(model, loader, cache_template, model_name):
    """Extract combined embeddings, cache to HDF5, or load if exists."""
    cache_file = os.path.abspath(cache_template.format(model_name=model_name))
    logger.info(f"CWD: {os.getcwd()}")
    logger.info(f"Checking cache at: {cache_file} (exists={os.path.exists(cache_file)})")
    if os.path.exists(cache_file):
        with h5py.File(cache_file, 'r') as f:
            embs = f['emb'][:]
            ids  = [x.decode() for x in f['ids'][:]]
        logger.info(f"Loaded {len(ids)} embeddings from cache.")
        return embs, ids

    model.eval()
    all_embs, all_ids = [], []
    with torch.no_grad():
        for imgs, batch_ids in tqdm(loader, desc="Extract embeddings"):
            x = imgs.to(device)
            comb, _ = model(x)
            all_embs.append(comb.cpu().numpy())
            all_ids.extend(batch_ids)
    embs = np.vstack(all_embs)

    # Write if doesn't exist
    with h5py.File(cache_file, 'x') as f:
        f.create_dataset('emb', data=embs)
        f.create_dataset('ids', data=np.array(all_ids, dtype='S'))
    logger.info(f"Cached {len(all_ids)} embeddings to {cache_file}")
    return embs, all_ids

# ───────────────────────────────────────────────────────────────────────────────
# Outlier Detection & Visualization
# ───────────────────────────────────────────────────────────────────────────────
def detect_outliers(embs, cont):
    iso = IsolationForest(contamination=cont, random_state=CONFIG["seed"])
    labs = iso.fit_predict(embs)
    scores = iso.decision_function(embs)
    return labs, scores

def plot_umap(embs, ids, labs):
    lowd = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=CONFIG["seed"]).fit_transform(embs)
    df = pd.DataFrame({'x':lowd[:,0],'y':lowd[:,1],'id':ids,'outlier':labs==-1})
    px.scatter(df, x='x', y='y', color='outlier', hover_data=['id'], title='UMAP Outliers').show()

def display_top_images(ds, scores, num=100):
    idxs = np.argsort(scores)[:num]
    fig, axes = plt.subplots(10, 10, figsize=(20,20))
    for ax, i in zip(axes.flatten(), idxs):
        img, _ = ds[i]
        arr = img.permute(1,2,0).cpu().numpy()
        ax.imshow(arr)
        ax.axis('off')
    plt.suptitle("Top 100 Outlier Images", fontsize=18)
    plt.tight_layout()
    plt.show()

# ───────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 1) Self-supervised dataloader
    ssl_ds = SSLDataset([linear_dir, log_dir], ssl_transform)
    ssl_loader = DataLoader(ssl_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                            num_workers=CONFIG["num_workers"])

    # 2) Inference dataloader with collate
    def infer_collate(batch):
        imgs, ids = zip(*batch)
        return torch.stack(imgs), list(ids)

    infer_ds = OEISDataset([linear_dir, log_dir], infer_transform)
    infer_loader = DataLoader(infer_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=CONFIG["num_workers"], collate_fn=infer_collate)

    # 3) Build model + projection head
    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    feat_dim = backbone.classifier[0].in_features
    out_dim  = backbone.classifier[0].out_features
    proj_head = ProjectionHead(in_dim=feat_dim+out_dim)
    embedder = CombinedEmbedder(backbone, proj_head).to(device)

    # 4) SSL pretraining
    self_supervised_pretrain(embedder, ssl_loader,
                             epochs=CONFIG["ssl_epochs"],
                             lr=CONFIG["ssl_lr"],
                             temp=CONFIG["ssl_temp"])

    # 5) Extract & cache embeddings
    cache_template = CONFIG["cache_template"]
    embeddings, ids = extract_and_cache_embeddings(embedder, infer_loader,
                                                   cache_template,
                                                   model_name="mobv3_ssl")

    # 6) Outlier detection & visualization
    labs, scores = detect_outliers(embeddings, CONFIG["contamination"])
    plot_umap(embeddings, ids, labs)
    display_top_images(infer_ds, scores, num=100)

    logger.info("Pipeline complete.")