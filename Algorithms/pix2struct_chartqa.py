import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import logging
import h5py
from sklearn.ensemble import IsolationForest
import hdbscan
from sklearn.metrics import silhouette_score
import umap
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from transformers import Pix2StructImageProcessor, Pix2StructVisionModel
import warnings

# Suppress specific FutureWarning from scikit-learn
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*"
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "batch_size": 8,
    "num_workers": 0,
    "base_directory": os.path.expanduser("~/Documents/OEIS_Sequence_Repository"),
    "feature_cache": "./features_{model_name}.h5",  # Updated to include model name
    "seed": 42,
    "contamination": 0.05,
}

# Set random seeds
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

# Set device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA (GPU) device")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS (Apple Silicon) device")
else:
    device = torch.device("cpu")
    logger.info("Using CPU device")

# Define directories
linear_directory = os.path.join(CONFIG["base_directory"], "Linear_Scatterplots")
log_directory = os.path.join(CONFIG["base_directory"], "Logarithmic_Scatterplots")

# Dataset
class OEISDataset(Dataset):
    def __init__(self, linear_dir, log_dir, transform=None):
        self.transform = transform
        self.images = []
        self.sequence_ids = []
        self.linear_count = 0
        self.logarithmic_count = 0
        
        for directory in [linear_dir, log_dir]:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                continue
            is_linear = (directory == linear_dir)
            for file in os.listdir(directory):
                if file.endswith(".png"):
                    self.images.append(os.path.join(directory, file))
                    self.sequence_ids.append(os.path.splitext(file)[0])
                    if is_linear:
                        self.linear_count += 1
                    else:
                        self.logarithmic_count += 1
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            return None, self.sequence_ids[idx]
        sequence_id = self.sequence_ids[idx]
        if self.transform:
            image = self.transform(image)
        return image, sequence_id

class OEISDatasetOriginal(Dataset):
    def __init__(self, linear_dir, log_dir):
        self.images = []
        self.sequence_ids = []
        self.linear_count = 0
        self.logarithmic_count = 0
        
        for directory in [linear_dir, log_dir]:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                continue
            is_linear = (directory == linear_dir)
            for file in os.listdir(directory):
                if file.endswith(".png"):
                    self.images.append(os.path.join(directory, file))
                    self.sequence_ids.append(os.path.splitext(file)[0])
                    if is_linear:
                        self.linear_count += 1
                    else:
                        self.logarithmic_count += 1
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            return None, self.sequence_ids[idx]
        sequence_id = self.sequence_ids[idx]
        return image, sequence_id

# Custom collate function to handle PIL images
def custom_collate(batch):
    images = [item[0] for item in batch]  # List of PIL.Image.Image or None
    sequence_ids = [item[1] for item in batch]  # List of strings
    return images, sequence_ids

# Feature extraction with model-specific cache and dummy header_text
def extract_embeddings_chunked(model, processor, data_loader, device, cache_file, chunk_size=10000):
    if os.path.exists(cache_file):
        logger.info(f"Loading cached features from {cache_file}")
        with h5py.File(cache_file, 'r') as f:
            embeddings = f['embeddings'][:]
            sequence_ids = f['sequence_ids'][:].astype(str).tolist()
        return embeddings, sequence_ids

    logger.info(f"No cached features found at {cache_file}. Extracting new embeddings...")
    all_embeddings = []
    all_sequence_ids = []
    for i, batch in enumerate(tqdm(data_loader, desc="Extracting embeddings")):
        images, sequence_ids = batch
        valid_indices = [j for j, img in enumerate(images) if img is not None]
        valid_images = [images[j] for j in valid_indices]
        valid_ids = [sequence_ids[j] for j in valid_indices]
        if not valid_images:
            continue
        # Provide dummy header_text to satisfy the processor
        header_text = [""] * len(valid_images)
        inputs = processor(images=valid_images, header_text=header_text, return_tensors="pt")
        # Extract flattened_patches and move to device
        flattened_patches = inputs['flattened_patches'].to(device)
        with torch.no_grad():
            outputs = model(flattened_patches)
        last_hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, 768)
        batch_embeddings = last_hidden_states.mean(dim=1).cpu().numpy()  # Shape: (batch_size, 768)
        all_embeddings.append(batch_embeddings)
        all_sequence_ids.extend(valid_ids)
        if len(all_sequence_ids) >= chunk_size:
            embeddings_array = np.vstack(all_embeddings)
            sequence_ids_array = np.array(all_sequence_ids, dtype='S')
            with h5py.File(cache_file, 'a') as f:
                if 'embeddings' not in f:
                    f.create_dataset('embeddings', data=embeddings_array, maxshape=(None, embeddings_array.shape[1]))
                    f.create_dataset('sequence_ids', data=sequence_ids_array, maxshape=(None,))
                else:
                    current_size = f['embeddings'].shape[0]
                    f['embeddings'].resize(current_size + embeddings_array.shape[0], axis=0)
                    f['embeddings'][current_size:] = embeddings_array
                    f['sequence_ids'].resize(current_size + len(sequence_ids_array), axis=0)
                    f['sequence_ids'][current_size:] = sequence_ids_array
            all_embeddings = []
            all_sequence_ids = []
    if all_embeddings:
        embeddings_array = np.vstack(all_embeddings)
        sequence_ids_array = np.array(all_sequence_ids, dtype='S')
        with h5py.File(cache_file, 'a') as f:
            if 'embeddings' not in f:
                f.create_dataset('embeddings', data=embeddings_array, maxshape=(None, embeddings_array.shape[1]))
                f.create_dataset('sequence_ids', data=sequence_ids_array, maxshape=(None,))
            else:
                current_size = f['embeddings'].shape[0]
                f['embeddings'].resize(current_size + embeddings_array.shape[0], axis=0)
                f['embeddings'][current_size:] = embeddings_array
                f['sequence_ids'].resize(current_size + len(sequence_ids_array), axis=0)
                f['sequence_ids'][current_size:] = sequence_ids_array
    with h5py.File(cache_file, 'r') as f:
        embeddings = f['embeddings'][:]
        sequence_ids = f['sequence_ids'][:].astype(str).tolist()
    logger.info(f"Embeddings saved to {cache_file}")
    return embeddings, sequence_ids

# Visualization functions
def display_sample_images(dataset, num_images=50):
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    selected_images = []
    selected_ids = []
    for idx in indices:
        img, seq_id = dataset[idx]
        if img is not None:
            selected_images.append(img)
            selected_ids.append(seq_id)
    rows, cols = 10, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 30))
    for i, (img, seq_id) in enumerate(zip(selected_images, selected_ids)):
        if i < rows * cols:
            axes.flatten()[i].imshow(img)
            axes.flatten()[i].set_title(f"{seq_id}", fontsize=8)
            axes.flatten()[i].axis("off")
    for i in range(len(selected_images), rows * cols):
        axes.flatten()[i].axis("off")
    plt.tight_layout()
    plt.show()

def detect_outliers(embeddings, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    labels = iso_forest.fit_predict(embeddings)
    scores = iso_forest.decision_function(embeddings)
    return (labels + 1) // 2, scores

def select_diverse_cases(embeddings, indices, num_cases=100, distance_threshold=0.5):
    selected_indices = []
    for idx in indices:
        if len(selected_indices) < num_cases:
            if all(np.linalg.norm(embeddings[idx] - embeddings[si]) > distance_threshold for si in selected_indices):
                selected_indices.append(idx)
    return selected_indices

def visualize_embeddings_plotly(embeddings, sequence_ids, outlier_labels, title="Embedding Visualization"):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, n_jobs=-1)
    embeddings_2d = reducer.fit_transform(embeddings)
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'sequence_id': sequence_ids,
        'is_outlier': outlier_labels == 1
    })
    fig = px.scatter(df, x='x', y='y', color='is_outlier', hover_data=['sequence_id'], title=title)
    fig.show()

def plot_outlier_scores(scores):
    plt.hist(scores, bins=50)
    plt.title("Distribution of Isolation Forest Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()

def visualize_clusters(embeddings, sequence_ids, min_cluster_size=100):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1, n_jobs=-1)
    embeddings_2d = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5)
    cluster_labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logger.info(f"HDBSCAN found {n_clusters} clusters")
    if n_clusters > 1:
        valid_indices = cluster_labels != -1
        if np.sum(valid_indices) > 1:
            silhouette_avg = silhouette_score(embeddings[valid_indices], cluster_labels[valid_indices])
            logger.info(f"Silhouette Score (excluding noise): {silhouette_avg:.4f}")
        else:
            logger.info("Not enough non-noise points for silhouette score")
    else:
        logger.info("No clusters or single cluster found; skipping silhouette score")
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'sequence_id': sequence_ids,
        'cluster': cluster_labels
    })
    fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=['sequence_id'], title="HDBSCAN Cluster Visualization")
    fig.show()

def display_top_cases_images_original(dataset, top_indices, num_images=100):
    selected_indices = random.sample(list(top_indices), min(num_images, len(top_indices)))
    selected_images = []
    selected_ids = []
    for idx in selected_indices:
        img, seq_id = dataset[idx]
        if img is not None:
            selected_images.append(img)
            selected_ids.append(seq_id)
    rows, cols = 10, 10
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    for i, (img, seq_id) in enumerate(zip(selected_images, selected_ids)):
        if i < rows * cols:
            axes.flatten()[i].imshow(img)
            axes.flatten()[i].set_title(f"{seq_id}", fontsize=8)
            axes.flatten()[i].axis("off")
    for i in range(len(selected_images), rows * cols):
        axes.flatten()[i].axis("off")
    plt.suptitle("Top 100 Most Interesting Cases (Original Images)", fontsize=16)
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    # Set model name (change this when switching models, e.g., "mobilenet")
    model_name = "pix2struct"

    # Compute model-specific cache file
    cache_file = CONFIG["feature_cache"].format(model_name=model_name)

    # Load dataset
    dataset = OEISDataset(linear_directory, log_directory, transform=None)
    if len(dataset) == 0:
        logger.error("No images found in the specified directories. Exiting.")
        exit(1)
    
    # Log image counts
    logger.info(f"Loaded {len(dataset)} images: {dataset.linear_count} linear, {dataset.logarithmic_count} logarithmic")
    
    # Create DataLoader with custom collate function
    data_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], collate_fn=custom_collate)
    original_dataset = OEISDatasetOriginal(linear_directory, log_directory)

    # Load pretrained model and processor
    processor = Pix2StructImageProcessor.from_pretrained("google/pix2struct-chartqa-base")
    model = Pix2StructVisionModel.from_pretrained("google/pix2struct-chartqa-base").to(device)
    model.eval()

    # Display sample images
    logger.info("Displaying 50 random sample images...")
    display_sample_images(dataset)

    # Extract embeddings using model-specific cache
    logger.info("Extracting embeddings...")
    all_embeddings, all_sequence_ids = extract_embeddings_chunked(model, processor, data_loader, device, cache_file)

    # Outlier detection
    logger.info("Detecting outliers...")
    iso_labels, iso_scores = detect_outliers(all_embeddings, CONFIG["contamination"])
    plot_outlier_scores(iso_scores)

    # Select top cases
    top_indices = np.argsort(iso_scores)[:100]
    top_cases = np.zeros(len(all_embeddings), dtype=int)
    top_cases[top_indices] = 1
    top_indices = select_diverse_cases(all_embeddings, top_indices, num_cases=100, distance_threshold=0.5)
    top_cases = np.zeros(len(all_embeddings), dtype=int)
    top_cases[top_indices] = 1

    # Visualizations
    logger.info("Generating visualizations...")
    visualize_embeddings_plotly(all_embeddings, all_sequence_ids, top_cases, "Top 100 Outliers (Isolation Forest)")
    visualize_clusters(all_embeddings, all_sequence_ids, min_cluster_size=100)
    display_top_cases_images_original(original_dataset, top_indices)

    logger.info(f"Top 100 Outliers Identified: {', '.join([all_sequence_ids[i] for i in top_indices[:5]])} (first 5 shown)")
    logger.info(f"Total Outliers (Isolation Forest): {np.sum(iso_labels == 0)}")