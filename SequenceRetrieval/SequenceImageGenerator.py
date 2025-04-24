import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

save_directory = os.path.expanduser("~/Documents/OEIS_Sequence_Repository") 

#REQUIRED inputs: the bfiles must have been retrieved for this script to work
bfile_directory = os.path.join(save_directory, "B-Files")

#image output location 
image_directory = os.path.join(save_directory, "Sequence_Images")
os.makedirs(image_directory,exist_ok=True) 

#determines if sequence plot should use linear or log scaling
def check_seq_type(seq_df):
    #removes infinite values
    finite_mask = np.isfinite(seq_df["Sequence_Value"].astype(float))
    seq_df = seq_df.loc[finite_mask, :].reset_index(drop=True)
    
    logplot = True
    tval = 100.0
    
    if len(seq_df) > 10:
        fv = np.percentile(np.abs(seq_df["Sequence_Value"].astype(float)), [0, 25, 50, 75, 100])
        if np.all(np.isfinite(fv)):
            iqr = fv[3] - fv[1]  # interquartile range
            spread = fv[4] - fv[3]  # max - Q3
            base = fv[1] - fv[0] + 1  # Q1 - min + 1
            if (spread / base) < tval:
                logplot = False
    
    return logplot

def read_bfiles(seq_list):

    all_sequences = []
    for i in range(len(seq_list)):
        bfile_name = seq_list[i]
        file_path = os.path.join(bfile_directory, bfile_name)
        try:
            df = pd.read_csv(file_path,sep=' ',comment='#',header=None,index_col=False,names=['index','Sequence_Value'])

            if check_seq_type(df):
                seq_type = "log"
            else:
                seq_type = "linear"
            
            dict = {"name":bfile_name,"seq":df,"seq_type":seq_type}
            all_sequences.append(dict)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return all_sequences

def create_sequence_image(seq,seq_name,seq_type):
    fig, ax = plt.subplots()
    
    if seq_type == 'linear':
        ax.scatter(x=seq['index'],y=seq['Sequence_Value'],c='black',s=1.25,marker="+")
        fig.set_size_inches(9.799307958,6)
    elif seq_type == 'log':
        if (seq['Sequence_Value'].astype(float) < 0).any():
            ax.scatter(x=seq['index'],y=seq['Sequence_Value'].astype(float).apply(np.arcsinh),c='black',s=1.25,marker="+")
        else:
            ax.scatter(x=seq['index'],y=seq['Sequence_Value'].astype(float).apply(np.abs).apply(np.log10),c='black',s=1.25,marker="+")
        fig.set_size_inches(9.965397924,6)
    
    ax.axis('off')

    output_dir = os.path.join(image_directory,seq_type)
    os.makedirs(output_dir,exist_ok=True) 
    
    plt.savefig(f'{output_dir}/{seq_name}.png',bbox_inches='tight')
    plt.close()

def generate_sequence_images(seq_list):
    all_seqs = read_bfiles(seq_list)

    for i in range(len(all_seqs)):
        create_sequence_image(all_seqs[i]["seq"],all_seqs[i]["name"],all_seqs[i]["seq_type"])

warnings.filterwarnings("ignore")

bfiles = os.listdir(bfile_directory)
generate_sequence_images(bfiles)
