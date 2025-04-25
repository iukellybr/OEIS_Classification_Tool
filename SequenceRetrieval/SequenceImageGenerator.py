import matplotlib
matplotlib.use('Agg')
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

#option to choose whether or not to rerun sequences which already have scatterplots
rerun_existing = False

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
            df = pd.read_csv(
                                file_path,
                                delim_whitespace=True,
                                comment='#',
                                header=None,
                                usecols=[0, 1],
                                names=['index', 'Sequence_Value'],
                                engine='python'
                            )


            if check_seq_type(df):
                seq_type = "log"
            else:
                seq_type = "linear"

            #remove file type extension and replace b with a so that scatterplot doesn't get saved as bXXXXXX.txt.png
            name_no_extension = os.path.splitext(bfile_name)[0]
            if name_no_ext.startswith("b"):
                name_no_ext = "a" + name_no_ext[1:]
            
            dict = {"name":name_no_extension,"seq":df,"seq_type":seq_type}
            all_sequences.append(dict)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return all_sequences

def create_sequence_image(seq,seq_name,seq_type):
    try:
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
    except Exception as e:
        print(f"Error generating image for {seq_name}: {e}")

def generate_sequence_images(seq_list):
    chunk_size = 5000

    for i in range(0, len(seq_list), chunk_size):
        print(f"processing chunk {i // chunk_size + 1} of {(len(seq_list)) // chunk_size}")

        batch = seq_list[i:i+chunk_size]

        all_seqs = read_bfiles(batch)

        for i in range(len(all_seqs)):
            seq_name = all_seqs[i]["name"]
            seq_type = all_seqs[i]["seq_type"]
            output_path = os.path.join(image_directory, seq_type, f"{seq_name}.png")
            if rerun_existing or not os.path.exists(output_path):
                create_sequence_image(all_seqs[i]["seq"],all_seqs[i]["name"],all_seqs[i]["seq_type"])

warnings.filterwarnings("ignore")

bfiles = sorted(os.listdir(bfile_directory))
generate_sequence_images(bfiles)
