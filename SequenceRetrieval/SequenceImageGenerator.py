import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

save_directory = os.path.expanduser("~/Documents/OEIS_Sequence_Repository") 

#REQUIRED inputs: the index and bfiles must have been retrieved for this script to work
index_directory = os.path.join(save_directory, "Sequence_Index.csv")
bfile_directory = os.path.join(save_directory, "B-Files")

#image output location 
image_directory = os.path.join(save_directory, "Sequence_Images")
os.makedirs(image_directory,exist_ok=True) 


#retrieves list of linear and log sequences
index = pd.read_csv(index_directory)
linear_index = index[index['Scatterplot Type'] == 'Linear']['Sequence ID'].reset_index(drop=True)
log_index = index[index['Scatterplot Type'] == 'Logarithmic']['Sequence ID'].reset_index(drop=True)

def read_bfiles(index):
    all_sequences = []
    for i in range(len(index)):
        bfile_name = index[i] + '.txt'
        file_path = os.path.join(bfile_directory, bfile_name)
        try:
            df = pd.read_csv(file_path,sep=' ',comment='#',header=None,index_col=False,names=['index','Sequence_Value'])
            dict = {"name":bfile_name,"seq":df}
            all_sequences.append(dict)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return all_sequences

def create_sequence_image(seq,seq_name,seq_type):
    fig, ax = plt.subplots()
    
    if seq_type == 'linear':
        ax.scatter(x=seq['index'],y=seq['Sequence_Value'],c='black',s=1.25)
        fig.set_size_inches(9.799307958,6)
    elif seq_type == 'log':
        ax.scatter(x=seq['index'],y=seq['Sequence_Value'].astype(float).apply(np.log10),c='black',s=1.25)
        #ax.set_yscale('symlog')
        fig.set_size_inches(9.965397924,6)
    
    ax.axis('off')

    plt.savefig(f'{image_directory}/{seq_name}.png',bbox_inches='tight')
    plt.close()

def generate_sequence_images(index,seq_type):
    all_seqs = read_bfiles(index)

    for i in range(len(all_seqs)):
        create_sequence_image(all_seqs[i]["seq"],all_seqs[i]["name"],seq_type)

#produces images for linear sequences
generate_sequence_images(linear_index,"linear")

#produces images for log sequences
warnings.filterwarnings("ignore")
generate_sequence_images(log_index,"log")