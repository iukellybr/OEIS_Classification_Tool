import matplotlib.pyplot as plt
import os
from os import walk
import pandas as pd

save_directory = os.path.expanduser("~/Documents/OEIS_Sequence_Repository") 
lin_directory = os.path.join(save_directory, "Sequence_Samples/linear")
log_directory = os.path.join(save_directory, "Sequence_Samples/log")
image_directory = os.path.join(save_directory, "Sequence_Images")

os.makedirs(image_directory,exist_ok=True) 

def read_sequences(root_folder):
    # List to store all dataframes
    all_dataframes = []
    
    # Walk through all subdirectories and files
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                try:
                    df = pd.read_csv(file_path).reset_index()
                    dict = {"name":filename,"seq":df}
                    all_dataframes.append(dict)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    print(f"Loaded {len(all_dataframes)} files")
    return all_dataframes

def create_sequence_image(seq,seq_name,seq_type):
    fig, ax = plt.subplots()
    
    if seq_type == 'linear':
        ax.scatter(x=seq['index'],y=seq['Sequence_Value'],c='black',s=2)
    elif seq_type == 'log':
        ax.scatter(x=seq['index'],y=seq['Sequence_Value'],c='black',s=2)
        ax.set_yscale('symlog')
    
    ax.axis('off')

    plt.savefig(f'{image_directory}/{seq_name}.png')
    plt.close()

def generate_sequence_images(root_folder,seq_type):
    all_seqs = read_sequences(root_folder)

    for i in range(len(all_seqs)):
        create_sequence_image(all_seqs[i]["seq"],all_seqs[i]["name"],seq_type)

generate_sequence_images(lin_directory,"linear")

generate_sequence_images(log_directory,"log")