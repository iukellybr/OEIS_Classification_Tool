import os
import pandas as pd

save_directory = os.path.expanduser("~/Documents/OEIS_Sequence_Repository") #should this be a parameter?
index_directory = os.path.join(save_directory, "Sequence_Index.csv")
bfile_directory = os.path.join(save_directory, "B-Files")
sample_directory = os.path.join(save_directory, "Sequence_Samples") #output location for split sequences

#REQUIREMENT: the index needs to be populated in order for the sequences to be split into linear and log groups
index = pd.read_csv(index_directory)

linear_index = index[index['Scatterplot Type'] == 'Linear']['Sequence ID'].reset_index(drop=True)
log_index = index[index['Scatterplot Type'] == 'Logarithmic']['Sequence ID'].reset_index(drop=True)

#REQUIREMENT: the bfiles need to have been retrieved in order to have seqence values to split
available_sequence_files = os.listdir(bfile_directory)
available_sequences_ids = [seq.split('.')[0] for seq in available_sequence_files]

sequence_id_df = pd.DataFrame(available_sequences_ids,columns=['Sequence ID'])

lin_sequences_to_sample = pd.merge(sequence_id_df,linear_index,on='Sequence ID')['Sequence ID']
log_sequences_to_sample = pd.merge(sequence_id_df,log_index,on='Sequence ID')['Sequence ID']

def read_bfile(seq_name):
    bfile_name = seq_name + '.txt'
    bfile_path = os.path.join(bfile_directory,bfile_name)
    bfile =  pd.read_csv(bfile_path,sep=' ',comment='#',header=None)
    sequence_integers = bfile[1]
    return(sequence_integers)

def seq_split(sequence, sample_type, n):
    if sample_type == 'n_splits':
        k, m = divmod(len(sequence), n)
        return (sequence[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    elif sample_type == 'splits_of_size_n':
        return [sequence[i:i + n] for i in range(0, len(sequence), n)]
    elif sample_type == 'every_nth':
        x = sequence[::n]
        return x
    
def save_sample(seq_samples,seq_name,seq_type,sample_type,n):
    os.makedirs(f'{sample_directory}/{seq_type}/{sample_type}',exist_ok=True)

    if sample_type == 'every_nth':
        df = pd.DataFrame(seq_samples,columns=['Sequence_Value'])
        df.to_csv(f'{sample_directory}/{seq_type}/{sample_type}/{seq_name}_every_{n}_values.txt',index=False)
    
    for i in range(len(seq_samples)):

        if sample_type == 'n_splits':
            df = pd.DataFrame(list(seq_samples[i]),columns=['Sequence_Value'])
            df.to_csv(f'{sample_directory}/{seq_type}/{sample_type}/{seq_name}_n_splits_sample_{i+1}_of_{n}.txt',index=False)
        elif sample_type == 'splits_of_size_n':
            df = pd.DataFrame(list(seq_samples[i]),columns=['Sequence_Value'])
            df.to_csv(f'{sample_directory}/{seq_type}/{sample_type}/{seq_name}_even_splits_of_size_{n}_sample_{i+1}.txt',index=False)

#TODO: these will need to pull their values from the streamlit app selections
#Sample types currently supported include splitting a sequence into n equally sized samples ("n_splits"), 
# splitting a sequence into samples of size n ("splits_of_size_n"), and grabbing every nth value of a 
# sequence and outputting that subset ("every_nth").
n=100 
sample_type='splits_of_size_n'

def generate_samples(sequences_to_sample,seq_type,sample_type,n):
    for seq_name in sequences_to_sample:
        seq = read_bfile(seq_name)
        
        seq_samples = list(seq_split(seq,sample_type,n))
    
        save_sample(seq_samples,seq_name,seq_type,sample_type,n)

generate_samples(lin_sequences_to_sample,'linear',sample_type,n)
generate_samples(log_sequences_to_sample,'log',sample_type,n)