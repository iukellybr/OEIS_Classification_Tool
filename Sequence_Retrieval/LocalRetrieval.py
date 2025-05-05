import requests
import os
import cv2 # "pip install opencv-python"; this is a computer vision library
import random
import time 
import numpy as np
import pandas as pd
import easyocr # reads words in images
import concurrent.futures
from threading import Lock
from PIL import Image, ImageDraw, ImageFont

# PURPOSE OF THIS SCRIPT - allow you to locally run sequence retrieval via API call to the OEIS website
# note that this is not recommended at scale; we tested scaling this script and found that the OEIS will throttle the retrieval
# the OEIS team also suggested not to pursue this approach; this approach is useful to retrieve data for exploration with maybe a small sample of 100-1000 sequences
# at scale, you will need to directly retrieve b-files from the OEIS Github repo using git LFS

# Variables to control options
include_pinplots = False
update_index = False
save_bfiles = False

# Sequence ID mode - sets how you want to retrieve sequences ("random", "fixed", "random+fixed", or "all" are the options)
sequence_id_mode = 'all'

# number of random sequences to generate
num_random = 50

# Define a fixed sequence ID set - change as needed; will eventually need to be set of all sequences/large set of sequences
base_sequence_ids = ("A000002","A337014","A000003","A000004","A000005","A000006","A000007","A000010") # Change this as needed

# Time tracker
start_time = time.time()

# Define save directories
save_directory = os.path.expanduser("~/Documents/OEIS_Sequence_Repository")
linear_directory = os.path.join(save_directory, "Linear_Scatterplots")
log_directory = os.path.join(save_directory, "Logarithmic_Scatterplots")
pinplot_directory = os.path.join(save_directory, "Pinplots")
index_directory = os.path.join(save_directory, "Sequence_Index.csv")
bfile_directory = os.path.join(save_directory, "B-Files")

# Set tesseract directory
# To utilize pytesseract you'll need to install it from https://github.com/UB-Mannheim/tesseract/wiki
# Recommended to save for entire computer with default location so it is available in C:\Program Files
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# initialize ocr_reader for reading scatterplot titles
ocr_reader = easyocr.Reader(['en'])

# Ensure save directories exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
if not os.path.exists(linear_directory):
    os.makedirs(linear_directory)
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
if not os.path.exists(pinplot_directory) and include_pinplots:
    os.makedirs(pinplot_directory)
if not os.path.exists(bfile_directory) and save_bfiles:
    os.makedirs(bfile_directory)
if os.path.exists(index_directory) and update_index:
    index_df = pd.read_csv(index_directory)
elif not os.path.exists(index_directory) and update_index:
    index_df = pd.DataFrame(columns=["Sequence ID", "Description", "References", "First 10 Terms", 
                                     "Cross-References", "Keywords", "Author", "Scatterplot Type"])

# Function to generate a specified number of random sequence IDs in the format "A######"
def generate_random_sequence_ids(num_ids=100):
    return tuple(f"A{str(random.randint(1, 382000)).zfill(6)}" for _ in range(num_ids))

# Function to fetch OEIS sequence metadata
def fetch_oeis_metadata(sequence_id):
    url = f"https://oeis.org/search?q=id:{sequence_id}&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract sequence data
        seq_data = data[0]

        return {
            "Sequence ID": sequence_id,
            "Description": seq_data.get("name", "N/A"),
            "References": seq_data.get("references", "N/A"),
            "First 10 Terms": ", ".join(map(str, seq_data.get("data", [])[:10])),
            "Cross-References": seq_data.get("xref", "N/A"),
            "Keywords": seq_data.get("keyword", "N/A"),
            "Author": seq_data.get("author", "N/A"),
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata for {sequence_id}: {e}")
        return None

# Function to download B-file
def download_oeis_sequence_bfile(sequence_id, save_dir):
    base_url = "https://oeis.org"
    sequence_id_lower = sequence_id.lower()
    url = f"{base_url}/{sequence_id}/b{sequence_id_lower[1:]}.txt"
    file_path = os.path.join(save_dir, f"{sequence_id}.txt")
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Save the content to a file in the appropriate subdirectory
        with open(file_path, "w") as file:
            file.write(response.text)
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred for {sequence_id}: {http_err}")
    except Exception as err:
        print(f"An error occurred for {sequence_id}: {err}")

if sequence_id_mode == 'random+fixed':
    sequence_ids = base_sequence_ids + generate_random_sequence_ids(num_ids=num_random)
elif sequence_id_mode == 'fixed':
    sequence_ids = base_sequence_ids
elif sequence_id_mode == 'all':
    sequence_ids = [f"A{str(i).zfill(6)}" for i in range(1, 382001)]
else:
    sequence_ids = generate_random_sequence_ids(num_ids=num_random)

metadata_dict = {}
if update_index:
    # Initialize a Lock to enable global handling of the metadata dictionary in the process_sequence function
    # this global handling is used to classify each sequence as linear/logarithmic based on the scatterplot titles
    index_df_lock = Lock()
    for seq_id in sequence_ids:
        # gather metadata for sequence IDs
        metadata = fetch_oeis_metadata(seq_id)
        if metadata:
            metadata_dict[seq_id] = metadata

def process_sequence(sequence_id):
    try:
        url = f"https://oeis.org/{sequence_id}/graph?png=1"

        # Retrieve the graph image
        response = requests.get(url)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        else:
            print(f"Failed to download the graph for sequence {sequence_id}. Response - {response.status_code}")
            return

        if save_bfiles:
            download_oeis_sequence_bfile(sequence_id, bfile_directory)

        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        height, width, _ = image.shape

        if height > 1000:
            # image has 3 graphs - take bottom graph
            cutoff = height // 3
            graph_2 = image[2*cutoff:, :]
        else:
            # Define crop regions - the image contains two graphs stacked vertically
            midpoint = height // 2 

            # Select scatterplot image
            graph_2 = image[midpoint:, :] 

        # Define cropping margins to identify scatterplot title (linear or logarithmic); somewhat arbitrary but tested numbers to make the cropping operation work well
        title_crop_top = 5
        title_crop_bottom = 375
        title_crop_left = 167
        title_crop_right = 267

        # Crop and save scatterplot title
        graph_2_title = graph_2[title_crop_top:-title_crop_bottom, title_crop_left:-title_crop_right]

        # Identify title to be logarithmic or not
        # extracted_title = pytesseract.image_to_string(graph_2_title, config="--psm 6")
        ocr_results = ocr_reader.readtext(graph_2_title)
        extracted_title = " ".join([res[1] for res in ocr_results])

        # Logarithmic scatterplot graphs have 2 axes, which means there are slightly different cropping margins
        if "Logarithmic" in extracted_title:
            # Define cropping margins to remove axes; somewhat arbitrary but tested numbers to make the cropping operation work well
            scatterplot_type = 'Logarithmic'
            graph2_crop_top = 37
            graph2_crop_bottom = 74
            graph2_crop_left = 60
            graph2_crop_right = 60
            graph2_save_dir = log_directory
        else:
            # Define cropping margins to remove axes; somewhat arbitrary but tested numbers to make the cropping operation work well
            scatterplot_type = 'Linear'
            graph2_crop_top = 37
            graph2_crop_bottom = 74
            graph2_crop_left = 83
            graph2_crop_right = 45
            graph2_save_dir = linear_directory

        graph_2_cropped = graph_2[graph2_crop_top:-graph2_crop_bottom, graph2_crop_left:-graph2_crop_right]
        graph_2_path = os.path.join(graph2_save_dir, f"{sequence_id}.png")
        cv2.imwrite(graph_2_path, graph_2_cropped)

        # Save the cropped images
        if include_pinplots:
            if height > 1000:
                # 3 graphs in image; take graph from first third
                graph_1 = image[:cutoff, :]
            else:
                # 2 graphs only
                graph_1 = image[:midpoint, :]  # Pinplot graph on top of image

            # Define cropping margins to remove axes; somewhat arbitrary but tested numbers to make the cropping operation work well
            graph1_crop_top = 37
            graph1_crop_bottom = 74
            graph1_crop_left = 83
            graph1_crop_right = 45

            # Crop the inner graph areas to only include 
            graph_1_cropped = graph_1[graph1_crop_top:-graph1_crop_bottom, graph1_crop_left:-graph1_crop_right]
            graph_1_path = os.path.join(pinplot_directory, f"{sequence_id}.png")
            cv2.imwrite(graph_1_path, graph_1_cropped)

        if update_index:
            metadata = metadata_dict.get(sequence_id)
            if metadata:
                metadata["Scatterplot Type"] = scatterplot_type
                with index_df_lock:
                    global index_df
                    index_df = index_df.loc[index_df["Sequence ID"] != sequence_id]
                    index_df = pd.concat([index_df, pd.DataFrame([metadata])], ignore_index=True)

    except:
        print(f'''Ingestion failed for sequence {sequence_id}.''')

chunk_size = 10000
total_chunks = (len(sequence_ids) + chunk_size - 1) // chunk_size  # ceiling division

for i in range(0, len(sequence_ids), chunk_size):
    chunk_start_time = time.time()
    chunk = sequence_ids[i:i + chunk_size]
    chunk_number = (i // chunk_size) + 1

    print(f"\nStarting chunk {chunk_number} of {total_chunks} "
          f"({len(chunk)} sequences)")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_sequence, chunk)

    chunk_elapsed = round(time.time() - chunk_start_time, 2)
    print(f"\nFinished chunk {chunk_number} in {chunk_elapsed} seconds.")


if update_index:
    index_df.to_csv(index_directory, index=False)
    print(f'''Sequence index saved at {index_directory}''')

end_time = time.time()
elapsed_time = round(end_time - start_time, 2)
print(f"\nScript completed in {elapsed_time} seconds for {len(sequence_ids)} sequences. \nInclude pinplots: {include_pinplots} \nUpdate index: {update_index} \nSave B-files: {save_bfiles}")