import requests
import os
import cv2 # "pip install opencv-python"; this is a computer vision library
import random
import time 
import pandas as pd
import pytesseract # "pip install pytesseract"; reads words in images
from PIL import Image, ImageDraw, ImageFont

# variables to control options
include_pinplots = True
update_index = True
save_bfiles = True

# time tracker
start_time = time.time()

# Define save directories
save_directory = os.path.expanduser("~/Documents/OEIS_Sequence_Repository")
linear_directory = os.path.join(save_directory, "Linear_Scatterplots")
log_directory = os.path.join(save_directory, "Logarithmic_Scatterplots")
pinplot_directory = os.path.join(save_directory, "Pinplots")
index_directory = os.path.join(save_directory, "Sequence_Index.csv")
bfile_directory = os.path.join(save_directory, "B-Files")

# set tesseract directory
# to utilize pytesseract you'll need to install it from https://github.com/UB-Mannheim/tesseract/wiki
# recommended to save for entire computer with default location so it is available in C:\Program Files
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
def generate_random_sequence_ids(num_ids=20):
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
    # Ensure the sequence ID is in the format A000001, A000002, etc.
    base_url = "https://oeis.org"
    sequence_id_lower = sequence_id.lower()
    url = f"{base_url}/{sequence_id}/b{sequence_id_lower[1:]}.txt"
    
    # Extract the subdirectory name based on sequence ID (first 4 characters)
    # sub_dir = os.path.join(save_dir, sequence_id[:4])
    # os.makedirs(sub_dir, exist_ok=True)  # Ensure the subdirectory exists

    # Define the full file path
    # file_path = os.path.join(sub_dir, f"{sequence_id}.txt")
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

# Generate random sequence IDs
random_sequence_ids = generate_random_sequence_ids(num_ids=500)

# Define a fixed sequence ID set - change as needed; will eventually need to be set of all sequences/large set of sequences
# base_sequence_ids = ("A380713","A337014","A000045","A367562","A368400","A368427","A367726","A128272","A102920","A102837","A292834", "A057390") # Change this as needed - set to false if just want random sequences
base_sequence_ids = False

if base_sequence_ids:
    sequence_ids = base_sequence_ids + random_sequence_ids
else:
    sequence_ids = random_sequence_ids

for sequence_id in sequence_ids:
    try:
        url = f"https://oeis.org/{sequence_id}/graph?png=1"

        # Download the graph image
        image_path = os.path.join(save_directory, f"{sequence_id}.png")
        response = requests.get(url)
        if response.status_code == 200:
            # Temporarily saves the default image from the site, which is a combined image that includes both graphs
            with open(image_path, "wb") as file:
                file.write(response.content)
        else:
            print(f"Failed to download the graph for sequence {sequence_id}. Response - {response.status_code}")
            continue

        if save_bfiles:
            download_oeis_sequence_bfile(sequence_id, bfile_directory)

        # Load the image back using cv2
        image = cv2.imread(image_path)
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
        extracted_title = pytesseract.image_to_string(graph_2_title, config="--psm 6")

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
            # Retrieve metadata
            metadata = fetch_oeis_metadata(sequence_id)
            if metadata:
                metadata["Scatterplot Type"] = scatterplot_type

                # Update existing row or append new entry
                index_df = index_df.loc[index_df["Sequence ID"] != sequence_id]
                index_df = pd.concat([index_df, pd.DataFrame([metadata])], ignore_index=True)

        # Delete the original downloaded image and title
        os.remove(image_path)
    except:
        print(f'''Ingestion failed for sequence {sequence_id}.''')
        continue

if update_index:
    index_df.to_csv(index_directory, index=False)
    print(f'''Sequence index saved at {index_directory}''')

end_time = time.time()
elapsed_time = round(end_time - start_time, 2)
print(f"\nScript completed in {elapsed_time} seconds for {len(sequence_ids)} sequences. \nInclude pinplots: {include_pinplots} \nUpdate index: {update_index} \nSave B-files: {save_bfiles}")