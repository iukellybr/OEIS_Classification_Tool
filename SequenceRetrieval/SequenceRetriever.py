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

class SequenceRetriever():

    def __init__(self, params=None):
        # control file options/selections
        self.include_scatterplots = params.get('include_scatterplots',True)
        self.include_pinplots = params.get('include_pinplots',False)
        self.update_index = params.get('update_index',False)
        self.save_bfiles = params.get('save_bfiles',False)

        # sequence ID mode - sets how you want to retrieve sequences ("random", "upto", "interval", or "range" are the options)
        # random - selects X random sequences
        # upto - selects sequences up to sequence id X
        # interval - selects every Nth sequence
        # range - selects sequences starting at X and ending at Y
        self.sequence_id_mode = params.get('sequence_id_mode','random')

        # partitioning params
        # TODO - integrate Xander's partitioning logic
        self.partition_sequences = params.get('partition_sequences',False)
        self.partition_count = params.get('partition_count',1)
        self.partition_sample = params.get('partition_sample',self.partition_count)

        # time tracker
        self.start_time = time.time()

        # define save directories
        self.save_directory = os.path.expanduser(params.get('save_directory',"~/Documents/OEIS_Sequence_Repository"))
        self.linear_directory = os.path.join(self.save_directory, "Linear_Scatterplots")
        self.log_directory = os.path.join(self.save_directory, "Logarithmic_Scatterplots")
        self.pinplot_directory = os.path.join(self.save_directory, "Pinplots")
        self.index_directory = os.path.join(self.save_directory, "Sequence_Index.csv")
        self.bfile_directory = os.path.join(self.save_directory, "B-Files")

        # ensure save directories exist
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        if not os.path.exists(self.linear_directory):
            os.makedirs(self.linear_directory)
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
        if not os.path.exists(self.pinplot_directory) and self.include_pinplots:
            os.makedirs(self.pinplot_directory)
        if not os.path.exists(self.bfile_directory) and self.save_bfiles:
            os.makedirs(self.bfile_directory)
        if os.path.exists(self.index_directory) and self.update_index:
            self.index_df = pd.read_csv(self.index_directory)
        elif not os.path.exists(self.index_directory) and self.update_index:
            self.index_df = pd.DataFrame(columns=["Sequence ID", "Description", "References", "First 10 Terms", 
                                            "Cross-References", "Keywords", "Author", "Scatterplot Type"])
            
        # create sequence_ids tuple
        if self.sequence_id_mode == 'random':
            self.sequence_ids = self.generate_random_sequence_ids(num_ids=params.get('num_random',5000))
        elif self.sequence_id_mode == 'upto':
            raise NotImplementedError("Selected sequence mode not yet supported.")
            # self.sequence_ids = 
        elif self.sequence_id_mode == 'interval':
            raise NotImplementedError("Selected sequence mode not yet supported.")
            # self.sequence_ids = 
        elif self.sequence_id_mode == 'range':
            raise NotImplementedError("Selected sequence mode not yet supported.")
            # self.sequence_ids = 

        # initialize ocr reader for reading scatterplot titles
        self.ocr_reader = easyocr.Reader(['en'])
        
        if self.update_index:
            # initialize a metadata dictionary to store sequence metadata
            # also initialize a Lock to enable global handling of the metadata dictionary in the process_sequence function
            # this global handling is used to classify each sequence as linear/logarithmic based on the scatterplot titles
            self.metadata_dict = {}
            self.index_df_lock = Lock()
            for seq_id in self.sequence_ids:
                # gather metadata for sequence IDs
                metadata = self.fetch_oeis_metadata(seq_id)
                if metadata:
                    self.metadata_dict[seq_id] = metadata

    # Function to generate a specified number of random sequence IDs in the format "A######"
    def generate_random_sequence_ids(self, num_ids=1):
        return tuple(f"A{str(random.randint(1, 382000)).zfill(6)}" for _ in range(num_ids))

    # Function to fetch OEIS sequence metadata
    def fetch_oeis_metadata(self, sequence_id):
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
    def download_oeis_sequence_bfile(self, sequence_id, save_dir):
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

    def process_sequence(self, sequence_id):
        try:
            url = f"https://oeis.org/{sequence_id}/graph?png=1"

            # Retrieve the graph image
            response = requests.get(url)
            if response.status_code == 200:
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            else:
                print(f"Failed to download the graph for sequence {sequence_id}. Response - {response.status_code}")
                return

            if self.save_bfiles:
                self.download_oeis_sequence_bfile(sequence_id, self.bfile_directory)

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
            ocr_results = self.ocr_reader.readtext(graph_2_title)
            extracted_title = " ".join([res[1] for res in ocr_results])

            # Logarithmic scatterplot graphs have 2 axes, which means there are slightly different cropping margins
            if "Logarithmic" in extracted_title:
                # Define cropping margins to remove axes; somewhat arbitrary but tested numbers to make the cropping operation work well
                scatterplot_type = 'Logarithmic'
                graph2_crop_top = 37
                graph2_crop_bottom = 74
                graph2_crop_left = 60
                graph2_crop_right = 60
                graph2_save_dir = self.log_directory
            else:
                # Define cropping margins to remove axes; somewhat arbitrary but tested numbers to make the cropping operation work well
                scatterplot_type = 'Linear'
                graph2_crop_top = 37
                graph2_crop_bottom = 74
                graph2_crop_left = 83
                graph2_crop_right = 45
                graph2_save_dir = self.linear_directory

            graph_2_cropped = graph_2[graph2_crop_top:-graph2_crop_bottom, graph2_crop_left:-graph2_crop_right]
            graph_2_path = os.path.join(graph2_save_dir, f"{sequence_id}.png")
            cv2.imwrite(graph_2_path, graph_2_cropped)

            # Save the cropped images
            if self.include_pinplots:
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
                graph_1_path = os.path.join(self.pinplot_directory, f"{sequence_id}.png")
                cv2.imwrite(graph_1_path, graph_1_cropped)

            if self.update_index:
                metadata = self.metadata_dict.get(sequence_id)
                if metadata:
                    metadata["Scatterplot Type"] = scatterplot_type
                    with self.index_df_lock:
                        self.index_df = self.index_df.loc[self.index_df["Sequence ID"] != sequence_id]
                        self.index_df = pd.concat([self.index_df, pd.DataFrame([metadata])], ignore_index=True)

        except:
            print(f'''Ingestion failed for sequence {sequence_id}.''')

    def execution_pipeline(self):
        chunk_size = 10000
        total_chunks = (len(self.sequence_ids) + chunk_size - 1) // chunk_size  # ceiling division

        for i in range(0, len(self.sequence_ids), chunk_size):
            chunk_start_time = time.time()
            chunk = self.sequence_ids[i:i + chunk_size]
            chunk_number = (i // chunk_size) + 1

            print(f"\nStarting chunk {chunk_number} of {total_chunks} "
                f"({len(chunk)} sequences)")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(self.process_sequence, chunk)

            chunk_elapsed = round(time.time() - chunk_start_time, 2)
            print(f"\nFinished chunk {chunk_number} in {chunk_elapsed} seconds.")


        if self.update_index:
            self.index_df.to_csv(self.index_directory, index=False)
            print(f'''Sequence index saved at {self.index_directory}''')

        end_time = time.time()
        elapsed_time = round(end_time - self.start_time, 2)
        print(f"\nScript completed in {elapsed_time} seconds for {len(self.sequence_ids)} sequences. \nInclude pinplots: {self.include_pinplots} \nUpdate index: {self.update_index} \nSave B-files: {self.save_bfiles}")