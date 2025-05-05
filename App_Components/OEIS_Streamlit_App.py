import streamlit as st
import os
import tkinter as tk
from tkinter import filedialog
from streamlit_file_browser import st_file_browser
# import classes to run app components
from SequenceRetriever import SequenceRetriever
from AlgorithmBuilder import OEIS_Algorithm_Builder
from AnalysisGenerator import OEIS_Analysis_Generator
from VisualizationBuilder import OEIS_Visualization_Builder

st.set_page_config(page_title="OEIS Sequence Clustering", layout="wide")

# settings mode toggle
settings_mode = st.radio(
    "Run Settings",
    options=["Default", "Custom"],
    index=0,
    horizontal=True
)

# retrieves sequence mode when running retriever
def get_sequence_mode(state):
    method = state.get("Choose sequence retrieval method:")
    if method == "All sequences up to a specified ID (Do not include 'A')":
        return "upto"
    elif method == "Random selection":
        return "random"
    elif method == "Range of sequences (Do not include 'A')":
        return "range"
    elif method == "Every N-th sequence":
        return "interval"
    return "random"

# runs with selected options
def run_ui():
    if st.button("Run"):
        # build retrieval params from session_state
        params = {
            'sequence_id_mode': get_sequence_mode(st.session_state),
            'num_random': int(st.session_state.get("num_random", "500") or 500), # TODO - dynamically add based on sequence ID mode
            'save_directory': os.path.expanduser("~/Documents/OEIS_Sequence_Repository"),
            'include_scatterplots': "Scatterplot Images" in st.session_state.filetypes,
            'include_pinplots': "Pinplot Images" in st.session_state.filetypes,
            'save_bfiles': "B-Files" in st.session_state.filetypes,
            'update_index': "Metadata/Index" in st.session_state.filetypes,
            'partition_sequences': st.session_state.get("enable_partitioning", False),
            'partition_count': int(st.session_state.get("split_parts") or 1), # TODO - dynamically only add if partition_sequences is true
            'partition_sample': int(st.session_state.get("first_parts") or 1) # TODO - dynamically only add if partition_sequences is true
        }

        # retrieval debug print
        st.write("Params for SequenceRetriever:", params)

        # hook to SequenceRetriever
        retriever = SequenceRetriever(params)
        retriever.execution_pipeline()

        st.success("Retrieval complete!")

if settings_mode == "Custom":
    col1, col2 = st.columns([1, 1], gap="large")

    # creates borders around sections
    def section_title(title, fn):
        with st.container():
            st.markdown(f"### {title}")
            fn()
            st.markdown("</div>", unsafe_allow_html=True)

    # left side
    with col1:
        # --- Sequence Retrieval Section ---
        def sequence_retrieval_ui():
            method = st.radio(
                "Choose sequence retrieval method:",
                [
                    "Random selection",
                    "All sequences up to a specified ID (Do not include 'A')",
                    "Range of sequences (Do not include 'A')",
                    "Every N-th sequence"
                ],
                index=0
            )
            if method == "Random selection":
                st.text_input("Number of Sequences:", value='500', key="num_random")
            elif method == "All sequences up to a specified ID (Do not include 'A')":
                st.text_input("Up To Sequence ID:", value='382000', key="up_to_id")
            elif method == "Range of sequences (Do not include 'A')":
                cols = st.columns(2)
                cols[0].text_input("From Sequence ID:", value='000001', key="range_from")
                cols[1].text_input("To Sequence ID:", value='005000', key="range_to")
            elif method == "Every N-th sequence":
                st.text_input("Interval:", value='100', key="every_n")
        
        # file retrieval selection
        def files_ui():
            st.multiselect(
                "Choose file types to retrieve:",
                ["B-Files", "Scatterplot Images", "Pinplot Images", "Metadata/Index"],
                default=["Scatterplot Images"],
                key="filetypes"
            )
        
        # partitioning section
        def partition_ui():
            enable = st.checkbox("Enable sequence partitioning", key="enable_partitioning")
            if enable:
                cols = st.columns(2)
                cols[0].text_input("Split into how many partitions?", key="split_parts")
                cols[1].text_input("Take first only first X partitions:", key="first_parts")
        
        section_title("Sequence Retrieval", sequence_retrieval_ui)
        st.markdown("<hr style='border: 0.5px solid rgba(0, 0, 0, 0.3); margin: 12px 0;'>", unsafe_allow_html=True)
        section_title("Files to Retrieve", files_ui)
        st.markdown("<hr style='border: 0.5px solid rgba(0, 0, 0, 0.3); margin: 12px 0;'>", unsafe_allow_html=True)
        section_title("Partitioning (Optional)", partition_ui)


    # right side
    with col2:
        # algorithm selection
        def algorithm_ui():
            st.selectbox("Select Algorithm:", ["Sequence Scatterplot Clustering"], key="algorithm")
            st.text_input("Min. Cluster Size:", value='2', key="cluster_size")
        
        # visualization options
        def insights_ui():
            st.multiselect(
                "Select visualizations and outputs to generate:",
                ["UMAP Visualizations", "Montage Visualizations", "Cluster Assignment CSVs"],
                key="visuals"
            )
        
        section_title("Algorithm Selection", algorithm_ui)
        st.markdown("<hr style='border: 0.5px solid rgba(0, 0, 0, 0.3); margin: 12px 0;'>", unsafe_allow_html=True)
        section_title("Analysis and Insights", insights_ui)
        st.markdown("<hr style='border: 0.5px solid rgba(0, 0, 0, 0.3); margin: 12px 0;'>", unsafe_allow_html=True)
        section_title("Run", run_ui)

else:
    # default session state variables
    st.session_state["filetypes"] = ["Scatterplot Images"]
    st.session_state["num_random"] = "500"
    st.session_state["enable_partitioning"] = False
    st.session_state["split_parts"] = "1"
    st.session_state["first_parts"] = "1"
    st.session_state["Choose sequence retrieval method:"] = "Random selection"

    # show Run button for "Default" state
    with st.container():
        st.markdown("<hr style='border: 0.5px solid rgba(0, 0, 0, 0.3); margin: 12px 0;'>", unsafe_allow_html=True)
        st.markdown("### Run")
        run_ui()