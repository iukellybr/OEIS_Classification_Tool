import streamlit as st
import os
import tkinter as tk
from tkinter import filedialog
from streamlit_file_browser import st_file_browser

st.set_page_config(page_title="OEIS Sequence Clustering", layout="wide")

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
                "All sequences up to a specified ID (Do not include 'A')",
                "Random selection",
                "Range of sequences (Do not include 'A')",
                "Every N-th sequence"
            ],
            index=0
        )
        if method == "All sequences up to a specified ID (Do not include 'A')":
            st.text_input("Up To Sequence ID:", value='382000', key="up_to_id")
        elif method == "Random selection":
            st.text_input("Number of Sequences:", value='5000', key="num_random")
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
    
    # run with selected options
    def run_ui():
        if st.button("Run"):
            st.success("Run clicked! (Functionality pending)")
    
    section_title("Algorithm Selection", algorithm_ui)
    st.markdown("<hr style='border: 0.5px solid rgba(0, 0, 0, 0.3); margin: 12px 0;'>", unsafe_allow_html=True)
    section_title("Analysis and Insights", insights_ui)
    st.markdown("<hr style='border: 0.5px solid rgba(0, 0, 0, 0.3); margin: 12px 0;'>", unsafe_allow_html=True)
    section_title("Run", run_ui)
