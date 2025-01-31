import streamlit as st
import pandas as pd

# Load dataset
csv_file = 'your_dataset.csv'  # Replace with your dataset file path

@st.cache_data
def load_data():
    dataset = pd.read_csv(csv_file)
    dataset.columns = dataset.columns.str.strip()  # Clean column names
    return dataset

dataset = load_data()

# Rename and select columns
columns_to_display = {
    "Node": "Node",
    "NearestARGDistance": "Nearest ARG Distance",
    "AverageARGDistance": "Average ARG Distance",
    "CommunicationEfficiency": "Communication Efficiency",
    "PositiveTopologyCoefficient": "Positive Topology Coefficient",
    "Degree": "Degree",
    "ClusteringCoefficient": "Clustering Coefficient",
    "BetweennessCentrality": "Betweenness Centrality",
    "ClosenessCentrality": "Closeness Centrality",
    "Eccentricity": "Eccentricity",
    "NeighborhoodConnectivity": "Neighborhood Connectivity",
    "TopologicalCoefficient": "Topological Coefficient"
}
table_data = dataset[list(columns_to_display.keys())].rename(columns=columns_to_display)

# Remove rows where any column is missing
table_data = table_data.dropna(subset=columns_to_display.values())

# Drop duplicate Nodes
table_data = table_data.drop_duplicates(subset=["Node"])

# Add new entry to dataset
st.sidebar.header("➕ Add New Entry")
new_row = {}
for col in columns_to_display.keys():
    new_row[col] = st.sidebar.text_input(col, "")

if st.sidebar.button("Add Row"):
    dataset = dataset.append(new_row, ignore_index=True)
    dataset.to_csv(csv_file, index=False)
    st.sidebar.success("✅ Row added successfully!")

# Download updated dataset
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="📥 Download dataset as CSV",
    data=convert_df(dataset),
    file_name="updated_dataset.csv",
    mime="text/csv",
)

# Display dataset
st.markdown("<h1 style='text-align: center; color: #3498db;'>📊 Dataset Overview</h1>", unsafe_allow_html=True)

# Table with improved style
st.markdown("""
    <style>
    .streamlit-table th {
        background-color: #2980b9;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .streamlit-table td {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.dataframe(table_data, use_container_width=True)

# Sidebar Styling: Custom CSS to make sidebar vibrant
st.markdown(
    """
    <style>
    .css-1d391kg { 
        background: linear-gradient(to right, #3498db, #9b59b6); 
        color: white;
    }
    .css-1r1gzwg { 
        color: white;
    }
    .css-5xo9mb {
        background-color: #3498db;
        color: white;
    }
    .css-v2ww9j { 
        color: #ecf0f1; 
    }
    .css-4y75v5 {
        color: #ecf0f1;
    }
    </style>
    """, unsafe_allow_html=True
)

# Add animations for the buttons and inputs for interactivity
st.sidebar.markdown("""
    <style>
    .stButton>button {
        background-color: #2980b9;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid #2980b9;
    }
    .stButton>button:hover {
        background-color: #3498db;
    }
    .stDownloadButton>button {
        background-color: #9b59b6;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid #9b59b6;
    }
    .stDownloadButton>button:hover {
        background-color: #8e44ad;
    }
    </style>
""", unsafe_allow_html=True)

# Add a footer with a custom message
st.markdown(
    "<footer style='text-align: center; font-size: 14px; color: gray;'>Developed by Your Name | Your Organization</footer>",
    unsafe_allow_html=True
)
