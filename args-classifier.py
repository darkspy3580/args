import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path
import numpy as np

# Define the column names and their descriptions
COLUMN_DESCRIPTIONS = {
    "Node": "Unique identifier for each node",
    "NearestARGDistance": "Distance to the nearest ARG",
    "AverageARGDistance": "Average distance to all ARGs",
    "CommunicationEfficiency": "Measure of communication efficiency",
    "PositiveTopologyCoefficient": "Positive topology coefficient value",
    "Degree": "Number of connections",
    "ClusteringCoefficient": "Measure of clustering",
    "BetweennessCentrality": "Betweenness centrality measure",
    "ClosenessCentrality": "Closeness centrality measure",
    "Eccentricity": "Maximum distance to other nodes",
    "NeighborhoodConnectivity": "Connectivity of neighboring nodes",
    "TopologicalCoefficient": "Topological coefficient value"
}

# Define feature extraction function
def extract_features(data):
    required_columns = [
        'NearestARGDistance', 'AverageARGDistance', 'CommunicationEfficiency',
        'PositiveTopologyCoefficient', 'Degree', 'ClusteringCoefficient',
        'BetweennessCentrality', 'ClosenessCentrality', 'Eccentricity',
        'NeighborhoodConnectivity', 'TopologicalCoefficient'
    ]
    # Check for missing columns before attempting extraction
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        st.error(f"Missing required columns in uploaded file: {missing}")
        return None
    return data[required_columns]

# Mobility Analyzer Class
class ARGMobilityAnalyzer:
    def __init__(self, data):
        self.arg_data = data
        self.prepare_mobility_features()

    def prepare_mobility_features(self):
        self.mobility_potential = self._calculate_mobility_potential()

    def _calculate_mobility_potential(self):
        # Check for required columns before calculation
        required_columns = ['CommunicationEfficiency', 'Degree', 'BetweennessCentrality', 
                           'ClusteringCoefficient', 'PositiveTopologyCoefficient', 'NeighborhoodConnectivity']
        
        missing = [col for col in required_columns if col not in self.arg_data.columns]
        if missing:
            st.error(f"Missing required columns for mobility analysis: {missing}")
            # Return default values if columns are missing
            return pd.Series([0.5] * len(self.arg_data))
        
        # Avoid division by zero errors
        max_degree = max(self.arg_data['Degree'].max(), 1)
        max_betweenness = max(self.arg_data['BetweennessCentrality'].max(), 1)
        max_neighborhood = max(self.arg_data['NeighborhoodConnectivity'].max(), 1)
        
        mobility_scores = (
            0.2 * self.arg_data['CommunicationEfficiency'] +
            0.15 * (self.arg_data['Degree'] / max_degree) +
            0.2 * (self.arg_data['BetweennessCentrality'] / max_betweenness) +
            0.15 * self.arg_data['ClusteringCoefficient'] +
            0.15 * self.arg_data['PositiveTopologyCoefficient'] +
            0.15 * (self.arg_data['NeighborhoodConnectivity'] / max_neighborhood)
        )
        
        # Handle case where all scores are the same (prevents division by zero)
        if mobility_scores.max() == mobility_scores.min():
            return pd.Series([0.5] * len(mobility_scores))
        
        return (mobility_scores - mobility_scores.min()) / (mobility_scores.max() - mobility_scores.min())

    def analyze_mobility(self):
        mobility_results = self.arg_data.copy()
        mobility_results['mobility_potential'] = self.mobility_potential
        mobility_results['mobility_category'] = mobility_results['mobility_potential'].apply(
            lambda x: '🟢 High Mobility' if x >= 0.7 else ('🟡 Moderate Mobility' if x >= 0.3 else '🔴 Low Mobility')
        )
        if 'Node' in self.arg_data.columns:
            mobility_results['Node'] = self.arg_data['Node']
        return mobility_results

def create_empty_dataframe():
    """Create an empty dataframe with the required columns"""
    return pd.DataFrame(columns=list(COLUMN_DESCRIPTIONS.keys()))

def validate_input(value, column):
    """Validate input values based on column type"""
    try:
        if column == "Node":
            return str(value)
        else:
            return float(value)
    except:
        st.error(f"Invalid input for {column}. Please enter a valid number.")
        return None

# Load the pre-trained model
model_path = Path(__file__).resolve().parent / "models" / "random_forest_model.pkl"

model = None
if model_path.exists():
    model = joblib.load(model_path)
else:
    st.warning("⚠️ Model file not found in the 'models' directory. ARG Classification will not work.")

# Streamlit App
def main():
    st.set_page_config(page_title="ARG Classifier and Mobility Analyzer", layout="wide")

    # Customizing background video
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        .main {
            background-color: transparent;
        }
        video {
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
        }
        .css-1v3fvcr {
            color: #fff !important;
        }
        .css-1n8z7r8 {
            background-color: #1e1e1e !important;
        }
        .css-1kyx3r0 {
            color: #ff5733;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 10px 24px;
            cursor: pointer;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stDownloadButton>button {
            background-color: #3498db;
            color: white;
            font-weight: bold;
            padding: 10px 24px;
            border-radius: 12px;
        }
        .stDownloadButton>button:hover {
            background-color: #2980b9;
        }
        </style>
        <video autoplay loop muted>
            <source src="static/background.mp4" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )

    # Title and Description
    st.title("🧬 ARG Classifier & Mobility Analyzer")
    st.markdown(
        """
        **🔍 Analyze Antibiotic Resistance Genes (ARG) !**  
        - 🧪 **ARG Classification**: Identify ARG or Non-ARG.
        - 🚀 **Mobility Analysis**: Evaluate ARG mobility potential and categories.
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for storing DataFrame
    if 'df' not in st.session_state:
        st.session_state.df = create_empty_dataframe()

    # Create tabs for organization with color customization
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset View", "📈 ARG Classification", "🔄 Mobility Analysis", "➕ Add New Data"
    ])

    with tab1:
        st.header("Dataset View")
        uploaded_file = st.file_uploader("📤 Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Check for required columns and show warnings if missing
                required_columns = list(COLUMN_DESCRIPTIONS.keys())
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.warning(f"⚠️ The uploaded file is missing these columns: {', '.join(missing_columns)}")
                    st.info("You can still view the data, but classification and mobility analysis may not work correctly.")
                
                st.session_state.df = df
                st.success("✅ File uploaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        if not st.session_state.df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Entries", len(st.session_state.df))
            with col2:
                st.metric("Columns", len(st.session_state.df.columns))
            with col3:
                st.metric("Missing Values", st.session_state.df.isna().sum().sum())

            st.subheader("Dataset Overview")
            selected_columns = st.multiselect(
                "Select columns to display",
                options=st.session_state.df.columns.tolist(),
                default=st.session_state.df.columns.tolist()
            )
            st.dataframe(
                st.session_state.df[selected_columns],
                use_container_width=True,
                height=400
            )

            if st.checkbox("Show Dataset Statistics"):
                st.subheader("Dataset Statistics")
                st.write(st.session_state.df.describe())

            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                "arg_data.csv",
                "text/csv",
                help="Download the current data as CSV file"
            )

            if st.button("🗑️ Clear Dataset"):
                st.session_state.df = create_empty_dataframe()
                st.success("✅ Dataset cleared!")
                st.experimental_rerun()

    with tab2:
        if st.session_state.df.empty:
            st.warning("⚠️ Please upload or add data first.")
        else:
            if model is None:
                st.error("❌ Model file is missing. ARG Classification cannot proceed.")
            else:
                st.header("ARG Classification Results")
                # Check if required columns exist before extraction
                features = extract_features(st.session_state.df)
                
                if features is not None:
                    # Only proceed with prediction if features are available
                    predictions = model.predict(features)
                    results_df = st.session_state.df.copy()
                    results_df['Predictions'] = pd.Series(predictions).map({1: '🟢 ARG', 0: '🔴 Non-ARG'})

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Ensure 'Node' column exists, if not use index
                        if 'Node' in results_df.columns:
                            display_df = results_df[["Node", "Predictions"]]
                        else:
                            display_df = pd.DataFrame({
                                "Index": range(len(results_df)),
                                "Predictions": results_df["Predictions"]
                            })
                        
                        st.dataframe(display_df, use_container_width=True)

                    with col2:
                        classification_counts = results_df['Predictions'].value_counts().reset_index()
                        classification_counts.columns = ['Category', 'Count']
                        fig = px.pie(
                            classification_counts,
                            names='Category',
                            values='Count',
                            title="ARG Classification Distribution",
                            color='Category',
                            color_discrete_map={'🟢 ARG': '#3498DB', '🔴 Non-ARG': '#E74C3C'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Classification Results",
                        csv,
                        "arg_predictions.csv",
                        "text/csv"
                    )
                else:
                    st.error("❌ Cannot perform classification due to missing required columns.")
                    st.info("Please ensure your dataset includes all required columns or add them in the 'Add New Data' tab.")

    with tab3:
        if st.session_state.df.empty:
            st.warning("⚠️ Please upload or add data first.")
        else:
            st.header("Mobility Analysis Results")
            
            # Check if required columns exist before analysis
            required_mobility_columns = ['CommunicationEfficiency', 'Degree', 'BetweennessCentrality', 
                                        'ClusteringCoefficient', 'PositiveTopologyCoefficient', 'NeighborhoodConnectivity']
            
            missing = [col for col in required_mobility_columns if col not in st.session_state.df.columns]
            
            if missing:
                st.error(f"❌ Cannot perform mobility analysis. Missing columns: {', '.join(missing)}")
                st.info("Please ensure your dataset includes all required columns or add them in the 'Add New Data' tab.")
            else:
                analyzer = ARGMobilityAnalyzer(st.session_state.df)
                results = analyzer.analyze_mobility()

                col1, col2 = st.columns([2, 1])
                with col1:
                    # Ensure 'Node' column exists
                    if 'Node' in results.columns:
                        display_df = results[["Node", "mobility_potential", "mobility_category"]]
                    else:
                        display_df = results[["mobility_potential", "mobility_category"]].copy()
                        display_df.insert(0, "Index", range(len(display_df)))
                    
                    st.dataframe(display_df, use_container_width=True)

                with col2:
                    mobility_counts = results['mobility_category'].value_counts().reset_index()
                    mobility_counts.columns = ['Mobility Category', 'Count']
                    fig = px.bar(
                        mobility_counts,
                        x='Mobility Category',
                        y='Count',
                        title="Mobility Category Distribution",
                        color='Mobility Category',
                        color_discrete_map={
                            '🔴 Low Mobility': '#2ECC71',
                            '🟡 Moderate Mobility': '#F1C40F',
                            '🟢 High Mobility': '#E74C3C'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Mobility Analysis",
                    csv,
                    "mobility_analysis.csv",
                    "text/csv"
                )

    with tab4:
        st.header("Add New Data")
        
        # Add a template download option
        template_df = pd.DataFrame(columns=list(COLUMN_DESCRIPTIONS.keys()))
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Template CSV",
            template_csv,
            "arg_template.csv",
            "text/csv",
            help="Download a template CSV with all required columns"
        )
        
        with st.form("add_data_form"):
            new_data = {}
            for col in COLUMN_DESCRIPTIONS.keys():
                new_data[col] = st.text_input(
                    f"{col}",
                    help=COLUMN_DESCRIPTIONS[col]
                )

            submitted = st.form_submit_button("Add Entry")
            if submitted:
                valid_data = {}
                all_valid = True
                for col, value in new_data.items():
                    if value:
                        validated_value = validate_input(value, col)
                        if validated_value is None:
                            all_valid = False
                            break
                        valid_data[col] = validated_value

                if all_valid and valid_data:
                    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([valid_data])], ignore_index=True)
                    st.success("✅ Entry added successfully!")
                else:
                    st.error("❌ Please make sure all fields are correctly filled.")

        new_entry_df = pd.DataFrame([new_data])
        st.download_button(
            label="📥 Download New Entry only as a CSV file ",
            data=new_entry_df.to_csv(index=False).encode('utf-8'),
            file_name="new_entry.csv",
            mime="text/csv",
            help="Download the new entry as a CSV file"
        )

if __name__ == '__main__':
    main()
