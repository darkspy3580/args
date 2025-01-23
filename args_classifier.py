import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# Define feature extraction function
def extract_features(data):
    required_columns = [
        'NearestARGDistance', 'AverageARGDistance', 'CommunicationEfficiency',
        'PositiveTopologyCoefficient', 'Degree', 'ClusteringCoefficient',
        'BetweennessCentrality', 'ClosenessCentrality', 'Eccentricity',
        'NeighborhoodConnectivity', 'TopologicalCoefficient'
    ]
    # Ensure required columns are present
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        raise ValueError(f"Missing required columns in uploaded file: {missing}")
    return data[required_columns]

# Mobility Analyzer Class
class ARGMobilityAnalyzer:
    def __init__(self, data):
        self.arg_data = data
        self.prepare_mobility_features()

    def prepare_mobility_features(self):
        self.mobility_potential = self._calculate_mobility_potential()

    def _calculate_mobility_potential(self):
        mobility_scores = (
            0.2 * self.arg_data['CommunicationEfficiency'] +
            0.15 * (self.arg_data['Degree'] / self.arg_data['Degree'].max()) +
            0.2 * (self.arg_data['BetweennessCentrality'] /
                   self.arg_data['BetweennessCentrality'].max()) +
            0.15 * self.arg_data['ClusteringCoefficient'] +
            0.15 * self.arg_data['PositiveTopologyCoefficient'] +
            0.15 * (self.arg_data['NeighborhoodConnectivity'] /
                    self.arg_data['NeighborhoodConnectivity'].max())
        )
        return (mobility_scores - mobility_scores.min()) / (mobility_scores.max() - mobility_scores.min())

    def analyze_mobility(self):
        mobility_results = self.arg_data.copy()
        mobility_results['mobility_potential'] = self.mobility_potential
        mobility_results['mobility_category'] = mobility_results['mobility_potential'].apply(
            lambda x: '🟢 High Mobility' if x >= 0.7 else ('🟡 Moderate Mobility' if x >= 0.3 else '🔴 Low Mobility')
        )

        # Add Node column to the final result
        if 'Node' in self.arg_data.columns:
            mobility_results['Node'] = self.arg_data['Node']
        return mobility_results

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

    # Add the background video
    st.markdown(
        """
        <style>
        # MainMenu {visibility: hidden;}
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

    # Sidebar Options
    st.sidebar.title("🔧 **Options**")
    task = st.sidebar.radio("📂 Choose Task", ["ARG Classification", "Mobility Analysis"])
    st.sidebar.markdown(
        """
        🎯 **Instructions:**  
        1️⃣ Upload a **CSV file** with required columns.  
        2️⃣ Select your task.  
        3️⃣ View and download results!  
        """,
        unsafe_allow_html=True,
    )

    # File Uploader
    uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            input_data = pd.read_csv(uploaded_file)

            if task == "ARG Classification":
                if model is None:
                    st.error("❌ Model file is missing. ARG Classification cannot proceed.")
                    return

                # ARG Classification Task
                st.markdown("<div class='sub-header' style='color: #3498DB;'>📋 **ARG Classification Results**</div>", unsafe_allow_html=True)
                features = extract_features(input_data)
                predictions = model.predict(features)
                input_data['Predictions'] = pd.Series(predictions).map({1: '🟢 ARG', 0: '🔴 Non-ARG'})

                # Add Node column to ARG classification results
                if 'Node' in input_data.columns:
                    input_data['Node'] = input_data['Node']
                
                st.dataframe(input_data[["Node","Predictions"]])  # Display predictions + Node column

                # ARG Classification Distribution (Pie Chart)
                classification_counts = input_data['Predictions'].value_counts().reset_index()
                classification_counts.columns = ['Category', 'Count']
                fig = px.pie(
                    classification_counts, 
                    names='Category', 
                    values='Count', 
                    title="ARG Classification Distribution",
                    color='Category',
                    color_discrete_map={'🟢 ARG': '#3498DB', '🔴 Non-ARG': '#E74C3C'}
                )
                st.plotly_chart(fig)

                # Download Button
                csv = input_data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions", csv, "arg_predictions.csv", "text/csv")

            elif task == "Mobility Analysis":
                # Mobility Analysis Task
                st.markdown("<div class='sub-header' style='color: #2ECC71;'>⚙️ Mobility Analysis Results</div>", unsafe_allow_html=True)
                analyzer = ARGMobilityAnalyzer(input_data)
                results = analyzer.analyze_mobility()
                st.dataframe(results[["Node","mobility_potential", "mobility_category"]])  # Display Mobility + Node column

                # Mobility Category Distribution (Bar Chart)
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
                st.plotly_chart(fig)

                # Download Button
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Mobility Analysis", csv, "mobility_analysis.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    else:
        st.info("ℹ️ Please upload a CSV file to proceed.")

if __name__ == '__main__':
    main()
