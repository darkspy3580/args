import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
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

# Weight justification for mobility scoring
WEIGHT_JUSTIFICATION = {
    "CommunicationEfficiency": {
        "weight": 0.20,
        "rationale": "High communication efficiency indicates faster information/gene transfer potential across the network, making it a primary driver of mobility."
    },
    "Degree": {
        "weight": 0.15,
        "rationale": "Number of connections represents potential transmission pathways, but raw degree alone doesn't capture the quality of connections."
    },
    "BetweennessCentrality": {
        "weight": 0.20,
        "rationale": "Nodes with high betweenness act as bridges in the network, controlling gene flow between communities and facilitating horizontal gene transfer."
    },
    "ClusteringCoefficient": {
        "weight": 0.15,
        "rationale": "High clustering indicates dense local connections that can facilitate rapid local spread within communities."
    },
    "PositiveTopologyCoefficient": {
        "weight": 0.15,
        "rationale": "Reflects the structural favorability of a node's position for maintaining stable connections conducive to gene transfer."
    },
    "NeighborhoodConnectivity": {
        "weight": 0.15,
        "rationale": "Well-connected neighbors provide multiple pathways for gene dissemination, enhancing overall mobility potential."
    }
}

# Define feature extraction function
def extract_features(data):
    required_columns = [
        'NearestARGDistance', 'AverageARGDistance', 'CommunicationEfficiency',
        'PositiveTopologyCoefficient', 'Degree', 'ClusteringCoefficient',
        'BetweennessCentrality', 'ClosenessCentrality', 'Eccentricity',
        'NeighborhoodConnectivity', 'TopologicalCoefficient'
    ]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        st.error(f"Missing required columns in uploaded file: {missing}")
        return None
    return data[required_columns]

# Mobility Analyzer Class with Sensitivity Analysis
class ARGMobilityAnalyzer:
    def __init__(self, data, custom_weights=None):
        self.arg_data = data
        self.custom_weights = custom_weights if custom_weights else {
            'CommunicationEfficiency': 0.20,
            'Degree': 0.15,
            'BetweennessCentrality': 0.20,
            'ClusteringCoefficient': 0.15,
            'PositiveTopologyCoefficient': 0.15,
            'NeighborhoodConnectivity': 0.15
        }
        self.prepare_mobility_features()

    def prepare_mobility_features(self):
        self.mobility_potential = self._calculate_mobility_potential()

    def _calculate_mobility_potential(self):
        required_columns = ['CommunicationEfficiency', 'Degree', 'BetweennessCentrality', 
                           'ClusteringCoefficient', 'PositiveTopologyCoefficient', 'NeighborhoodConnectivity']
        
        missing = [col for col in required_columns if col not in self.arg_data.columns]
        if missing:
            st.error(f"Missing required columns for mobility analysis: {missing}")
            return pd.Series([0.5] * len(self.arg_data))
        
        max_degree = max(self.arg_data['Degree'].max(), 1)
        max_betweenness = max(self.arg_data['BetweennessCentrality'].max(), 1)
        max_neighborhood = max(self.arg_data['NeighborhoodConnectivity'].max(), 1)
        
        mobility_scores = (
            self.custom_weights['CommunicationEfficiency'] * self.arg_data['CommunicationEfficiency'] +
            self.custom_weights['Degree'] * (self.arg_data['Degree'] / max_degree) +
            self.custom_weights['BetweennessCentrality'] * (self.arg_data['BetweennessCentrality'] / max_betweenness) +
            self.custom_weights['ClusteringCoefficient'] * self.arg_data['ClusteringCoefficient'] +
            self.custom_weights['PositiveTopologyCoefficient'] * self.arg_data['PositiveTopologyCoefficient'] +
            self.custom_weights['NeighborhoodConnectivity'] * (self.arg_data['NeighborhoodConnectivity'] / max_neighborhood)
        )
        
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

    def perform_sensitivity_analysis(self, perturbation_range=0.1, num_iterations=10):
        """
        Perform sensitivity analysis by varying weights and measuring impact on results
        
        Parameters:
        - perturbation_range: How much to vary each weight (±)
        - num_iterations: Number of random weight combinations to test
        """
        base_results = self.analyze_mobility()
        base_categories = base_results['mobility_category'].value_counts()
        
        sensitivity_results = []
        
        # Test uniform weight distribution
        uniform_weights = {key: 1/6 for key in self.custom_weights.keys()}
        uniform_analyzer = ARGMobilityAnalyzer(self.arg_data, uniform_weights)
        uniform_results = uniform_analyzer.analyze_mobility()
        
        # Random perturbations
        for i in range(num_iterations):
            perturbed_weights = {}
            for key in self.custom_weights.keys():
                perturbation = np.random.uniform(-perturbation_range, perturbation_range)
                perturbed_weights[key] = max(0.05, self.custom_weights[key] + perturbation)
            
            # Normalize weights to sum to 1
            total = sum(perturbed_weights.values())
            perturbed_weights = {k: v/total for k, v in perturbed_weights.items()}
            
            perturbed_analyzer = ARGMobilityAnalyzer(self.arg_data, perturbed_weights)
            perturbed_results = perturbed_analyzer.analyze_mobility()
            
            # Calculate category changes
            category_changes = (base_results['mobility_category'] != perturbed_results['mobility_category']).sum()
            change_percentage = (category_changes / len(base_results)) * 100
            
            sensitivity_results.append({
                'iteration': i + 1,
                'category_changes': category_changes,
                'change_percentage': change_percentage,
                'weights': perturbed_weights.copy()
            })
        
        return {
            'base_results': base_results,
            'uniform_results': uniform_results,
            'sensitivity_results': sensitivity_results,
            'base_categories': base_categories
        }

def create_empty_dataframe():
    return pd.DataFrame(columns=list(COLUMN_DESCRIPTIONS.keys()))

def validate_input(value, column):
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

    # Styling
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 0rem; padding-bottom: 0rem;}
        .main {background-color: transparent;}
        video {position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%; z-index: -1;}
        .stButton>button {background-color: #4CAF50; color: white; font-weight: bold; border-radius: 12px; padding: 10px 24px; transition: 0.3s;}
        .stButton>button:hover {background-color: #45a049;}
        .stDownloadButton>button {background-color: #3498db; color: white; font-weight: bold; padding: 10px 24px; border-radius: 12px;}
        .stDownloadButton>button:hover {background-color: #2980b9;}
        </style>
        <video autoplay loop muted>
            <source src="static/background.mp4" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧬 ARG Classifier & Mobility Analyzer")
    st.markdown(
        """
        **🔍 Analyze Antibiotic Resistance Genes (ARG) !**  
        - 🧪 **ARG Classification**: Identify ARG or Non-ARG.
        - 🚀 **Mobility Analysis**: Evaluate ARG mobility potential and categories.
        - 📊 **Sensitivity Analysis**: Assess robustness of mobility scoring weights.
        """,
        unsafe_allow_html=True,
    )

    if 'df' not in st.session_state:
        st.session_state.df = create_empty_dataframe()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset View", "📈 ARG Classification", "🔄 Mobility Analysis", "🔬 Sensitivity Analysis", "➕ Add New Data"
    ])

    with tab1:
        st.header("Dataset View")
        uploaded_file = st.file_uploader("📤 Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
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
            st.dataframe(st.session_state.df[selected_columns], use_container_width=True, height=400)

            if st.checkbox("Show Dataset Statistics"):
                st.subheader("Dataset Statistics")
                st.write(st.session_state.df.describe())

            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "arg_data.csv", "text/csv")

            if st.button("🗑️ Clear Dataset"):
                st.session_state.df = create_empty_dataframe()
                st.success("✅ Dataset cleared!")
                st.rerun()

    with tab2:
        if st.session_state.df.empty:
            st.warning("⚠️ Please upload or add data first.")
        else:
            if model is None:
                st.error("❌ Model file is missing. ARG Classification cannot proceed.")
            else:
                st.header("ARG Classification Results")
                features = extract_features(st.session_state.df)
                
                if features is not None:
                    predictions = model.predict(features)
                    results_df = st.session_state.df.copy()
                    results_df['Predictions'] = pd.Series(predictions).map({1: '🟢 ARG', 0: '🔴 Non-ARG'})

                    col1, col2 = st.columns([2, 1])
                    with col1:
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
                        fig = px.pie(classification_counts, names='Category', values='Count',
                                   title="ARG Classification Distribution",
                                   color='Category',
                                   color_discrete_map={'🟢 ARG': '#3498DB', '🔴 Non-ARG': '#E74C3C'})
                        st.plotly_chart(fig, use_container_width=True)

                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Classification Results", csv, "arg_predictions.csv", "text/csv")

    with tab3:
        if st.session_state.df.empty:
            st.warning("⚠️ Please upload or add data first.")
        else:
            st.header("Mobility Analysis Results")
            
            required_mobility_columns = ['CommunicationEfficiency', 'Degree', 'BetweennessCentrality', 
                                        'ClusteringCoefficient', 'PositiveTopologyCoefficient', 'NeighborhoodConnectivity']
            
            missing = [col for col in required_mobility_columns if col not in st.session_state.df.columns]
            
            if missing:
                st.error(f"❌ Cannot perform mobility analysis. Missing columns: {', '.join(missing)}")
            else:
                # Display weight justification
                with st.expander("📖 View Weight Justification for Mobility Scoring"):
                    st.markdown("### Weight Rationale")
                    for feature, info in WEIGHT_JUSTIFICATION.items():
                        st.markdown(f"**{feature}** (Weight: {info['weight']:.2f})")
                        st.markdown(f"_{info['rationale']}_")
                        st.markdown("---")
                
                analyzer = ARGMobilityAnalyzer(st.session_state.df)
                results = analyzer.analyze_mobility()

                col1, col2 = st.columns([2, 1])
                with col1:
                    if 'Node' in results.columns:
                        display_df = results[["Node", "mobility_potential", "mobility_category"]]
                    else:
                        display_df = results[["mobility_potential", "mobility_category"]].copy()
                        display_df.insert(0, "Index", range(len(display_df)))
                    st.dataframe(display_df, use_container_width=True)

                with col2:
                    mobility_counts = results['mobility_category'].value_counts().reset_index()
                    mobility_counts.columns = ['Mobility Category', 'Count']
                    fig = px.bar(mobility_counts, x='Mobility Category', y='Count',
                               title="Mobility Category Distribution",
                               color='Mobility Category',
                               color_discrete_map={
                                   '🔴 Low Mobility': '#2ECC71',
                                   '🟡 Moderate Mobility': '#F1C40F',
                                   '🟢 High Mobility': '#E74C3C'
                               })
                    st.plotly_chart(fig, use_container_width=True)

                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Mobility Analysis", csv, "mobility_analysis.csv", "text/csv")

    with tab4:
        if st.session_state.df.empty:
            st.warning("⚠️ Please upload or add data first.")
        else:
            st.header("Sensitivity Analysis")
            
            required_mobility_columns = ['CommunicationEfficiency', 'Degree', 'BetweennessCentrality', 
                                        'ClusteringCoefficient', 'PositiveTopologyCoefficient', 'NeighborhoodConnectivity']
            
            missing = [col for col in required_mobility_columns if col not in st.session_state.df.columns]
            
            if missing:
                st.error(f"❌ Cannot perform sensitivity analysis. Missing columns: {', '.join(missing)}")
            else:
                st.markdown("""
                **Sensitivity analysis** evaluates how changes in weight assignments affect mobility classification results.
                This helps assess the robustness of the mobility scoring system.
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    perturbation = st.slider("Weight Perturbation Range (±)", 0.05, 0.30, 0.10, 0.05)
                with col2:
                    iterations = st.slider("Number of Iterations", 10, 100, 20, 10)
                
                if st.button("🔬 Run Sensitivity Analysis"):
                    with st.spinner("Performing sensitivity analysis..."):
                        analyzer = ARGMobilityAnalyzer(st.session_state.df)
                        sensitivity_data = analyzer.perform_sensitivity_analysis(perturbation, iterations)
                        
                        st.success("✅ Analysis complete!")
                        
                        # Display results
                        st.subheader("Results Summary")
                        
                        sens_df = pd.DataFrame(sensitivity_data['sensitivity_results'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Category Changes", 
                                     f"{sens_df['category_changes'].mean():.1f}")
                        with col2:
                            st.metric("Max Category Changes", 
                                     f"{sens_df['category_changes'].max():.0f}")
                        with col3:
                            st.metric("Avg Change Percentage", 
                                     f"{sens_df['change_percentage'].mean():.1f}%")
                        
                        # Visualization
                        fig = px.histogram(sens_df, x='change_percentage',
                                         title="Distribution of Category Change Percentages",
                                         labels={'change_percentage': 'Percentage of Categories Changed'},
                                         nbins=20)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Weight variance impact
                        st.subheader("Weight Impact Analysis")
                        weight_impacts = []
                        for result in sensitivity_data['sensitivity_results']:
                            for key, value in result['weights'].items():
                                weight_impacts.append({
                                    'Feature': key,
                                    'Weight': value,
                                    'Change %': result['change_percentage']
                                })
                        
                        weight_df = pd.DataFrame(weight_impacts)
                        fig2 = px.scatter(weight_df, x='Weight', y='Change %', color='Feature',
                                        title="Impact of Weight Variations on Results",
                                        trendline="lowess")
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Statistical summary
                        st.subheader("Statistical Summary")
                        st.dataframe(sens_df[['iteration', 'category_changes', 'change_percentage']].describe())

    with tab5:
        st.header("Add New Data")
        
        template_df = pd.DataFrame(columns=list(COLUMN_DESCRIPTIONS.keys()))
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Template CSV", template_csv, "arg_template.csv", "text/csv")
        
        with st.form("add_data_form"):
            new_data = {}
            for col in COLUMN_DESCRIPTIONS.keys():
                new_data[col] = st.text_input(f"{col}", help=COLUMN_DESCRIPTIONS[col])

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

        new_entry_df = pd.DataFrame([new_data])
        st.download_button(
            label="📥 Download New Entry only as a CSV file",
            data=new_entry_df.to_csv(index=False).encode('utf-8'),
            file_name="new_entry.csv",
            mime="text/csv"
        )

if __name__ == '__main__':
    main()
