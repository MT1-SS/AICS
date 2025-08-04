"""
AI for Vulnerability Assessment - Interactive Streamlit Dashboard
Course: AI in Cybersecurity - Class 06
Instructor: Steve Smith

This interactive dashboard demonstrates all key concepts from the lesson:
1. Vulnerability Risk Prioritization
2. Vulnerability Prediction
3. Security Anomaly Detection
4. Smart Fuzzing Simulation
5. Comprehensive Risk Assessment

To run this dashboard:
1. Save as 'va_dashboard.py'
2. Install: pip install streamlit plotly
3. Run: streamlit run va_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI Vulnerability Assessment Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high { color: #d63384; font-weight: bold; }
    .risk-medium { color: #fd7e14; font-weight: bold; }
    .risk-low { color: #20c997; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">🔐 AI for Vulnerability Assessment Dashboard</h1>', unsafe_allow_html=True)

st.markdown("""
Welcome to the interactive AI Vulnerability Assessment Dashboard! This tool demonstrates the key concepts 
from **Class 06: AI for Vulnerability Assessment** through hands-on machine learning applications.

**🎯 Learning Objectives:**
- Experience AI-powered vulnerability prioritization
- Understand predictive vulnerability assessment
- Explore security anomaly detection
- Simulate smart fuzzing techniques
""")

# Sidebar Configuration
st.sidebar.header("🛠️ Dashboard Configuration")
st.sidebar.markdown("---")

# Data generation parameters
st.sidebar.subheader("📊 Dataset Parameters")
n_vulnerabilities = st.sidebar.slider("Number of Vulnerabilities", 100, 2000, 1000, 100)
n_software_packages = st.sidebar.slider("Number of Software Packages", 100, 1000, 500, 50)
n_network_records = st.sidebar.slider("Network Traffic Records", 500, 5000, 2000, 250)

# Model parameters
st.sidebar.subheader("🤖 Model Parameters")
risk_threshold = st.sidebar.slider("High Risk Threshold", 0.1, 0.9, 0.7, 0.05)
anomaly_contamination = st.sidebar.slider("Anomaly Detection Sensitivity", 0.01, 0.2, 0.1, 0.01)

# Analysis options
st.sidebar.subheader("🔍 Analysis Options")
show_feature_importance = st.sidebar.checkbox("Show Feature Importance", True)
show_model_details = st.sidebar.checkbox("Show Model Details", False)
show_raw_data = st.sidebar.checkbox("Show Raw Data", False)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Adjust the parameters above to see how they affect the AI models!")

# Caching functions for better performance
@st.cache_data
def create_vulnerability_dataset(n_samples):
    """Create synthetic vulnerability dataset"""
    np.random.seed(42)
    
    data = {
        'vuln_id': [f'CVE-2024-{i:04d}' for i in range(n_samples)],
        'cvss_score': np.random.uniform(1.0, 10.0, n_samples),
        'asset_criticality': np.random.choice(['Low', 'Medium', 'High', 'Critical'], 
                                            n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'internet_facing': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'exploit_available': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'patch_available': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'system_uptime_days': np.random.uniform(1, 365, n_samples),
        'vulnerability_age_days': np.random.uniform(1, 1000, n_samples),
        'affected_systems': np.random.randint(1, 100, n_samples),
        'business_impact': np.random.choice(['Low', 'Medium', 'High'], 
                                          n_samples, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic priority scoring
    priority_score = (
        df['cvss_score'] * 0.3 +
        df['asset_criticality'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}) * 0.25 +
        df['internet_facing'] * 2 +
        df['exploit_available'] * 2 +
        (1 - df['patch_available']) * 1.5 +
        df['business_impact'].map({'Low': 1, 'Medium': 2, 'High': 3}) * 0.2
    )
    
    df['high_priority'] = (priority_score > priority_score.quantile(0.7)).astype(int)
    df['priority_score'] = priority_score
    
    return df

@st.cache_data
def create_software_dataset(n_samples):
    """Create software vulnerability prediction dataset"""
    np.random.seed(42)
    
    data = {
        'software_id': [f'SW-{i:04d}' for i in range(n_samples)],
        'lines_of_code': np.random.lognormal(10, 1, n_samples).astype(int),
        'cyclomatic_complexity': np.random.uniform(1, 50, n_samples),
        'code_age_months': np.random.uniform(1, 120, n_samples),
        'developer_experience': np.random.uniform(1, 20, n_samples),
        'code_review_coverage': np.random.uniform(0, 1, n_samples),
        'third_party_dependencies': np.random.randint(0, 200, n_samples),
        'security_testing': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'language': np.random.choice(['C', 'C++', 'Java', 'Python', 'JavaScript'], n_samples),
        'open_source': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Predict vulnerability count based on realistic factors
    vuln_score = (
        np.log(df['lines_of_code']) * 0.1 +
        df['cyclomatic_complexity'] * 0.05 +
        df['code_age_months'] * 0.01 +
        (20 - df['developer_experience']) * 0.1 +
        (1 - df['code_review_coverage']) * 2 +
        df['third_party_dependencies'] * 0.005 +
        (1 - df['security_testing']) * 1.5 +
        np.random.normal(0, 0.5, n_samples)
    )
    
    df['predicted_vulnerabilities'] = np.maximum(0, vuln_score).round().astype(int)
    
    return df

@st.cache_data
def create_network_traffic_dataset(n_samples):
    """Create network traffic dataset for anomaly detection"""
    np.random.seed(42)
    
    # Normal traffic (90% of data)
    normal_size = int(n_samples * 0.9)
    normal_data = {
        'packet_size': np.random.normal(1500, 300, normal_size),
        'connection_duration': np.random.exponential(30, normal_size),
        'bytes_sent': np.random.lognormal(8, 1, normal_size),
        'bytes_received': np.random.lognormal(8, 1, normal_size),
        'packets_per_second': np.random.gamma(2, 10, normal_size),
        'unique_destinations': np.random.poisson(5, normal_size),
        'port_scans': np.random.poisson(0.1, normal_size),
        'failed_connections': np.random.poisson(1, normal_size)
    }
    
    # Anomalous traffic (10% of data)
    anomaly_size = n_samples - normal_size
    anomaly_data = {
        'packet_size': np.random.normal(3000, 1000, anomaly_size),
        'connection_duration': np.random.exponential(5, anomaly_size),
        'bytes_sent': np.random.lognormal(12, 1.5, anomaly_size),
        'bytes_received': np.random.lognormal(6, 1, anomaly_size),
        'packets_per_second': np.random.gamma(5, 50, anomaly_size),
        'unique_destinations': np.random.poisson(50, anomaly_size),
        'port_scans': np.random.poisson(10, anomaly_size),
        'failed_connections': np.random.poisson(20, anomaly_size)
    }
    
    # Combine data
    all_data = {}
    for key in normal_data:
        all_data[key] = np.concatenate([normal_data[key], anomaly_data[key]])
    
    all_data['is_anomaly'] = np.concatenate([np.zeros(normal_size), np.ones(anomaly_size)])
    
    df = pd.DataFrame(all_data)
    return df.sample(frac=1).reset_index(drop=True)

# Main Dashboard Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Vulnerability Prioritization", 
    "🔮 Vulnerability Prediction", 
    "🚨 Anomaly Detection", 
    "🔬 Smart Fuzzing", 
    "🏆 Comprehensive Dashboard"
])

# Tab 1: Vulnerability Prioritization
with tab1:
    st.header("🎯 AI-Powered Vulnerability Prioritization")
    st.markdown("""
    This section demonstrates how AI can prioritize vulnerabilities beyond simple CVSS scores,
    considering organizational context and threat intelligence.
    """)
    
    # Generate and process vulnerability data
    vuln_df = create_vulnerability_dataset(n_vulnerabilities)
    
    # Prepare features for ML
    le_asset = LabelEncoder()
    le_business = LabelEncoder()
    
    vuln_features = vuln_df.copy()
    vuln_features['asset_criticality_encoded'] = le_asset.fit_transform(vuln_df['asset_criticality'])
    vuln_features['business_impact_encoded'] = le_business.fit_transform(vuln_df['business_impact'])
    
    feature_columns = [
        'cvss_score', 'asset_criticality_encoded', 'internet_facing',
        'exploit_available', 'patch_available', 'system_uptime_days',
        'vulnerability_age_days', 'affected_systems', 'business_impact_encoded'
    ]
    
    X = vuln_features[feature_columns]
    y = vuln_features['high_priority']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions and probabilities
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X)[:, 1]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Vulnerabilities", len(vuln_df))
    with col2:
        high_priority_count = (y_prob > risk_threshold).sum()
        st.metric("High Priority (AI)", high_priority_count)
    with col3:
        traditional_high = (vuln_df['cvss_score'] >= 7.0).sum()
        st.metric("High CVSS (≥7.0)", traditional_high)
    with col4:
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{accuracy:.1%}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Score Distribution
        fig = px.histogram(
            x=y_prob, 
            nbins=30,
            title="AI Risk Score Distribution",
            labels={'x': 'AI Risk Probability', 'y': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Risk Threshold ({risk_threshold})")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CVSS vs AI Risk Comparison
        vuln_df['ai_risk_score'] = y_prob
        fig = px.scatter(
            vuln_df, 
            x='cvss_score', 
            y='ai_risk_score',
            color='asset_criticality',
            title="CVSS vs AI Risk Score",
            labels={'cvss_score': 'CVSS Score', 'ai_risk_score': 'AI Risk Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    if show_feature_importance:
        st.subheader("📊 Feature Importance Analysis")
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_importance, 
            x='importance', 
            y='feature',
            orientation='h',
            title="Which Factors Drive Vulnerability Priority?",
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Vulnerabilities Table
    st.subheader("🔍 Top Risk Vulnerabilities")
    top_vulns = vuln_df.nlargest(10, 'ai_risk_score')[
        ['vuln_id', 'cvss_score', 'ai_risk_score', 'asset_criticality', 
         'internet_facing', 'exploit_available']
    ].copy()
    
    # Format the ai_risk_score for better readability
    top_vulns['ai_risk_score'] = top_vulns['ai_risk_score'].round(3)
    
    # Add risk level indicator as a separate column instead of background coloring
    def get_risk_level(score):
        if score > 0.8:
            return "🔴 High"
        elif score > 0.6:
            return "🟡 Medium"
        else:
            return "🟢 Low"
    
    top_vulns['Risk Level'] = top_vulns['ai_risk_score'].apply(get_risk_level)
    
    # Reorder columns to show risk level
    top_vulns = top_vulns[['vuln_id', 'cvss_score', 'ai_risk_score', 'Risk Level', 'asset_criticality', 
                          'internet_facing', 'exploit_available']]
    
    st.dataframe(top_vulns, use_container_width=True)

# Tab 2: Vulnerability Prediction
with tab2:
    st.header("🔮 Vulnerability Prediction in Software")
    st.markdown("""
    This section demonstrates how AI can predict vulnerability likelihood in software
    based on code characteristics and development practices.
    """)
    
    # Generate software data
    software_df = create_software_dataset(n_software_packages)
    
    # Prepare features
    le_language = LabelEncoder()
    software_features = software_df.copy()
    software_features['language_encoded'] = le_language.fit_transform(software_df['language'])
    
    software_feature_columns = [
        'lines_of_code', 'cyclomatic_complexity', 'code_age_months',
        'developer_experience', 'code_review_coverage', 'third_party_dependencies',
        'security_testing', 'language_encoded', 'open_source'
    ]
    
    X_soft = software_features[software_feature_columns]
    y_soft = software_features['predicted_vulnerabilities']
    
    # Train prediction model
    X_train_soft, X_test_soft, y_train_soft, y_test_soft = train_test_split(
        X_soft, y_soft, test_size=0.2, random_state=42
    )
    
    rf_pred = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_pred.fit(X_train_soft, y_train_soft)
    
    y_pred_soft = rf_pred.predict(X_test_soft)
    r2_score = rf_pred.score(X_test_soft, y_test_soft)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Software Packages", len(software_df))
    with col2:
        avg_predicted = software_df['predicted_vulnerabilities'].mean()
        st.metric("Avg Predicted Vulns", f"{avg_predicted:.1f}")
    with col3:
        high_risk_software = (software_df['predicted_vulnerabilities'] > 10).sum()
        st.metric("High Risk Software", high_risk_software)
    with col4:
        st.metric("Model R² Score", f"{r2_score:.3f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction vs Actual
        fig = px.scatter(
            x=y_test_soft, 
            y=y_pred_soft,
            title="Vulnerability Prediction: Actual vs Predicted",
            labels={'x': 'Actual Vulnerabilities', 'y': 'Predicted Vulnerabilities'}
        )
        # Add perfect prediction line
        max_val = max(y_test_soft.max(), y_pred_soft.max())
        fig.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Vulnerability distribution by programming language
        lang_vulns = software_df.groupby('language')['predicted_vulnerabilities'].mean().reset_index()
        fig = px.bar(
            lang_vulns,
            x='language',
            y='predicted_vulnerabilities',
            title="Average Vulnerabilities by Programming Language",
            labels={'predicted_vulnerabilities': 'Avg Vulnerabilities', 'language': 'Language'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Code Quality vs Vulnerabilities
    st.subheader("📈 Code Quality Impact on Security")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Code Review Coverage vs Vulnerabilities", "Developer Experience vs Vulnerabilities")
    )
    
    fig.add_trace(
        go.Scatter(
            x=software_df['code_review_coverage'],
            y=software_df['predicted_vulnerabilities'],
            mode='markers',
            name='Code Review',
            opacity=0.6
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=software_df['developer_experience'],
            y=software_df['predicted_vulnerabilities'],
            mode='markers',
            name='Dev Experience',
            opacity=0.6
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Impact of Development Practices on Security")
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Anomaly Detection
with tab3:
    st.header("🚨 Security Anomaly Detection")
    st.markdown("""
    This section demonstrates how AI can detect unusual network patterns
    that might indicate security incidents or attacks.
    """)
    
    # Generate network traffic data
    traffic_df = create_network_traffic_dataset(n_network_records)
    
    # Prepare features for anomaly detection
    traffic_features = [
        'packet_size', 'connection_duration', 'bytes_sent', 'bytes_received',
        'packets_per_second', 'unique_destinations', 'port_scans', 'failed_connections'
    ]
    
    X_traffic = traffic_df[traffic_features]
    y_traffic = traffic_df['is_anomaly']
    
    # Standardize features
    scaler = StandardScaler()
    X_traffic_scaled = scaler.fit_transform(X_traffic)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=anomaly_contamination, random_state=42)
    anomaly_pred = iso_forest.fit_predict(X_traffic_scaled)
    anomaly_pred_binary = (anomaly_pred == -1).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_traffic, anomaly_pred_binary)
    recall = recall_score(y_traffic, anomaly_pred_binary)
    f1 = f1_score(y_traffic, anomaly_pred_binary)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Network Records", len(traffic_df))
    with col2:
        actual_anomalies = y_traffic.sum()
        st.metric("Actual Anomalies", actual_anomalies)
    with col3:
        detected_anomalies = anomaly_pred_binary.sum()
        st.metric("Detected Anomalies", detected_anomalies)
    with col4:
        st.metric("Detection Precision", f"{precision:.1%}")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision", f"{precision:.3f}")
    with col2:
        st.metric("Recall", f"{recall:.3f}")
    with col3:
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly Detection Scatter Plot
        fig = px.scatter(
            traffic_df,
            x='bytes_sent',
            y='bytes_received',
            color=y_traffic.astype(str),
            title="Actual Anomalies in Network Traffic",
            labels={'color': 'Anomaly', 'bytes_sent': 'Bytes Sent', 'bytes_received': 'Bytes Received'},
            log_x=True,
            log_y=True,
            color_discrete_map={'0': 'blue', '1': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Detected Anomalies
        fig = px.scatter(
            traffic_df,
            x='bytes_sent',
            y='bytes_received',
            color=anomaly_pred_binary.astype(str),
            title="AI-Detected Anomalies",
            labels={'color': 'Detected', 'bytes_sent': 'Bytes Sent', 'bytes_received': 'Bytes Received'},
            log_x=True,
            log_y=True,
            color_discrete_map={'0': 'blue', '1': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("🎯 Detection Performance Analysis")
    
    cm = confusion_matrix(y_traffic, anomaly_pred_binary)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix - Anomaly Detection",
        labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'}
    )
    
    fig.update_xaxes(tickvals=[0, 1], ticktext=['Normal', 'Anomaly'])
    fig.update_yaxes(tickvals=[0, 1], ticktext=['Normal', 'Anomaly'])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Traffic Pattern Analysis
    st.subheader("📊 Traffic Pattern Analysis")
    
    # Compare normal vs anomalous traffic characteristics
    normal_traffic = traffic_df[traffic_df['is_anomaly'] == 0]
    anomalous_traffic = traffic_df[traffic_df['is_anomaly'] == 1]
    
    comparison_metrics = []
    for feature in traffic_features:
        comparison_metrics.append({
            'Feature': feature,
            'Normal (Mean)': normal_traffic[feature].mean(),
            'Anomalous (Mean)': anomalous_traffic[feature].mean(),
            'Difference': anomalous_traffic[feature].mean() - normal_traffic[feature].mean()
        })
    
    comparison_df = pd.DataFrame(comparison_metrics)
    st.dataframe(comparison_df, use_container_width=True)

# Tab 4: Smart Fuzzing
with tab4:
    st.header("🔬 Smart Fuzzing Simulation")
    st.markdown("""
    This section simulates AI-powered vulnerability discovery through intelligent
    fuzzing techniques that adapt and learn from testing results.
    """)
    
    # Fuzzing simulation parameters
    st.subheader("⚙️ Fuzzing Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        fuzzing_rounds = st.selectbox("Fuzzing Rounds", [5, 10, 15, 20], index=1)
        tests_per_round = st.selectbox("Tests per Round", [100, 250, 500, 1000], index=1)
    
    with col2:
        vulnerability_types = st.multiselect(
            "Vulnerability Types to Test",
            ['buffer_overflow', 'sql_injection', 'xss', 'format_string', 'integer_overflow'],
            default=['buffer_overflow', 'sql_injection', 'xss']
        )
    
    if st.button("🚀 Run Smart Fuzzing Simulation"):
        # Simulate smart fuzzing with learning
        np.random.seed(42)
        
        # Base success rates for different vulnerability types
        base_success_rates = {
            'buffer_overflow': 0.15,
            'sql_injection': 0.08,
            'xss': 0.12,
            'format_string': 0.06,
            'integer_overflow': 0.04
        }
        
        fuzzing_results = []
        
        for vuln_type in vulnerability_types:
            base_rate = base_success_rates[vuln_type]
            
            for round_num in range(1, fuzzing_rounds + 1):
                # Simulate learning - success rate improves over rounds
                learning_factor = 1 + (round_num - 1) * 0.1
                adaptive_rate = min(base_rate * learning_factor, 0.5)  # Cap at 50%
                
                tests_this_round = np.random.randint(
                    int(tests_per_round * 0.8), 
                    int(tests_per_round * 1.2)
                )
                successes = np.random.binomial(tests_this_round, adaptive_rate)
                
                fuzzing_results.append({
                    'Vulnerability Type': vuln_type,
                    'Round': round_num,
                    'Tests': tests_this_round,
                    'Successes': successes,
                    'Success Rate': successes / tests_this_round,
                    'Expected Rate': adaptive_rate
                })
        
        fuzzing_df = pd.DataFrame(fuzzing_results)
        
        # Summary metrics
        total_tests = fuzzing_df['Tests'].sum()
        total_successes = fuzzing_df['Successes'].sum()
        overall_success_rate = total_successes / total_tests
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tests", f"{total_tests:,}")
        with col2:
            st.metric("Vulnerabilities Found", total_successes)
        with col3:
            st.metric("Overall Success Rate", f"{overall_success_rate:.1%}")
        with col4:
            estimated_time = total_tests * 0.1  # Assume 0.1 seconds per test
            st.metric("Estimated Time", f"{estimated_time:.0f}s")
        
        # Fuzzing Results Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Learning curves by vulnerability type
            fig = px.line(
                fuzzing_df,
                x='Round',
                y='Success Rate',
                color='Vulnerability Type',
                title="Smart Fuzzing Learning Curves",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Total vulnerabilities found by type
            vuln_summary = fuzzing_df.groupby('Vulnerability Type')['Successes'].sum().reset_index()
            fig = px.bar(
                vuln_summary,
                x='Vulnerability Type',
                y='Successes',
                title="Vulnerabilities Found by Type",
                color='Successes',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("📊 Detailed Fuzzing Results")
        st.dataframe(fuzzing_df, use_container_width=True)
        
        # Insights
        st.subheader("🔍 Key Insights")
        
        best_type = vuln_summary.loc[vuln_summary['Successes'].idxmax(), 'Vulnerability Type']
        worst_type = vuln_summary.loc[vuln_summary['Successes'].idxmin(), 'Vulnerability Type']
        
        st.success(f"🎯 **Most Effective:** {best_type} fuzzing found the most vulnerabilities")
        st.info(f"📈 **Learning Effect:** Success rates improved over {fuzzing_rounds} rounds due to AI adaptation")
        st.warning(f"⚠️ **Challenging Target:** {worst_type} vulnerabilities were hardest to discover")

# Tab 5: Comprehensive Dashboard
with tab5:
    st.header("🏆 Comprehensive AI Security Assessment")
    st.markdown("""
    This integrated dashboard combines all AI techniques to provide a complete
    view of your organization's security posture and AI capabilities.
    """)
    
    # Generate all datasets
    vuln_df = create_vulnerability_dataset(n_vulnerabilities)
    software_df = create_software_dataset(n_software_packages)
    traffic_df = create_network_traffic_dataset(n_network_records)
    
    # Overall Security Score Calculation
    # This combines vulnerability assessment, prediction, and anomaly detection
    
    # Vulnerability assessment score
    high_priority_vulns = (vuln_df['priority_score'] > vuln_df['priority_score'].quantile(0.8)).sum()
    vuln_score = max(0, 100 - (high_priority_vulns / len(vuln_df)) * 100)
    
    # Software security score
    high_risk_software = (software_df['predicted_vulnerabilities'] > 10).sum()
    software_score = max(0, 100 - (high_risk_software / len(software_df)) * 100)
    
    # Network security score
    actual_anomalies = traffic_df['is_anomaly'].sum()
    network_score = max(0, 100 - (actual_anomalies / len(traffic_df)) * 100)
    
    # Overall security posture
    overall_score = (vuln_score + software_score + network_score) / 3
    
    # Security Score Display
    st.subheader("🎯 AI Security Assessment Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Security Score", 
            f"{overall_score:.0f}/100",
            delta=f"{overall_score-75:.0f}" if overall_score > 75 else f"{overall_score-75:.0f}"
        )
    
    with col2:
        st.metric(
            "Vulnerability Management", 
            f"{vuln_score:.0f}/100",
            delta=None
        )
    
    with col3:
        st.metric(
            "Software Security", 
            f"{software_score:.0f}/100",
            delta=None
        )
    
    with col4:
        st.metric(
            "Network Security", 
            f"{network_score:.0f}/100",
            delta=None
        )
    
    # Risk Level Indicator
    if overall_score >= 80:
        st.success("🟢 **LOW RISK** - Strong security posture with effective AI implementation")
    elif overall_score >= 60:
        st.warning("🟡 **MEDIUM RISK** - Good security posture with room for AI enhancement")
    else:
        st.error("🔴 **HIGH RISK** - Significant security concerns requiring immediate AI-powered improvements")
    
    # Executive Summary Charts
    st.subheader("📊 Executive Security Dashboard")
    
    # Security scores radar chart
    categories = ['Vulnerability<br>Management', 'Software<br>Security', 'Network<br>Security']
    scores = [vuln_score, software_score, network_score]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],  # Close the shape
        theta=categories + [categories[0]],
        fill='toself',
        name='Current Security Posture',
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[80, 80, 80, 80],  # Target scores
        theta=categories + [categories[0]],
        fill='tonext',
        name='Target Security Level',
        line_color='green',
        opacity=0.3
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Security Posture Assessment"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # AI Impact Analysis
        ai_impact_data = {
            'AI Application': [
                'Vulnerability Prioritization',
                'Predictive Assessment', 
                'Anomaly Detection',
                'Smart Fuzzing'
            ],
            'Efficiency Gain': [85, 70, 90, 60],
            'Accuracy Improvement': [75, 80, 65, 85],
            'Cost Reduction': [60, 90, 80, 70]
        }
        
        ai_df = pd.DataFrame(ai_impact_data)
        
        fig = px.bar(
            ai_df.melt(id_vars=['AI Application'], 
                      value_vars=['Efficiency Gain', 'Accuracy Improvement', 'Cost Reduction']),
            x='AI Application',
            y='value',
            color='variable',
            title="AI Impact on Security Operations",
            labels={'value': 'Improvement (%)', 'variable': 'Metric'},
            barmode='group'
        )
        
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 AI-Powered Security Recommendations")
    
    recommendations = []
    
    if vuln_score < 75:
        recommendations.append({
            'Priority': 'High',
            'Area': 'Vulnerability Management',
            'Recommendation': 'Implement AI-powered vulnerability prioritization to focus on critical assets',
            'Expected Impact': 'Reduce remediation time by 60%'
        })
    
    if software_score < 75:
        recommendations.append({
            'Priority': 'Medium',
            'Area': 'Software Security',
            'Recommendation': 'Integrate predictive vulnerability assessment in CI/CD pipeline',
            'Expected Impact': 'Prevent 70% of vulnerabilities from reaching production'
        })
    
    if network_score < 75:
        recommendations.append({
            'Priority': 'High',
            'Area': 'Network Security',
            'Recommendation': 'Deploy AI-based anomaly detection for real-time threat monitoring',
            'Expected Impact': 'Improve threat detection by 80%'
        })
    
    if overall_score >= 80:
        recommendations.append({
            'Priority': 'Low',
            'Area': 'Optimization',
            'Recommendation': 'Explore advanced AI techniques like adversarial training and federated learning',
            'Expected Impact': 'Maintain security leadership position'
        })
    
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        
        # Add visual priority indicators instead of background coloring
        def format_priority(priority):
            if priority == 'High':
                return "🔴 High"
            elif priority == 'Medium':
                return "🟡 Medium"
            else:
                return "🟢 Low"
        
        rec_df['Priority'] = rec_df['Priority'].apply(format_priority)
        
        st.dataframe(rec_df, use_container_width=True)
    
    # Implementation Roadmap
    st.subheader("🗺️ AI Security Implementation Roadmap")
    
    roadmap_data = {
        'Phase': ['Phase 1 (0-3 months)', 'Phase 2 (3-6 months)', 'Phase 3 (6-12 months)'],
        'Focus Area': [
            'Vulnerability Prioritization & Basic ML',
            'Predictive Assessment & Anomaly Detection', 
            'Advanced AI & Integration'
        ],
        'Key Activities': [
            'Deploy vulnerability scanner ML, Train security team',
            'Implement SIEM anomaly detection, CI/CD integration',
            'Advanced threat hunting, AI model optimization'
        ],
        'Expected ROI': ['30-50% efficiency gain', '50-70% threat reduction', '70-90% automation level']
    }
    
    roadmap_df = pd.DataFrame(roadmap_data)
    st.dataframe(roadmap_df, use_container_width=True)

# Show raw data if requested
if show_raw_data:
    st.subheader("📋 Raw Data Preview")
    
    data_choice = st.selectbox(
        "Choose dataset to view:",
        ["Vulnerabilities", "Software", "Network Traffic"]
    )
    
    if data_choice == "Vulnerabilities":
        st.dataframe(vuln_df.head(100), use_container_width=True)
    elif data_choice == "Software":
        st.dataframe(software_df.head(100), use_container_width=True)
    else:
        st.dataframe(traffic_df.head(100), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔐 AI for Vulnerability Assessment Dashboard | Course: AI in Cybersecurity - Class 06</p>
    <p>Built with Streamlit | Data generated for educational purposes</p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**📚 Learning Resources:**
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Vulnerability Management](https://owasp.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
""")

st.sidebar.success("✅ Dashboard loaded successfully!")