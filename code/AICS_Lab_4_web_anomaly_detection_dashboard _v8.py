import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from scipy import stats
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitAnomalyDetector:
    """Streamlit-adapted version of the UnsupervisedWebAnomalyDetector"""
    
    def __init__(self):
        self.data = None
        self.original_data = None
        self.features = None
        self.scaled_features = None
        self.scaler = StandardScaler()
        self.anomaly_scores = {}
        self.outliers = {}
        self.labels = None
        self.confidence_scores = {}
        self.security_features = []
        self.pca_analysis = {}
        self.attack_analysis = {}
        
    def load_and_prepare_data(self, uploaded_file, max_rows=None):
        """Load and prepare data from uploaded file"""
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                if max_rows:
                    self.original_data = pd.read_csv(uploaded_file, nrows=max_rows)
                    self.is_sampled = True
                    self.sample_size = max_rows
                else:
                    self.original_data = pd.read_csv(uploaded_file)
                    self.is_sampled = False
                    self.sample_size = None
            else:
                st.error("Please upload a CSV file")
                return False
            
            # Clean column names
            self.original_data.columns = self.original_data.columns.str.strip()
            
            # Reset index immediately after loading to ensure consistency
            self.original_data = self.original_data.reset_index(drop=True)
            
            # Store original data and create working copy
            self.data = self.original_data.copy()
            
            # Check for labels BEFORE any data manipulation
            if 'Label' in self.data.columns:
                self.labels = self.data['Label'].copy().reset_index(drop=True)
                self.data = self.data.drop('Label', axis=1).reset_index(drop=True)
            else:
                self.labels = None
            
            print(f"After label extraction - Data shape: {self.data.shape}")
            if self.labels is not None:
                print(f"Labels length: {len(self.labels)}")
                
            # Handle missing and infinite values
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            
            # Only keep numeric columns
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            non_numeric_dropped = len(self.data.columns) - len(numeric_columns)
            
            if non_numeric_dropped > 0:
                print(f"Dropping {non_numeric_dropped} non-numeric columns")
            
            self.data = self.data[numeric_columns].copy()
            
            # Reset index after column filtering
            self.data = self.data.reset_index(drop=True)
            print(f"After numeric filtering - Data shape: {self.data.shape}")
            
            # Fill missing values with median (column by column to avoid issues)
            for col in self.data.columns:
                if self.data[col].isnull().any():
                    median_val = self.data[col].median()
                    if pd.isna(median_val):  # If all values are NaN, use 0
                        median_val = 0
                    self.data[col] = self.data[col].fillna(median_val)
            
            print(f"After missing value handling - Data shape: {self.data.shape}")
            
            # Remove constant features
            constant_cols = []
            for col in self.data.columns:
                if self.data[col].var() == 0 or self.data[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if len(constant_cols) > 0:
                print(f"Removing {len(constant_cols)} constant features")
                self.data = self.data.drop(constant_cols, axis=1)
                
            # Final reset index to ensure everything is aligned
            self.data = self.data.reset_index(drop=True)
            print(f"Final data shape: {self.data.shape}")
            
            # Validate data is not empty
            if self.data.shape[0] == 0:
                st.error("No data remaining after preprocessing")
                return False
                
            if self.data.shape[1] == 0:
                st.error("No numeric features found in the data")
                return False
            
            # Identify security features
            self.security_features = self.identify_security_features()
            
            # Scale features with explicit validation
            try:
                print(f"Scaling features - Input shape: {self.data.shape}")
                scaled_array = self.scaler.fit_transform(self.data)
                print(f"Scaled array shape: {scaled_array.shape}")
                
                # Create new index for scaled features to avoid any index issues
                new_index = range(len(self.data))
                
                self.scaled_features = pd.DataFrame(
                    scaled_array,
                    columns=self.data.columns,
                    index=new_index
                )
                
                print(f"Scaled features shape: {self.scaled_features.shape}")
                
            except Exception as e:
                st.error(f"Error during feature scaling: {str(e)}")
                print(f"Scaling error details: {e}")
                return False
            
            # Final validation
            if self.labels is not None and len(self.labels) != len(self.data):
                st.warning(f"Length mismatch after preprocessing: Data={len(self.data)}, Labels={len(self.labels)}")
                # Truncate labels to match data if needed
                if len(self.labels) > len(self.data):
                    self.labels = self.labels[:len(self.data)].reset_index(drop=True)
                    st.info("Truncated labels to match data length")
                else:
                    st.warning("Data preprocessing may have affected label alignment")
            
            print(f"Final validation - Data: {self.data.shape}, Scaled: {self.scaled_features.shape}")
            if self.labels is not None:
                print(f"Labels: {len(self.labels)}")
                
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            print(f"Full error details: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def identify_security_features(self):
        """Identify security-relevant features"""
        security_keywords = [
            'duration', 'packets', 'bytes', 'port', 'flag', 'length',
            'rate', 'iat', 'window', 'segment', 'bulk', 'active', 'idle',
            'flow', 'syn', 'fin', 'rst', 'psh', 'ack', 'urg'
        ]
        
        security_features = []
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in security_keywords):
                security_features.append(col)
        
        return security_features
    
    def isolation_forest_detection(self, contamination=0.05, n_estimators=100, max_features='auto', bootstrap=False):
        """Isolation Forest anomaly detection with advanced parameters"""
        # Handle max_features parameter
        if max_features == 'auto':
            max_features_val = min(10, self.scaled_features.shape[1])
        elif max_features == 'sqrt':
            max_features_val = int(np.sqrt(self.scaled_features.shape[1]))
        elif max_features == 'log2':
            max_features_val = int(np.log2(self.scaled_features.shape[1]))
        elif max_features == 'all':
            max_features_val = self.scaled_features.shape[1]
        else:
            max_features_val = max_features
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=n_estimators,
            max_features=max_features_val,
            bootstrap=bootstrap
        )
        
        anomaly_labels = iso_forest.fit_predict(self.scaled_features)
        anomaly_scores = iso_forest.decision_function(self.scaled_features)
        
        outlier_indices = np.where(anomaly_labels == -1)[0]
        
        self.outliers['isolation_forest'] = outlier_indices
        self.anomaly_scores['isolation_forest'] = anomaly_scores
        
        return outlier_indices, anomaly_scores
    
    def local_outlier_factor_detection(self, n_neighbors=20, contamination=0.05, algorithm='auto', 
                                     leaf_size=30, metric='minkowski', p=2):
        """Local Outlier Factor detection with advanced parameters"""
        # Adjust n_neighbors for small datasets
        n_neighbors = min(n_neighbors, len(self.scaled_features) - 1)
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p
        )
        
        anomaly_labels = lof.fit_predict(self.scaled_features)
        anomaly_scores = lof.negative_outlier_factor_
        
        outlier_indices = np.where(anomaly_labels == -1)[0]
        
        self.outliers['lof'] = outlier_indices
        self.anomaly_scores['lof'] = anomaly_scores
        
        return outlier_indices, anomaly_scores
    
    def kmeans_outlier_detection(self, n_clusters=None, contamination=0.05, algorithm='lloyd', 
                                init='k-means++', n_init=10):
        """K-means clustering based outlier detection with advanced parameters"""
        if n_clusters is None:
            n_samples = len(self.scaled_features)
            n_clusters = max(3, min(15, n_samples // 500))
        
        # Ensure we don't have more clusters than samples
        n_clusters = min(n_clusters, len(self.scaled_features))
        
        # Validate algorithm parameter for current scikit-learn version
        valid_algorithms = ['lloyd', 'elkan']
        if algorithm not in valid_algorithms:
            algorithm = 'lloyd'  # Default fallback
        
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=n_init,
            algorithm=algorithm,
            init=init
        )
        cluster_labels = kmeans.fit_predict(self.scaled_features)
        cluster_centers = kmeans.cluster_centers_
        
        # Calculate distances from points to their cluster centers
        distances = []
        for i, point in enumerate(self.scaled_features.values):
            cluster_id = cluster_labels[i]
            center = cluster_centers[cluster_id]
            distance = np.linalg.norm(point - center)
            distances.append(distance)
        
        distances = np.array(distances)
        threshold = np.percentile(distances, (1 - contamination) * 100)
        
        outlier_indices = np.where(distances > threshold)[0]
        
        self.outliers['kmeans'] = outlier_indices
        self.anomaly_scores['kmeans'] = -distances  # Negative for consistency
        
        return outlier_indices, distances
    
    def one_class_svm_detection(self, contamination=0.05, kernel='rbf', gamma='scale', 
                               degree=3, max_iter=5000):
        """One-Class SVM detection with advanced parameters"""
        nu = min(contamination, 0.5)
        
        svm = OneClassSVM(
            nu=nu, 
            kernel=kernel, 
            gamma=gamma,
            degree=degree,
            max_iter=max_iter
        )
        svm.fit(self.scaled_features)
        
        anomaly_labels = svm.predict(self.scaled_features)
        decision_scores = svm.decision_function(self.scaled_features)
        
        outlier_indices = np.where(anomaly_labels == -1)[0]
        
        self.outliers['one_class_svm'] = outlier_indices
        self.anomaly_scores['one_class_svm'] = decision_scores
        
        return outlier_indices, decision_scores
    
    def dbscan_clustering(self, eps=0.5, min_samples=5, algorithm='auto', metric='euclidean'):
        """DBSCAN clustering for outlier detection with advanced parameters"""
        # Adjust min_samples for small datasets
        min_samples = min(min_samples, len(self.scaled_features) // 10)
        min_samples = max(min_samples, 2)  # Ensure at least 2
        
        dbscan = DBSCAN(
            eps=eps, 
            min_samples=min_samples,
            algorithm=algorithm,
            metric=metric
        )
        cluster_labels = dbscan.fit_predict(self.scaled_features)
        
        outlier_indices = np.where(cluster_labels == -1)[0]
        
        self.outliers['dbscan'] = outlier_indices
        # For DBSCAN, we'll create pseudo-scores based on distance to nearest cluster
        if len(outlier_indices) > 0:
            pseudo_scores = np.zeros(len(self.scaled_features))
            # Mark outliers with negative scores
            pseudo_scores[outlier_indices] = -1
            self.anomaly_scores['dbscan'] = pseudo_scores
        
        return outlier_indices, cluster_labels
    
    def generate_confidence_scores(self):
        """Generate confidence scores from multiple methods"""
        if not self.outliers:
            return np.array([])
        
        n_samples = len(self.data)
        confidence_scores = np.zeros(n_samples)
        method_count = 0
        
        for method_name, outlier_indices in self.outliers.items():
            if len(outlier_indices) > 0:
                method_count += 1
                confidence_scores[outlier_indices] += 1
        
        if method_count > 0:
            confidence_scores = confidence_scores / method_count
            self.confidence_scores['ensemble'] = confidence_scores
            
        return confidence_scores
    
    def analyze_attack_patterns(self):
        """Analyze attack patterns in detected outliers"""
        if not self.outliers:
            return {}
        
        all_outliers = set()
        for outliers in self.outliers.values():
            all_outliers.update(outliers)
        
        if len(all_outliers) == 0:
            return {}
        
        analysis = {}
        outlier_data = self.data.iloc[list(all_outliers)]
        
        # Analyze destination ports if available
        if 'Destination Port' in outlier_data.columns:
            port_counts = outlier_data['Destination Port'].value_counts()
            port_services = {
                22: 'SSH', 80: 'HTTP', 443: 'HTTPS', 21: 'FTP',
                25: 'SMTP', 53: 'DNS', 3389: 'RDP', 1433: 'SQL Server'
            }
            
            target_services = {}
            for port, count in port_counts.head(10).items():
                service = port_services.get(port, f'Port-{port}')
                target_services[service] = {
                    'port': port,
                    'count': count,
                    'percentage': (count / len(all_outliers)) * 100
                }
            
            analysis['target_services'] = target_services
        
        # Analyze flow characteristics
        if self.security_features:
            feature_analysis = {}
            normal_data = self.data.drop(list(all_outliers))
            
            for feature in self.security_features[:5]:
                if feature in outlier_data.columns:
                    outlier_mean = outlier_data[feature].mean()
                    normal_mean = normal_data[feature].mean()
                    
                    feature_analysis[feature] = {
                        'outlier_mean': outlier_mean,
                        'normal_mean': normal_mean,
                        'difference_pct': ((outlier_mean - normal_mean) / normal_mean * 100) if normal_mean != 0 else 0
                    }
            
            analysis['feature_analysis'] = feature_analysis
        
        self.attack_analysis = analysis
        return analysis

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = StreamlitAnomalyDetector()
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Main dashboard
def main():
    st.markdown('<div class="main-header">🔍 Advanced Anomaly Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Detect anomalies in network traffic using multiple ML algorithms including K-means and SVM**")
    
    # Sidebar for configuration
    st.sidebar.header("📊 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload network traffic data in CSV format"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # Data loading options
        st.sidebar.subheader("Data Loading Options")
        
        # Sample size for large files
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 10:
            st.sidebar.warning(f"Large file detected ({file_size_mb:.1f} MB)")
            use_sampling = st.sidebar.checkbox("Use sampling for large files", value=True)
            if use_sampling:
                max_rows = st.sidebar.slider("Maximum rows to analyze", 1000, 50000, 10000)
            else:
                max_rows = None
        else:
            max_rows = None
        
        # Algorithm selection
        st.sidebar.subheader("🤖 Detection Algorithms")
        
        algorithms = {
            'Isolation Forest': st.sidebar.checkbox('Isolation Forest', value=True),
            'Local Outlier Factor': st.sidebar.checkbox('Local Outlier Factor', value=True),
            'K-means Clustering': st.sidebar.checkbox('K-means Clustering', value=True),
            'One-Class SVM': st.sidebar.checkbox('One-Class SVM', value=True),
            'DBSCAN': st.sidebar.checkbox('DBSCAN', value=False)
        }
        
        # Parameters
        st.sidebar.subheader("⚙️ Basic Parameters")
        contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.2, 0.05, 0.01)
        n_neighbors = st.sidebar.slider("LOF Neighbors", 5, 50, 20)
        
        # Advanced hyperparameter tuning
        st.sidebar.subheader("🔧 Advanced Hyperparameter Tuning")
        
        with st.sidebar.expander("🌲 Isolation Forest Parameters"):
            if algorithms['Isolation Forest']:
                iso_n_estimators = st.slider("Number of Trees", 50, 500, 100, 50, key="iso_trees")
                iso_max_features = st.selectbox("Max Features per Tree", 
                                              ['auto', 'sqrt', 'log2', 'all'], 
                                              index=0, key="iso_features")
                iso_bootstrap = st.checkbox("Bootstrap Samples", value=False, key="iso_bootstrap")
            else:
                iso_n_estimators, iso_max_features, iso_bootstrap = 100, 'auto', False
        
        with st.sidebar.expander("🎯 K-means Parameters"):
            if algorithms['K-means Clustering']:
                kmeans_n_clusters = st.slider("Number of Clusters", 3, 25, 8, 1, key="kmeans_clusters")
                kmeans_algorithm = st.selectbox("Algorithm", 
                                              ['lloyd', 'elkan'], 
                                              index=0, key="kmeans_algo")
                kmeans_init = st.selectbox("Initialization", 
                                         ['k-means++', 'random'], 
                                         index=0, key="kmeans_init")
                kmeans_n_init = st.slider("Number of Initializations", 5, 20, 10, 5, key="kmeans_ninit")
            else:
                kmeans_n_clusters, kmeans_algorithm, kmeans_init, kmeans_n_init = 8, 'lloyd', 'k-means++', 10
        
        with st.sidebar.expander("🎪 One-Class SVM Parameters"):
            if algorithms['One-Class SVM']:
                svm_kernel = st.selectbox("Kernel Type", 
                                        ['rbf', 'linear', 'poly', 'sigmoid'], 
                                        index=0, key="svm_kernel")
                svm_gamma = st.selectbox("Gamma", 
                                       ['scale', 'auto'], 
                                       index=0, key="svm_gamma")
                if svm_gamma not in ['scale', 'auto']:
                    svm_gamma_custom = st.slider("Custom Gamma", 0.001, 10.0, 1.0, 0.001, key="svm_gamma_val")
                else:
                    svm_gamma_custom = None
                
                svm_degree = st.slider("Polynomial Degree", 2, 5, 3, 1, key="svm_degree") if svm_kernel == 'poly' else 3
                svm_max_iter = st.slider("Max Iterations", 1000, 10000, 5000, 1000, key="svm_maxiter")
            else:
                svm_kernel, svm_gamma, svm_gamma_custom, svm_degree, svm_max_iter = 'rbf', 'scale', None, 3, 5000
        
        with st.sidebar.expander("🔍 LOF Parameters"):
            if algorithms['Local Outlier Factor']:
                lof_algorithm = st.selectbox("Algorithm", 
                                           ['auto', 'ball_tree', 'kd_tree', 'brute'], 
                                           index=0, key="lof_algo")
                lof_leaf_size = st.slider("Leaf Size", 10, 50, 30, 5, key="lof_leaf")
                lof_metric = st.selectbox("Distance Metric", 
                                        ['minkowski', 'euclidean', 'manhattan'], 
                                        index=0, key="lof_metric")
                lof_p = st.slider("Minkowski Parameter", 1, 3, 2, 1, key="lof_p") if lof_metric == 'minkowski' else 2
            else:
                lof_algorithm, lof_leaf_size, lof_metric, lof_p = 'auto', 30, 'minkowski', 2
        
        with st.sidebar.expander("🌐 DBSCAN Parameters"):
            if algorithms['DBSCAN']:
                dbscan_eps = st.slider("Eps (Neighborhood Distance)", 0.1, 2.0, 0.5, 0.1, key="dbscan_eps")
                dbscan_min_samples = st.slider("Min Samples per Cluster", 2, 20, 5, 1, key="dbscan_min")
                dbscan_algorithm = st.selectbox("Algorithm", 
                                              ['auto', 'ball_tree', 'kd_tree', 'brute'], 
                                              index=0, key="dbscan_algo")
                dbscan_metric = st.selectbox("Distance Metric", 
                                           ['euclidean', 'manhattan', 'minkowski'], 
                                           index=0, key="dbscan_metric")
            else:
                dbscan_eps, dbscan_min_samples, dbscan_algorithm, dbscan_metric = 0.5, 5, 'auto', 'euclidean'
        
        # Performance optimization section
        st.sidebar.subheader("⚡ Performance Optimization")
        
        with st.sidebar.expander("🚀 Speed vs Accuracy"):
            performance_mode = st.radio("Performance Mode", 
                                      ['Balanced', 'Speed Optimized', 'Accuracy Optimized'], 
                                      index=0, key="perf_mode")
            
            if performance_mode == 'Speed Optimized':
                st.info("🏃‍♂️ Optimized for speed - reduced parameters")
                # Override some parameters for speed
                iso_n_estimators = min(iso_n_estimators, 100)
                kmeans_n_init = min(kmeans_n_init, 5)
                svm_max_iter = min(svm_max_iter, 2000)
                
            elif performance_mode == 'Accuracy Optimized':
                st.info("🎯 Optimized for accuracy - enhanced parameters")
                # Override some parameters for accuracy
                iso_n_estimators = max(iso_n_estimators, 200)
                kmeans_n_init = max(kmeans_n_init, 15)
                svm_max_iter = max(svm_max_iter, 8000)
        
        # Parameter validation and recommendations
        st.sidebar.subheader("💡 Parameter Recommendations")
        
        with st.sidebar.expander("📊 Auto-Tuning Suggestions"):
            dataset_size = len(st.session_state.detector.data) if hasattr(st.session_state.detector, 'data') and st.session_state.detector.data is not None else 0
            
            if dataset_size > 0:
                # Provide recommendations based on dataset size
                if dataset_size < 1000:
                    st.warning("🔸 Small dataset detected")
                    st.write("• Reduce contamination (0.02-0.03)")
                    st.write("• Use fewer clusters (3-5)")
                    st.write("• Lower LOF neighbors (5-10)")
                    st.write("• Use 'lloyd' algorithm for K-means")
                elif dataset_size > 50000:
                    st.info("🔸 Large dataset detected")
                    st.write("• Consider speed optimization")
                    st.write("• Use more clusters (15-25)")
                    st.write("• Higher LOF neighbors (25-40)")
                    st.write("• Use 'elkan' algorithm for K-means (faster for large datasets)")
                else:
                    st.success("🔸 Medium dataset - current defaults should work well")
                    st.write("• 'lloyd' algorithm recommended for K-means")
                
                # Contamination rate validation
                expected_anomaly_rate = contamination * 100
                st.write(f"📈 Expecting ~{expected_anomaly_rate:.1f}% anomalies")
                
                if hasattr(st.session_state.detector, 'labels') and st.session_state.detector.labels is not None:
                    actual_anomaly_rate = (st.session_state.detector.labels != st.session_state.detector.labels.value_counts().index[0]).mean() * 100
                    diff = abs(expected_anomaly_rate - actual_anomaly_rate)
                    
                    if diff > 2:
                        st.warning(f"⚠️ Expected ({expected_anomaly_rate:.1f}%) vs Actual ({actual_anomaly_rate:.1f}%) anomaly rates differ significantly")
                        st.write(f"💡 Consider setting contamination to {actual_anomaly_rate/100:.3f}")
        
        # Parameter summary display
        if st.session_state.analysis_complete:
            st.sidebar.subheader("📋 Last Analysis Summary")
            
            with st.sidebar.expander("📊 Parameter Summary", expanded=True):
                st.write(f"**Contamination:** {contamination}")
                st.write(f"**LOF Neighbors:** {n_neighbors}")
                st.write(f"**Performance Mode:** {performance_mode}")
                
                if algorithms['Isolation Forest']:
                    final_iso_estimators = iso_n_estimators
                    if performance_mode == 'Speed Optimized':
                        final_iso_estimators = min(iso_n_estimators, 100)
                    elif performance_mode == 'Accuracy Optimized':
                        final_iso_estimators = max(iso_n_estimators, 200)
                    st.write(f"**ISO Trees:** {final_iso_estimators}")
                
                if algorithms['K-means Clustering']:
                    st.write(f"**K-means Clusters:** {kmeans_n_clusters}")
                
                if algorithms['One-Class SVM']:
                    st.write(f"**SVM Kernel:** {svm_kernel}")
                
                if algorithms['DBSCAN']:
                    st.write(f"**DBSCAN Eps:** {dbscan_eps}")
        
        # Quick parameter optimization suggestions
        if st.session_state.analysis_complete and hasattr(st.session_state.detector, 'outliers'):
            st.sidebar.subheader("🎯 Quick Optimizations")
            
            outlier_counts = [len(outliers) for outliers in st.session_state.detector.outliers.values()]
            if len(outlier_counts) > 0:
                min_outliers = min(outlier_counts)
                max_outliers = max(outlier_counts)
                
                if max_outliers > min_outliers * 3:  # High variability
                    st.sidebar.warning("⚠️ High method disagreement")
                    st.sidebar.write("Try adjusting:")
                    st.sidebar.write("• Contamination rate")
                    st.sidebar.write("• LOF neighbors")
                    
                avg_outliers = np.mean(outlier_counts)
                total_samples = len(st.session_state.detector.data)
                outlier_rate = avg_outliers / total_samples
                
                if outlier_rate < 0.005:  # < 0.5%
                    st.sidebar.info("💡 Very few outliers detected")
                    st.sidebar.write("Consider:")
                    st.sidebar.write("• Increase contamination")
                    st.sidebar.write("• Decrease LOF neighbors")
                    
                elif outlier_rate > 0.15:  # > 15%
                    st.sidebar.warning("💡 Many outliers detected")
                    st.sidebar.write("Consider:")
                    st.sidebar.write("• Decrease contamination")
                    st.sidebar.write("• Increase LOF neighbors")
        
        # Analysis button
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Loading and analyzing data..."):
                # Load data
                success = st.session_state.detector.load_and_prepare_data(uploaded_file, max_rows)
                
                if success:
                    st.session_state.analysis_complete = True
                    
                    # Run selected algorithms
                    progress_bar = st.progress(0)
                    total_algorithms = sum(algorithms.values())
                    current_step = 0
                    
                    if algorithms['Isolation Forest']:
                        # Apply performance mode adjustments
                        final_iso_estimators = iso_n_estimators
                        if performance_mode == 'Speed Optimized':
                            final_iso_estimators = min(iso_n_estimators, 100)
                        elif performance_mode == 'Accuracy Optimized':
                            final_iso_estimators = max(iso_n_estimators, 200)
                        
                        st.session_state.detector.isolation_forest_detection(
                            contamination=contamination,
                            n_estimators=final_iso_estimators,
                            max_features=iso_max_features,
                            bootstrap=iso_bootstrap
                        )
                        current_step += 1
                        progress_bar.progress(current_step / total_algorithms)
                    
                    if algorithms['Local Outlier Factor']:
                        # Apply performance mode adjustments
                        final_lof_neighbors = n_neighbors
                        if performance_mode == 'Speed Optimized':
                            final_lof_neighbors = min(n_neighbors, 15)
                        elif performance_mode == 'Accuracy Optimized':
                            final_lof_neighbors = max(n_neighbors, 25)
                        
                        st.session_state.detector.local_outlier_factor_detection(
                            n_neighbors=final_lof_neighbors,
                            contamination=contamination,
                            algorithm=lof_algorithm,
                            leaf_size=lof_leaf_size,
                            metric=lof_metric,
                            p=lof_p
                        )
                        current_step += 1
                        progress_bar.progress(current_step / total_algorithms)
                    
                    if algorithms['K-means Clustering']:
                        # Apply performance mode adjustments
                        final_kmeans_init = kmeans_n_init
                        if performance_mode == 'Speed Optimized':
                            final_kmeans_init = min(kmeans_n_init, 5)
                        elif performance_mode == 'Accuracy Optimized':
                            final_kmeans_init = max(kmeans_n_init, 15)
                        
                        st.session_state.detector.kmeans_outlier_detection(
                            n_clusters=kmeans_n_clusters,
                            contamination=contamination,
                            algorithm=kmeans_algorithm,
                            init=kmeans_init,
                            n_init=final_kmeans_init
                        )
                        current_step += 1
                        progress_bar.progress(current_step / total_algorithms)
                    
                    if algorithms['One-Class SVM']:
                        # Apply performance mode adjustments
                        final_svm_maxiter = svm_max_iter
                        if performance_mode == 'Speed Optimized':
                            final_svm_maxiter = min(svm_max_iter, 2000)
                        elif performance_mode == 'Accuracy Optimized':
                            final_svm_maxiter = max(svm_max_iter, 8000)
                        
                        # Handle custom gamma
                        final_gamma = svm_gamma if svm_gamma_custom is None else svm_gamma_custom
                        
                        st.session_state.detector.one_class_svm_detection(
                            contamination=contamination,
                            kernel=svm_kernel,
                            gamma=final_gamma,
                            degree=svm_degree,
                            max_iter=final_svm_maxiter
                        )
                        current_step += 1
                        progress_bar.progress(current_step / total_algorithms)
                    
                    if algorithms['DBSCAN']:
                        st.session_state.detector.dbscan_clustering(
                            eps=dbscan_eps,
                            min_samples=dbscan_min_samples,
                            algorithm=dbscan_algorithm,
                            metric=dbscan_metric
                        )
                        current_step += 1
                        progress_bar.progress(current_step / total_algorithms)
                    
                    # Generate confidence scores and analyze patterns
                    st.session_state.detector.generate_confidence_scores()
                    st.session_state.detector.analyze_attack_patterns()
                    
                    progress_bar.progress(1.0)
                    st.success("Analysis complete!")
    
    # Main content area
    if st.session_state.analysis_complete and st.session_state.detector.data is not None:
        display_results()
    elif uploaded_file is not None:
        st.info("Configure your analysis parameters in the sidebar and click 'Run Analysis'")
    else:
        display_welcome()

def display_welcome():
    """Display welcome screen with instructions"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### 🎯 Welcome to Advanced Anomaly Detection
        
        This dashboard provides comprehensive anomaly detection for network traffic data using multiple machine learning algorithms:
        
        **🔧 Available Algorithms:**
        - **Isolation Forest**: Tree-based isolation method
        - **Local Outlier Factor**: Density-based detection
        - **K-means Clustering**: Distance-based detection
        - **One-Class SVM**: Boundary-based detection
        - **DBSCAN**: Cluster-based outlier identification
        
        **📊 Features:**
        - Interactive visualizations
        - Method comparison and ensemble analysis
        - Attack pattern analysis
        - Performance metrics and ROC curves
        - Confidence scoring
        - Export capabilities
        
        **🚀 Getting Started:**
        1. Upload your CSV file using the sidebar
        2. Configure detection parameters
        3. Select algorithms to run
        4. Click "Run Analysis"
        5. Explore the results in multiple tabs
        
        **📋 Data Requirements:**
        - CSV format with numeric features
        - Network traffic or similar time-series data
        - Optional 'Label' column for ground truth evaluation
        """)

def display_results():
    """Display analysis results in tabs"""
    detector = st.session_state.detector
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🔍 Detection Results", 
        "📈 Visualizations", 
        "🎯 Attack Analysis",
        "📋 Performance",
        "💾 Export"
    ])
    
    with tab1:
        display_overview_tab(detector)
    
    with tab2:
        display_detection_results_tab(detector)
    
    with tab3:
        display_visualizations_tab(detector)
    
    with tab4:
        display_attack_analysis_tab(detector)
    
    with tab5:
        display_performance_tab(detector)
    
    with tab6:
        display_export_tab(detector)

def display_overview_tab(detector):
    """Display overview statistics"""
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(detector.data):,}")
    
    with col2:
        st.metric("Features", len(detector.data.columns))
    
    with col3:
        st.metric("Security Features", len(detector.security_features))
    
    with col4:
        total_outliers = len(set().union(*detector.outliers.values())) if detector.outliers else 0
        st.metric("Total Outliers", total_outliers)
    
    # Sampling information
    if hasattr(detector, 'is_sampled') and detector.is_sampled:
        st.warning(f"📊 **Sampling Applied**: This analysis uses {len(detector.data):,} samples from the original dataset.")
        if hasattr(detector, 'sample_size'):
            st.info(f"Requested sample size: {detector.sample_size:,}")
    
    # Label information
    if detector.labels is not None:
        st.success(f"✅ **Ground Truth Available**: {len(detector.labels):,} labels found")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Label distribution
            label_counts = detector.labels.value_counts()
            st.subheader("🏷️ Label Distribution")
            
            for label, count in label_counts.items():
                percentage = (count / len(detector.labels)) * 100
                st.write(f"• **{label}**: {count:,} ({percentage:.1f}%)")
        
        with col2:
            # Label distribution pie chart
            fig_labels = go.Figure(data=[go.Pie(
                labels=label_counts.index,
                values=label_counts.values,
                hole=0.3
            )])
            fig_labels.update_layout(
                title="Label Distribution",
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig_labels, use_container_width=True)
    else:
        st.info("ℹ️ **No Ground Truth**: No 'Label' column found. Performance evaluation will be limited.")
    
    # Data quality indicators
    st.subheader("🔍 Data Quality")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Missing values check
        if hasattr(detector, 'original_data'):
            missing_pct = (detector.original_data.isnull().sum().sum() / (detector.original_data.shape[0] * detector.original_data.shape[1])) * 100
            st.metric("Missing Values", f"{missing_pct:.2f}%")
        else:
            st.metric("Missing Values", "0.00%")
    
    with col2:
        # Numeric features ratio
        numeric_ratio = (len(detector.data.columns) / len(detector.original_data.columns)) * 100
        st.metric("Numeric Features", f"{numeric_ratio:.1f}%")
    
    with col3:
        # Data consistency score
        if detector.labels is not None and len(detector.labels) == len(detector.data):
            consistency_score = 100.0
        elif detector.labels is not None:
            consistency_score = 50.0  # Labels exist but length mismatch
        else:
            consistency_score = 75.0  # No labels but data is consistent
        
        color = "normal" if consistency_score == 100 else ("inverse" if consistency_score < 75 else "off")
        st.metric("Data Consistency", f"{consistency_score:.0f}%", delta_color=color)
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(detector.data.head(), use_container_width=True)
    
    # Feature statistics
    st.subheader("📈 Feature Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature types
        numeric_features = len(detector.data.select_dtypes(include=[np.number]).columns)
        
        fig = go.Figure(data=[
            go.Bar(x=['Numeric Features', 'Security Features'], 
                   y=[numeric_features, len(detector.security_features)],
                   marker_color=['lightblue', 'lightcoral'])
        ])
        fig.update_layout(title="Feature Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top security features by variance
        if len(detector.security_features) > 0:
            security_data = detector.data[detector.security_features]
            feature_vars = security_data.var().sort_values(ascending=False).head(5)
            
            fig = go.Figure(data=[
                go.Bar(x=feature_vars.values, 
                       y=[f[:15] + '...' if len(f) > 15 else f for f in feature_vars.index],
                       orientation='h',
                       marker_color='lightgreen')
            ])
            fig.update_layout(title="Top Security Features (by variance)", height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No security features identified in the dataset")
    
    # Show any anomaly score validation issues
    if hasattr(detector, 'anomaly_scores') and detector.anomaly_scores:
        st.subheader("🔧 Anomaly Score Validation")
        
        score_status = []
        for method, scores in detector.anomaly_scores.items():
            status = "✅"
            issues = []
            
            if len(scores) != len(detector.data):
                status = "❌"
                issues.append(f"Length mismatch ({len(scores)} vs {len(detector.data)})")
            
            if np.any(np.isnan(scores)):
                status = "⚠️"
                issues.append(f"{np.sum(np.isnan(scores))} NaN values")
            
            if np.var(scores) == 0:
                status = "⚠️"
                issues.append("No variance")
            
            score_status.append({
                'Method': method.replace('_', ' ').title(),
                'Status': status,
                'Issues': ', '.join(issues) if issues else 'None',
                'Score Range': f"[{np.min(scores[~np.isnan(scores)]):.3f}, {np.max(scores[~np.isnan(scores)]):.3f}]" if not np.all(np.isnan(scores)) else "All NaN"
            })
        
        if score_status:
            score_df = pd.DataFrame(score_status)
            st.dataframe(score_df, use_container_width=True)
            
            # Summary
            good_scores = sum(1 for s in score_status if s['Status'] == '✅')
            total_scores = len(score_status)
            
            if good_scores == total_scores:
                st.success(f"🎉 All {total_scores} anomaly score arrays are valid!")
            else:
                st.warning(f"⚠️ {good_scores}/{total_scores} anomaly score arrays are fully valid. Check issues above.")
    
    # Current Analysis Configuration
    st.subheader("⚙️ Current Analysis Configuration")
    
    # Check if we're in session state and have run analysis
    if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
        
        # Create configuration display
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.markdown("**🎛️ Basic Parameters:**")
            # These would come from the sidebar state - since we can't access them directly,
            # we'll show default values and note that this shows the last run configuration
            st.write("• Contamination Rate: Check sidebar for current setting")
            st.write("• LOF Neighbors: Check sidebar for current setting")
            
            st.markdown("**🚀 Performance Mode:**")
            st.write("• Mode: Check sidebar for current setting")
            
        with config_col2:
            st.markdown("**🔧 Advanced Parameters:**")
            st.write("**Isolation Forest:**")
            st.write("• Trees: Configured via advanced panel")
            st.write("• Max Features: Configured via advanced panel")
            
            st.write("**K-means:**")
            st.write("• Clusters: Configured via advanced panel")
            st.write("• Algorithm: Configured via advanced panel")
            
            st.write("**SVM:**")
            st.write("• Kernel: Configured via advanced panel")
            st.write("• Gamma: Configured via advanced panel")
        
        st.info("💡 **Parameter Tip**: The exact parameters used are configured in the sidebar. "
               "Expand the 'Advanced Hyperparameter Tuning' sections to see and modify all available options.")
        
        # Performance recommendations based on current results
        if detector.outliers:
            total_methods = len(detector.outliers)
            total_outliers = len(set().union(*detector.outliers.values()))
            outlier_rate = total_outliers / len(detector.data) * 100
            
            st.markdown("**📊 Configuration Assessment:**")
            
            if outlier_rate < 1:
                st.warning("🔍 Very few outliers detected - consider increasing contamination rate")
            elif outlier_rate > 10:
                st.warning("⚠️ High outlier rate - consider decreasing contamination rate or checking data quality")
            else:
                st.success("✅ Outlier detection rate appears reasonable")
            
            # Method agreement assessment
            if total_methods > 1:
                method_outliers = [len(outliers) for outliers in detector.outliers.values()]
                std_dev = np.std(method_outliers)
                mean_outliers = np.mean(method_outliers)
                cv = std_dev / mean_outliers if mean_outliers > 0 else 0
                
                if cv > 0.5:
                    st.info("📈 High variability between methods - consider parameter tuning")
                else:
                    st.success("🤝 Good agreement between detection methods")
    
    else:
        st.info("⏳ Run analysis to see configuration assessment and parameter recommendations")
        
        # Show parameter tuning guidance
        st.markdown("**🎯 Parameter Tuning Guide:**")
        st.write("1. **Start with contamination rate** - Match your expected anomaly percentage")
        st.write("2. **Adjust algorithm-specific parameters** - Use the advanced panels in sidebar")
        st.write("3. **Check method agreement** - Multiple methods should find similar patterns")
        st.write("4. **Validate with ground truth** - If available, use Performance tab")
        st.write("5. **Iterate based on results** - Use the recommendations that appear after analysis")


def display_detection_results_tab(detector):
    """Display detection results"""
    st.subheader("🔍 Detection Results by Method")
    
    if not detector.outliers:
        st.warning("No detection results available. Please run the analysis first.")
        return
    
    # Method comparison
    method_data = []
    for method, outliers in detector.outliers.items():
        percentage = (len(outliers) / len(detector.data)) * 100
        method_data.append({
            'Method': method.replace('_', ' ').title(),
            'Outliers': len(outliers),
            'Percentage': percentage
        })
    
    results_df = pd.DataFrame(method_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Method Comparison")
        st.dataframe(results_df, use_container_width=True)
    
    with col2:
        # Bar chart
        fig = px.bar(
            results_df, 
            x='Method', 
            y='Outliers',
            title="Outliers Detected by Method",
            color='Percentage',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Confidence scores
    if 'ensemble' in detector.confidence_scores:
        st.subheader("🎯 Confidence Analysis")
        
        confidence_scores = detector.confidence_scores['ensemble']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_conf = np.sum(confidence_scores >= 0.7)
            st.metric("High Confidence (≥70%)", high_conf)
        
        with col2:
            medium_conf = np.sum((confidence_scores >= 0.5) & (confidence_scores < 0.7))
            st.metric("Medium Confidence (50-70%)", medium_conf)
        
        with col3:
            low_conf = np.sum((confidence_scores > 0) & (confidence_scores < 0.5))
            st.metric("Low Confidence (<50%)", low_conf)
        
        # Confidence distribution
        fig = px.histogram(
            confidence_scores[confidence_scores > 0],
            nbins=20,
            title="Confidence Score Distribution",
            labels={'value': 'Confidence Score', 'count': 'Number of Outliers'}
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange", annotation_text="Medium")
        fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High")
        st.plotly_chart(fig, use_container_width=True)

def display_visualizations_tab(detector):
    """Display interactive visualizations"""
    st.subheader("📈 Interactive Visualizations")
    
    if not detector.outliers:
        st.warning("No detection results available for visualization.")
        return
    
    # PCA visualization
    st.subheader("🔬 PCA Analysis")
    
    # Always perform 2D PCA for method-specific analysis
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(detector.scaled_features)
    
    # Create outlier mask
    all_outliers = set().union(*detector.outliers.values())
    is_outlier = np.array([i in all_outliers for i in range(len(detector.data))])
    
    # PCA visualization options
    pca_view = st.radio("Select PCA View:", ["2D View", "3D View"], horizontal=True)
    
    if pca_view == "2D View":
        # Create 2D PCA plot
        fig = go.Figure()
        
        # Normal points
        fig.add_trace(go.Scatter(
            x=pca_features[~is_outlier, 0],
            y=pca_features[~is_outlier, 1],
            mode='markers',
            name='Normal',
            marker=dict(color='lightblue', size=4, opacity=0.6)
        ))
        
        # Outlier points
        fig.add_trace(go.Scatter(
            x=pca_features[is_outlier, 0],
            y=pca_features[is_outlier, 1],
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=6, opacity=0.8)
        ))
        
        fig.update_layout(
            title=f"2D PCA Visualization (PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # 3D View
        # Perform 3D PCA
        pca_3d = PCA(n_components=3)
        pca_features_3d = pca_3d.fit_transform(detector.scaled_features)
        
        # Create 3D PCA plot
        fig_3d = go.Figure()
        
        # Normal points
        fig_3d.add_trace(go.Scatter3d(
            x=pca_features_3d[~is_outlier, 0],
            y=pca_features_3d[~is_outlier, 1],
            z=pca_features_3d[~is_outlier, 2],
            mode='markers',
            name='Normal',
            marker=dict(
                color='lightblue',
                size=3,
                opacity=0.6,
                line=dict(width=0)
            ),
            text=[f"Sample {i}" for i in range(len(detector.data)) if not is_outlier[i]],
            hovertemplate="<b>Normal Traffic</b><br>" +
                         "PC1: %{x:.3f}<br>" +
                         "PC2: %{y:.3f}<br>" +
                         "PC3: %{z:.3f}<br>" +
                         "%{text}<extra></extra>"
        ))
        
        # Outlier points
        fig_3d.add_trace(go.Scatter3d(
            x=pca_features_3d[is_outlier, 0],
            y=pca_features_3d[is_outlier, 1],
            z=pca_features_3d[is_outlier, 2],
            mode='markers',
            name='Outliers',
            marker=dict(
                color='red',
                size=5,
                opacity=0.9,
                line=dict(width=1, color='darkred')
            ),
            text=[f"Outlier {i}" for i in range(len(detector.data)) if is_outlier[i]],
            hovertemplate="<b>Anomalous Traffic</b><br>" +
                         "PC1: %{x:.3f}<br>" +
                         "PC2: %{y:.3f}<br>" +
                         "PC3: %{z:.3f}<br>" +
                         "%{text}<extra></extra>"
        ))
        
        # Add confidence scores if available
        if 'ensemble' in detector.confidence_scores:
            confidence_scores = detector.confidence_scores['ensemble']
            high_conf_mask = (confidence_scores >= 0.7) & is_outlier
            
            if np.any(high_conf_mask):
                fig_3d.add_trace(go.Scatter3d(
                    x=pca_features_3d[high_conf_mask, 0],
                    y=pca_features_3d[high_conf_mask, 1],
                    z=pca_features_3d[high_conf_mask, 2],
                    mode='markers',
                    name='High Confidence Outliers',
                    marker=dict(
                        color='orange',
                        size=7,
                        opacity=1.0,
                        symbol='diamond',
                        line=dict(width=2, color='darkorange')
                    ),
                    text=[f"High Conf {i} (Score: {confidence_scores[i]:.3f})" 
                          for i in range(len(detector.data)) if high_conf_mask[i]],
                    hovertemplate="<b>High Confidence Anomaly</b><br>" +
                                 "PC1: %{x:.3f}<br>" +
                                 "PC2: %{y:.3f}<br>" +
                                 "PC3: %{z:.3f}<br>" +
                                 "%{text}<extra></extra>"
                ))
        
        fig_3d.update_layout(
            title=f"3D PCA Visualization (PC1: {pca_3d.explained_variance_ratio_[0]:.1%}, "
                  f"PC2: {pca_3d.explained_variance_ratio_[1]:.1%}, "
                  f"PC3: {pca_3d.explained_variance_ratio_[2]:.1%})",
            scene=dict(
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                zaxis_title="Principal Component 3",
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="lightgray"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="lightgray"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="lightgray"),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # 3D PCA insights
        st.info(f"💡 **3D PCA Insights**: The first 3 components explain "
                f"{pca_3d.explained_variance_ratio_[:3].sum():.1%} of the total variance. "
                f"Rotate and zoom the plot to explore anomaly clusters in 3D space!")
    
    # PCA explained variance
    pca_var = PCA(n_components=min(10, detector.scaled_features.shape[1]))
    pca_var.fit(detector.scaled_features)
    
    # Variance explanation chart
    fig_var = go.Figure()
    
    components = list(range(1, len(pca_var.explained_variance_ratio_) + 1))
    explained_var = pca_var.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    fig_var.add_trace(go.Bar(
        x=components,
        y=explained_var,
        name='Individual',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig_var.add_trace(go.Scatter(
        x=components,
        y=cumulative_var,
        mode='lines+markers',
        name='Cumulative',
        line=dict(color='red', width=3),
        marker=dict(size=8, color='red'),
        yaxis='y2'
    ))
    
    fig_var.update_layout(
        title="PCA Explained Variance Analysis",
        xaxis_title="Principal Component",
        yaxis_title="Individual Variance Explained",
        yaxis2=dict(
            title="Cumulative Variance Explained",
            overlaying='y',
            side='right'
        ),
        height=400
    )
    
    st.plotly_chart(fig_var, use_container_width=True)
    
    # Method-specific visualizations
    st.subheader("🔍 Method-Specific Analysis")
    
    # Debug information
    available_methods = list(detector.outliers.keys())
    
    if len(available_methods) == 0:
        st.warning("⚠️ No detection methods have been run yet. Please run the analysis first.")
        return
    elif len(available_methods) == 1:
        st.info(f"ℹ️ Only one method available: {available_methods[0]}. Select more algorithms in the sidebar and run analysis again for comparison.")
    
    # Show method status
    with st.expander("🔍 Method Execution Status"):
        method_status = []
        total_samples = len(detector.data)
        
        for method in available_methods:
            outliers = detector.outliers[method]
            has_scores = method in detector.anomaly_scores
            score_info = "✅ Available" if has_scores else "❌ Missing"
            
            method_status.append({
                'Method': method.replace('_', ' ').title(),
                'Outliers Detected': len(outliers),
                'Detection Rate': f"{len(outliers)/total_samples*100:.2f}%",
                'Anomaly Scores': score_info
            })
        
        status_df = pd.DataFrame(method_status)
        st.dataframe(status_df, use_container_width=True)
        
        if len(available_methods) < 3:
            st.info("💡 **Tip**: Enable more algorithms in the sidebar (Local Outlier Factor, K-means, One-Class SVM, DBSCAN) for comprehensive analysis.")
    
    selected_method = st.selectbox(
        "Select method for detailed analysis:",
        available_methods,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if selected_method in detector.anomaly_scores:
        scores = detector.anomaly_scores[selected_method]
        outlier_indices = detector.outliers[selected_method]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            fig = go.Figure()
            
            # Normal scores
            normal_mask = np.ones(len(scores), dtype=bool)
            normal_mask[outlier_indices] = False
            
            fig.add_trace(go.Histogram(
                x=scores[normal_mask],
                name='Normal',
                opacity=0.7,
                nbinsx=30
            ))
            
            # Outlier scores
            if len(outlier_indices) > 0:
                fig.add_trace(go.Histogram(
                    x=scores[outlier_indices],
                    name='Outliers',
                    opacity=0.7,
                    nbinsx=15
                ))
            
            fig.update_layout(
                title=f"{selected_method} Score Distribution",
                xaxis_title="Anomaly Score",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # PCA colored by scores
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pca_features[:, 0],
                y=pca_features[:, 1],
                mode='markers',
                marker=dict(
                    color=scores,
                    colorscale='RdYlBu',
                    size=5,
                    colorbar=dict(title="Anomaly Score")
                ),
                text=[f"Sample {i}<br>Score: {score:.3f}" for i, score in enumerate(scores)],
                hovertemplate="%{text}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"PCA colored by {selected_method} Scores",
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_attack_analysis_tab(detector):
    """Display attack pattern analysis"""
    st.subheader("🎯 Attack Pattern Analysis")
    
    if not detector.attack_analysis:
        st.warning("No attack analysis available. Analysis may not have been performed.")
        return
    
    # Target services analysis
    if 'target_services' in detector.attack_analysis:
        st.subheader("📡 Target Services")
        
        target_data = detector.attack_analysis['target_services']
        
        # Create dataframe for display
        services_df = pd.DataFrame([
            {
                'Service': service,
                'Port': data['port'],
                'Attack Count': data['count'],
                'Percentage': f"{data['percentage']:.1f}%"
            }
            for service, data in target_data.items()
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(services_df, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = px.pie(
                values=[data['count'] for data in target_data.values()],
                names=list(target_data.keys()),
                title="Attack Distribution by Target Service"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    if 'feature_analysis' in detector.attack_analysis:
        st.subheader("📊 Feature Characteristics")
        
        feature_data = detector.attack_analysis['feature_analysis']
        
        feature_df = pd.DataFrame([
            {
                'Feature': feature,
                'Normal Mean': f"{data['normal_mean']:.3f}",
                'Outlier Mean': f"{data['outlier_mean']:.3f}",
                'Difference %': f"{data['difference_pct']:+.1f}%"
            }
            for feature, data in feature_data.items()
        ])
        
        st.dataframe(feature_df, use_container_width=True)
        
        # Feature comparison chart
        features = list(feature_data.keys())
        normal_means = [data['normal_mean'] for data in feature_data.values()]
        outlier_means = [data['outlier_mean'] for data in feature_data.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=features,
            y=normal_means,
            name='Normal Traffic',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=features,
            y=outlier_means,
            name='Anomalous Traffic',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Feature Comparison: Normal vs Anomalous Traffic",
            xaxis_title="Features",
            yaxis_title="Mean Value",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_performance_tab(detector):
    """Display performance metrics"""
    st.subheader("📋 Performance Analysis")
    
    # Always show method agreement analysis first
    st.subheader("🤝 Method Agreement Analysis")
    
    methods = list(detector.outliers.keys())
    if len(methods) >= 2:
        # Method agreement matrix
        agreement_matrix = np.zeros((len(methods), len(methods)))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    set1 = set(detector.outliers[method1])
                    set2 = set(detector.outliers[method2])
                    if len(set1) > 0 and len(set2) > 0:
                        jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                        agreement_matrix[i, j] = jaccard
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.imshow(
                agreement_matrix,
                x=[m.replace('_', ' ').title() for m in methods],
                y=[m.replace('_', ' ').title() for m in methods],
                title="Method Agreement Matrix (Jaccard Similarity)",
                color_continuous_scale='Blues',
                text_auto='.2f'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Method overlap statistics
            st.subheader("📊 Overlap Statistics")
            
            overlap_data = []
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods[i+1:], i+1):
                    set1 = set(detector.outliers[method1])
                    set2 = set(detector.outliers[method2])
                    
                    if len(set1) > 0 and len(set2) > 0:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        jaccard = intersection / union if union > 0 else 0
                        
                        overlap_data.append({
                            'Method Pair': f"{method1[:8]} ↔ {method2[:8]}",
                            'Common Outliers': intersection,
                            'Jaccard Score': f"{jaccard:.3f}"
                        })
            
            if overlap_data:
                overlap_df = pd.DataFrame(overlap_data)
                overlap_df = overlap_df.sort_values('Common Outliers', ascending=False)
                st.dataframe(overlap_df, use_container_width=True)
    else:
        st.info("Need at least 2 methods for agreement analysis.")
    
    # Detection efficiency analysis
    st.subheader("⚡ Detection Efficiency")
    
    efficiency_data = []
    total_samples = len(detector.data)
    
    for method, outliers in detector.outliers.items():
        detection_rate = len(outliers) / total_samples * 100
        efficiency_data.append({
            'Method': method.replace('_', ' ').title(),
            'Outliers Detected': len(outliers),
            'Detection Rate (%)': f"{detection_rate:.2f}%",
            'Samples Processed': total_samples
        })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    st.dataframe(efficiency_df, use_container_width=True)
    
    # Detection rate visualization
    fig_eff = px.bar(
        efficiency_df,
        x='Method',
        y='Outliers Detected',
        title="Outliers Detected by Method",
        color='Outliers Detected',
        color_continuous_scale='Reds'
    )
    fig_eff.update_layout(height=400)
    st.plotly_chart(fig_eff, use_container_width=True)
    
    # Ground truth evaluation (if available)
    if detector.labels is not None:
        st.subheader("🎯 Ground Truth Evaluation")
        
        try:
            # Validate data consistency first
            if len(detector.labels) != len(detector.data):
                st.error(f"❌ **Data Inconsistency Detected**: Labels ({len(detector.labels)}) and data ({len(detector.data)}) have different lengths.")
                st.info("This usually happens when sampling is applied. Please ensure the 'Label' column is included when uploading sampled data.")
                return
            
            # Convert labels to binary
            unique_labels = detector.labels.unique()
            st.info(f"Ground truth labels found: {', '.join(map(str, unique_labels))}")
            
            if 'BENIGN' in unique_labels:
                y_true = (detector.labels != 'BENIGN').astype(int)
            else:
                # Try to identify normal vs anomalous labels
                label_counts = detector.labels.value_counts()
                most_common_label = label_counts.index[0]
                y_true = (detector.labels != most_common_label).astype(int)
                st.info(f"Using '{most_common_label}' as normal class")
            
            # Additional validation
            max_outlier_index = max([max(outliers) if len(outliers) > 0 else -1 for outliers in detector.outliers.values()])
            if max_outlier_index >= len(y_true):
                st.error(f"❌ **Index Error**: Maximum outlier index ({max_outlier_index}) exceeds data length ({len(y_true)}).")
                st.info("This indicates an indexing problem. Please try rerunning the analysis.")
                return
            
            # Calculate metrics for each method
            performance_data = []
            
            for method, outlier_indices in detector.outliers.items():
                # Validate outlier indices
                valid_indices = [idx for idx in outlier_indices if idx < len(y_true)]
                if len(valid_indices) != len(outlier_indices):
                    st.warning(f"⚠️ {method}: {len(outlier_indices) - len(valid_indices)} invalid indices removed")
                
                y_pred = np.zeros(len(y_true))
                if len(valid_indices) > 0:
                    y_pred[valid_indices] = 1
                
                try:
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    
                    performance_data.append({
                        'Method': method.replace('_', ' ').title(),
                        'Precision': f"{precision:.3f}",
                        'Recall': f"{recall:.3f}",
                        'F1-Score': f"{f1:.3f}",
                        'True Positives': int(np.sum((y_pred == 1) & (y_true == 1))),
                        'False Positives': int(np.sum((y_pred == 1) & (y_true == 0))),
                        'False Negatives': int(np.sum((y_pred == 0) & (y_true == 1))),
                        'Valid Detections': len(valid_indices)
                    })
                except Exception as e:
                    st.warning(f"Could not calculate metrics for {method}: {str(e)}")
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                st.dataframe(perf_df, use_container_width=True)
                
                # ROC Curves
                st.subheader("📈 ROC Analysis")
                
                fig = go.Figure()
                roc_data = []
                
                for method, outlier_indices in detector.outliers.items():
                    if method in detector.anomaly_scores:
                        scores = detector.anomaly_scores[method]
                        
                        # Validate score array length
                        if len(scores) != len(y_true):
                            st.warning(f"⚠️ {method}: Score array length mismatch ({len(scores)} vs {len(y_true)}). Skipping ROC calculation.")
                            continue
                        
                        # Check for invalid scores (NaN, inf)
                        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                            valid_mask = ~(np.isnan(scores) | np.isinf(scores))
                            if np.sum(valid_mask) < len(scores) * 0.8:  # If more than 20% invalid
                                st.warning(f"⚠️ {method}: Too many invalid scores ({np.sum(~valid_mask)} invalid). Skipping ROC calculation.")
                                continue
                            else:
                                st.info(f"ℹ️ {method}: Filtering {np.sum(~valid_mask)} invalid scores")
                                scores_clean = scores[valid_mask]
                                y_true_clean = y_true[valid_mask]
                        else:
                            scores_clean = scores
                            y_true_clean = y_true
                        
                        # Check for score variance
                        if np.var(scores_clean) == 0:
                            st.warning(f"⚠️ {method}: All scores are identical (no variance). Cannot calculate ROC.")
                            continue
                        
                        # Check if we have both classes
                        if len(np.unique(y_true_clean)) < 2:
                            st.warning(f"⚠️ {method}: Only one class present in labels. Cannot calculate ROC.")
                            continue
                        
                        # Handle different score interpretations
                        if method in ['isolation_forest', 'one_class_svm']:
                            y_scores = -scores_clean  # More negative = more anomalous
                        elif method in ['lof']:
                            y_scores = scores_clean   # More negative = more anomalous (already negative)
                        elif method == 'kmeans':
                            y_scores = scores_clean   # More negative = more anomalous (negative distances)
                        elif method == 'dbscan':
                            # DBSCAN doesn't have meaningful scores for ROC
                            st.info(f"ℹ️ {method}: DBSCAN doesn't produce continuous scores suitable for ROC analysis.")
                            continue
                        else:
                            y_scores = scores_clean
                        
                        try:
                            fpr, tpr, thresholds = roc_curve(y_true_clean, y_scores)
                            roc_auc = auc(fpr, tpr)
                            
                            # Validate AUC
                            if np.isnan(roc_auc) or np.isinf(roc_auc):
                                st.warning(f"⚠️ {method}: Invalid AUC calculated ({roc_auc}). Skipping.")
                                continue
                            
                            fig.add_trace(go.Scatter(
                                x=fpr,
                                y=tpr,
                                mode='lines',
                                name=f"{method.replace('_', ' ').title()} (AUC = {roc_auc:.3f})",
                                line=dict(width=2),
                                hovertemplate=f"<b>{method.replace('_', ' ').title()}</b><br>" +
                                            "FPR: %{x:.3f}<br>" +
                                            "TPR: %{y:.3f}<br>" +
                                            f"AUC: {roc_auc:.3f}<extra></extra>"
                            ))
                            
                            roc_data.append({
                                'Method': method.replace('_', ' ').title(),
                                'AUC Score': f"{roc_auc:.3f}",
                                'Valid Samples': len(y_true_clean),
                                'Score Range': f"[{np.min(y_scores):.3f}, {np.max(y_scores):.3f}]"
                            })
                            
                        except Exception as e:
                            st.warning(f"Could not calculate ROC for {method}: {str(e)}")
                
                # Only show ROC plot if we have valid curves
                if len(roc_data) > 0:
                    # Add diagonal line
                    fig.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(dash='dash', color='gray', width=1),
                        hovertemplate="<b>Random Classifier</b><br>" +
                                    "FPR: %{x:.3f}<br>" +
                                    "TPR: %{y:.3f}<br>" +
                                    "AUC: 0.500<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        title="ROC Curves Comparison",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=500,
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1])
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("🏆 AUC Rankings")
                        roc_df = pd.DataFrame(roc_data)
                        roc_df['AUC_numeric'] = roc_df['AUC Score'].astype(float)
                        roc_df = roc_df.sort_values('AUC_numeric', ascending=False)
                        
                        # Display table without the helper column
                        display_df = roc_df.drop('AUC_numeric', axis=1)
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Best method highlight
                        if len(roc_df) > 0:
                            best_method = roc_df.iloc[0]['Method']
                            best_auc = roc_df.iloc[0]['AUC Score']
                            st.success(f"🥇 Best Method: **{best_method}** (AUC: {best_auc})")
                            
                            # Performance interpretation
                            best_auc_val = float(best_auc)
                            if best_auc_val >= 0.9:
                                st.success("🟢 Excellent performance!")
                            elif best_auc_val >= 0.8:
                                st.info("🔵 Good performance")
                            elif best_auc_val >= 0.7:
                                st.warning("🟡 Fair performance")
                            else:
                                st.error("🔴 Poor performance - consider adjusting parameters")
                else:
                    st.warning("⚠️ **No valid ROC curves generated**")
                    st.info("This can happen when:")
                    st.info("• All anomaly scores are identical (no variance)")
                    st.info("• Only one class is present in the sample")
                    st.info("• Anomaly scores contain too many invalid values")
                    st.info("• Sample size is too small for reliable ROC calculation")
            
        except Exception as e:
            st.error(f"Error in ground truth evaluation: {str(e)}")
            
            # Additional debugging information
            if hasattr(detector, 'is_sampled') and detector.is_sampled:
                st.info("💡 **Sampling detected**: This error may be due to data sampling. "
                       "Try using a smaller sample size or ensure the Label column is preserved during sampling.")
            
            # Show data dimensions for debugging
            st.info(f"🔍 **Debug Info**: Data shape: {detector.data.shape}, "
                   f"Labels length: {len(detector.labels) if detector.labels is not None else 'None'}, "
                   f"Max outlier indices: {[max(outliers) if len(outliers) > 0 else 'None' for outliers in detector.outliers.values()]}")
    
    else:
        st.info("💡 **No ground truth labels available** - Upload data with a 'Label' column for performance evaluation.")
        
        # Show sampling information if applicable
        if hasattr(detector, 'is_sampled') and detector.is_sampled:
            st.warning(f"⚠️ **Data Sampling Applied**: Analyzing {len(detector.data):,} samples "
                      f"from the original dataset. For complete performance analysis, ensure your "
                      f"sample includes the 'Label' column.")

def display_export_tab(detector):
    """Display export options"""
    st.subheader("💾 Export Results")
    
    if not detector.outliers:
        st.warning("No results to export. Please run the analysis first.")
        return
    
    # Prepare export data
    all_outliers = set().union(*detector.outliers.values())
    
    # Create results dataframe
    results_df = detector.data.copy()
    results_df['is_outlier'] = False
    results_df.loc[list(all_outliers), 'is_outlier'] = True
    
    # Add confidence scores if available
    if 'ensemble' in detector.confidence_scores:
        results_df['confidence_score'] = detector.confidence_scores['ensemble']
    
    # Add method-specific flags
    for method, outliers in detector.outliers.items():
        results_df[f'{method}_outlier'] = False
        results_df.loc[outliers, f'{method}_outlier'] = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Summary Statistics")
        st.write(f"Total samples: {len(results_df):,}")
        st.write(f"Total outliers: {len(all_outliers):,}")
        st.write(f"Outlier percentage: {len(all_outliers)/len(results_df)*100:.2f}%")
        st.write(f"Methods used: {len(detector.outliers)}")
    
    with col2:
        st.subheader("📋 Export Options")
        
        # Download full results
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )
        
        # Download outliers only
        outliers_df = results_df[results_df['is_outlier']]
        
        if len(outliers_df) > 0:
            outliers_buffer = io.StringIO()
            outliers_df.to_csv(outliers_buffer, index=False)
            
            st.download_button(
                label="📥 Download Outliers Only (CSV)",
                data=outliers_buffer.getvalue(),
                file_name="detected_outliers.csv",
                mime="text/csv"
            )
        
        # Download analysis summary
        summary_data = {
            'method': [],
            'outliers_detected': [],
            'percentage': []
        }
        
        for method, outliers in detector.outliers.items():
            summary_data['method'].append(method)
            summary_data['outliers_detected'].append(len(outliers))
            summary_data['percentage'].append(len(outliers)/len(results_df)*100)
        
        summary_df = pd.DataFrame(summary_data)
        summary_buffer = io.StringIO()
        summary_df.to_csv(summary_buffer, index=False)
        
        st.download_button(
            label="📥 Download Summary (CSV)",
            data=summary_buffer.getvalue(),
            file_name="analysis_summary.csv",
            mime="text/csv"
        )
    
    # Preview results
    st.subheader("👀 Results Preview")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        show_outliers_only = st.checkbox("Show outliers only")
    
    with col2:
        if 'ensemble' in detector.confidence_scores:
            min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.1)
        else:
            min_confidence = 0.0
    
    # Filter data
    display_df = results_df.copy()
    
    if show_outliers_only:
        display_df = display_df[display_df['is_outlier']]
    
    if 'confidence_score' in display_df.columns:
        display_df = display_df[display_df['confidence_score'] >= min_confidence]
    
    st.dataframe(display_df.head(100), use_container_width=True)
    
    if len(display_df) > 100:
        st.info(f"Showing first 100 rows out of {len(display_df)} total rows matching filters.")

if __name__ == "__main__":
    main()
