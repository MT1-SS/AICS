import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI for Incident Response Dashboard",
    page_icon="🛡️",
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
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">🛡️ AI for Incident Response Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Class 8: Accelerating Security Operations with Intelligent Automation**")

# Sidebar for controls
st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Generate or use cached data
@st.cache_data
def generate_security_events(n_events=1000):
    """Generate synthetic security events for demonstration"""
    
    # Event types and their characteristics
    event_types = {
        'Malware Detection': {'severity': 'High', 'auto_response': True, 'false_positive_rate': 0.05},
        'Phishing Email': {'severity': 'Medium', 'auto_response': True, 'false_positive_rate': 0.10},
        'Suspicious Login': {'severity': 'Medium', 'auto_response': False, 'false_positive_rate': 0.15},
        'Network Anomaly': {'severity': 'Low', 'auto_response': False, 'false_positive_rate': 0.20},
        'DDoS Attack': {'severity': 'Critical', 'auto_response': True, 'false_positive_rate': 0.02},
        'Data Exfiltration': {'severity': 'Critical', 'auto_response': True, 'false_positive_rate': 0.03},
        'Privilege Escalation': {'severity': 'High', 'auto_response': False, 'false_positive_rate': 0.08}
    }
    
    events = []
    start_time = datetime.now() - timedelta(days=30)
    
    for i in range(n_events):
        event_type = random.choices(
            list(event_types.keys()),
            weights=[0.25, 0.20, 0.30, 0.15, 0.05, 0.03, 0.02]
        )[0]
        
        characteristics = event_types[event_type]
        
        # Generate event timestamp
        event_time = start_time + timedelta(
            minutes=random.randint(0, 30*24*60)
        )
        
        # Determine if this is a true positive or false positive
        is_true_positive = random.random() > characteristics['false_positive_rate']
        
        event = {
            'event_id': f"EVT_{i+1:06d}",
            'timestamp': event_time,
            'event_type': event_type,
            'severity': characteristics['severity'],
            'source_ip': f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            'destination_ip': f"10.0.{random.randint(1,254)}.{random.randint(1,254)}",
            'user_id': f"user_{random.randint(1,500):03d}",
            'asset_id': f"asset_{random.randint(1,100):03d}",
            'confidence_score': random.uniform(0.6, 0.95) if is_true_positive else random.uniform(0.3, 0.7),
            'auto_response_enabled': characteristics['auto_response'],
            'is_true_positive': is_true_positive,
            'status': 'Open'
        }
        
        events.append(event)
    
    return pd.DataFrame(events)

@st.cache_data
def ai_incident_triage(events_df):
    """AI-powered incident triage and prioritization"""
    
    # Create features for ML model
    le_event_type = LabelEncoder()
    le_severity = LabelEncoder()
    
    events_df['event_type_encoded'] = le_event_type.fit_transform(events_df['event_type'])
    events_df['severity_encoded'] = le_severity.fit_transform(events_df['severity'])
    
    # Calculate priority score using multiple factors
    severity_weights = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
    events_df['severity_weight'] = events_df['severity'].map(severity_weights)
    
    # Priority score calculation
    events_df['priority_score'] = (
        events_df['confidence_score'] * 0.4 +
        events_df['severity_weight'] / 4 * 0.6
    )
    
    # Assign priority levels with consistent ordering
    events_df['priority'] = pd.cut(
        events_df['priority_score'],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['P4-Low', 'P3-Medium', 'P2-High', 'P1-Critical']
    )
    
    # Add time-based features for anomaly detection
    events_df['hour'] = pd.to_datetime(events_df['timestamp']).dt.hour
    events_df['day_of_week'] = pd.to_datetime(events_df['timestamp']).dt.dayofweek
    
    # Generate MTTR (Mean Time To Response) simulation
    events_df['mttr_hours'] = np.where(
        events_df['auto_response_enabled'],
        np.random.uniform(0.1, 0.5, len(events_df)),  # Automated: 6-30 minutes
        np.random.uniform(1, 8, len(events_df))       # Manual: 1-8 hours
    )
    
    # Anomaly detection
    features = events_df[['event_type_encoded', 'severity_encoded', 'confidence_score', 'hour', 'day_of_week']]
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = iso_forest.fit_predict(features)
    events_df['is_anomaly'] = anomaly_labels == -1
    
    return events_df

# Sidebar controls
n_events = st.sidebar.slider("Number of Security Events", 100, 2000, 1000, 100)
refresh_data = st.sidebar.button("🔄 Refresh Data")

# Generate data
if refresh_data:
    st.cache_data.clear()

security_events = generate_security_events(n_events)
security_events = ai_incident_triage(security_events)

# Calculate KPIs
total_events = len(security_events)
high_priority_events = len(security_events[security_events['priority'].isin(['P1-Critical', 'P2-High'])])
auto_resolved_events = len(security_events[security_events['auto_response_enabled']])
anomalous_events = security_events['is_anomaly'].sum()
avg_mttr = security_events['mttr_hours'].mean()
automation_rate = (auto_resolved_events / total_events) * 100

# Main Dashboard
st.markdown('<h2 class="section-header">📊 SOC Dashboard - Real-Time Security Metrics</h2>', unsafe_allow_html=True)

# KPI Row
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        label="Total Events",
        value=f"{total_events:,}",
        delta=f"+{random.randint(10, 50)} today"
    )

with col2:
    st.metric(
        label="High Priority",
        value=f"{high_priority_events:,}",
        delta=f"+{random.randint(2, 15)} today",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Auto-Resolved",
        value=f"{auto_resolved_events:,}",
        delta=f"{automation_rate:.1f}% rate"
    )

with col4:
    st.metric(
        label="Anomalies",
        value=f"{anomalous_events:,}",
        delta=f"+{random.randint(1, 5)} today",
        delta_color="inverse"
    )

with col5:
    st.metric(
        label="Avg MTTR",
        value=f"{avg_mttr:.1f}h",
        delta="-2.3h improvement",
        delta_color="normal"
    )

with col6:
    automation_improvement = random.uniform(15, 25)
    st.metric(
        label="Efficiency Gain",
        value=f"{automation_improvement:.1f}%",
        delta="vs. manual process"
    )

# Main Visualizations
st.markdown('<h2 class="section-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)

# Define consistent priority colors for use across multiple charts
priority_colors = {
    'P1-Critical': '#d62728',  # Red
    'P2-High': '#ff7f0e',     # Orange  
    'P3-Medium': '#2ca02c',   # Green
    'P4-Low': '#1f77b4'       # Blue
}

# Row 1: Event Volume and Priority Distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Event Volume Over Time")
    
    # Group events by day for time series
    events_by_day = security_events.copy()
    events_by_day['date'] = pd.to_datetime(events_by_day['timestamp']).dt.date
    daily_counts = events_by_day.groupby(['date', 'severity']).size().reset_index(name='count')
    
    fig_timeline = px.line(
        daily_counts, 
        x='date', 
        y='count', 
        color='severity',
        title="Security Events Timeline",
        color_discrete_map={
            'Critical': '#d62728',
            'High': '#ff7f0e', 
            'Medium': '#2ca02c',
            'Low': '#1f77b4'
        }
    )
    fig_timeline.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Events",
        hovermode='x unified'
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

with col2:
    st.subheader("🎯 Incident Priority Distribution")
    
    priority_counts = security_events['priority'].value_counts()
    
    fig_priority = px.pie(
        values=priority_counts.values,
        names=priority_counts.index,
        title="Priority Level Breakdown",
        color_discrete_map=priority_colors
    )
    fig_priority.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_priority, use_container_width=True)

# Row 2: MTTR Analysis and Event Types
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ Response Time Analysis")
    
    # MTTR comparison
    mttr_comparison = security_events.groupby('auto_response_enabled')['mttr_hours'].agg(['mean', 'std']).reset_index()
    mttr_comparison['response_type'] = mttr_comparison['auto_response_enabled'].map({
        True: 'AI-Automated',
        False: 'Manual Response'
    })
    
    fig_mttr = go.Figure()
    
    # Add bars for mean MTTR
    fig_mttr.add_trace(go.Bar(
        x=mttr_comparison['response_type'],
        y=mttr_comparison['mean'],
        error_y=dict(type='data', array=mttr_comparison['std']),
        name='Mean Time to Response',
        marker_color=['#2ca02c', '#d62728'],
        text=[f"{val:.2f}h" for val in mttr_comparison['mean']],
        textposition='auto'
    ))
    
    fig_mttr.update_layout(
        title="MTTR: AI-Automated vs Manual Response",
        xaxis_title="Response Type",
        yaxis_title="Hours",
        showlegend=False
    )
    
    st.plotly_chart(fig_mttr, use_container_width=True)

with col2:
    st.subheader("🚨 Security Event Types")
    
    event_type_counts = security_events['event_type'].value_counts()
    
    fig_events = px.bar(
        x=event_type_counts.values,
        y=event_type_counts.index,
        orientation='h',
        title="Event Type Distribution",
        color=event_type_counts.values,
        color_continuous_scale='Viridis'
    )
    fig_events.update_layout(
        xaxis_title="Number of Events",
        yaxis_title="Event Type",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_events, use_container_width=True)

# Row 3: Advanced Analytics
st.markdown('<h2 class="section-header">🔬 Advanced AI Analytics</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 AI Confidence vs Priority Scoring")
    
    # Scatter plot of confidence vs priority
    fig_scatter = px.scatter(
        security_events,
        x='confidence_score',
        y='priority_score',
        color='priority',
        size='mttr_hours',
        hover_data=['event_type', 'is_anomaly'],
        title="AI Triage Analysis",
        color_discrete_map=priority_colors  # Use the same color mapping
    )
    fig_scatter.update_layout(
        xaxis_title="AI Confidence Score",
        yaxis_title="Priority Score"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("🔍 Anomaly Detection Results")
    
    # Anomaly detection visualization
    anomaly_data = security_events.groupby(['event_type', 'is_anomaly']).size().reset_index(name='count')
    anomaly_data['anomaly_status'] = anomaly_data['is_anomaly'].map({True: 'Anomalous', False: 'Normal'})
    
    fig_anomaly = px.bar(
        anomaly_data,
        x='event_type',
        y='count',
        color='anomaly_status',
        title="Normal vs Anomalous Events by Type",
        color_discrete_map={'Normal': '#1f77b4', 'Anomalous': '#d62728'}
    )
    fig_anomaly.update_layout(
        xaxis_title="Event Type",
        yaxis_title="Number of Events",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)

# Threat Intelligence Simulation
st.markdown('<h2 class="section-header">🌐 Threat Intelligence Correlation</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Threat Intel Sources")
    
    # Simulate threat intelligence sources
    ti_sources = {
        'Commercial Feeds': random.randint(1000, 1500),
        'OSINT': random.randint(800, 1200),
        'Government': random.randint(200, 400),
        'Internal': random.randint(50, 150),
        'Industry Sharing': random.randint(300, 600)
    }
    
    fig_ti = px.treemap(
        names=list(ti_sources.keys()),
        values=list(ti_sources.values()),
        title="Threat Intelligence Sources"
    )
    st.plotly_chart(fig_ti, use_container_width=True)

with col2:
    st.subheader("🔗 Correlation Success Rate")
    
    # Simulate correlation data
    correlation_data = {
        'Source': ['IOCs', 'Domains', 'IPs', 'File Hashes', 'TTPs'],
        'Matches': [random.randint(50, 200) for _ in range(5)],
        'Total': [random.randint(500, 1000) for _ in range(5)]
    }
    correlation_df = pd.DataFrame(correlation_data)
    correlation_df['Success_Rate'] = (correlation_df['Matches'] / correlation_df['Total'] * 100).round(1)
    
    fig_correlation = px.bar(
        correlation_df,
        x='Source',
        y='Success_Rate',
        title="Threat Intel Correlation Success Rates",
        color='Success_Rate',
        color_continuous_scale='RdYlGn',
        text='Success_Rate'
    )
    fig_correlation.update_traces(texttemplate='%{text}%', textposition='outside')
    fig_correlation.update_layout(
        xaxis_title="Intelligence Type",
        yaxis_title="Success Rate (%)",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_correlation, use_container_width=True)

# Automated Response Playbooks
st.markdown('<h2 class="section-header">🤖 Automated Response Playbooks</h2>', unsafe_allow_html=True)

# Playbook execution simulation
playbook_data = {
    'Malware Detection': {'actions': 6, 'avg_time': '45 seconds', 'success_rate': 94},
    'Phishing Email': {'actions': 5, 'avg_time': '30 seconds', 'success_rate': 97},
    'DDoS Attack': {'actions': 4, 'avg_time': '60 seconds', 'success_rate': 91},
    'Data Exfiltration': {'actions': 7, 'avg_time': '90 seconds', 'success_rate': 89},
    'Suspicious Login': {'actions': 3, 'avg_time': '20 seconds', 'success_rate': 96}
}

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("⚙️ Playbook Actions")
    playbook_df = pd.DataFrame(playbook_data).T.reset_index()
    playbook_df.columns = ['Event_Type', 'Actions', 'Avg_Time', 'Success_Rate']
    
    fig_actions = px.bar(
        playbook_df,
        x='Event_Type',
        y='Actions',
        title="Automated Actions per Event Type",
        color='Actions',
        color_continuous_scale='Blues'
    )
    fig_actions.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig_actions, use_container_width=True)

with col2:
    st.subheader("⏱️ Response Speed")
    response_times = [45, 30, 60, 90, 20]  # seconds
    
    fig_speed = px.bar(
        x=list(playbook_data.keys()),
        y=response_times,
        title="Average Response Time (seconds)",
        color=response_times,
        color_continuous_scale='RdYlGn_r'
    )
    fig_speed.update_layout(
        xaxis_title="Event Type",
        xaxis_tickangle=-45,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_speed, use_container_width=True)

with col3:
    st.subheader("✅ Success Rates")
    success_rates = [94, 97, 91, 89, 96]
    
    fig_success = px.bar(
        x=list(playbook_data.keys()),
        y=success_rates,
        title="Playbook Success Rate (%)",
        color=success_rates,
        color_continuous_scale='RdYlGn'
    )
    fig_success.update_layout(
        xaxis_title="Event Type",
        xaxis_tickangle=-45,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_success, use_container_width=True)

# Real-time Event Stream Simulation
st.markdown('<h2 class="section-header">📡 Live Security Event Stream</h2>', unsafe_allow_html=True)

# Show recent events table
st.subheader("🔴 Recent Security Events")

# Filter for high priority events
recent_events = security_events[
    security_events['priority'].isin(['P1-Critical', 'P2-High'])
].nlargest(10, 'timestamp')[
    ['event_id', 'timestamp', 'event_type', 'severity', 'priority', 'confidence_score', 'auto_response_enabled']
]

# Style the dataframe with consistent priority colors
def style_priority(val):
    if val == 'P1-Critical':
        return 'background-color: #ffcdd2'  # Light red
    elif val == 'P2-High':
        return 'background-color: #ffe0b2'  # Light orange
    elif val == 'P3-Medium':
        return 'background-color: #c8e6c8'  # Light green
    elif val == 'P4-Low':
        return 'background-color: #bbdefb'  # Light blue
    return ''

styled_events = recent_events.style.applymap(style_priority, subset=['priority'])
st.dataframe(styled_events, use_container_width=True)

# Cost-Benefit Analysis
st.markdown('<h2 class="section-header">💰 ROI Analysis</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("💵 Cost Comparison")
    
    # Calculate costs
    analyst_cost_per_hour = 75
    auto_cost = auto_resolved_events * security_events[security_events['auto_response_enabled']]['mttr_hours'].mean() * analyst_cost_per_hour
    manual_cost = (total_events - auto_resolved_events) * security_events[~security_events['auto_response_enabled']]['mttr_hours'].mean() * analyst_cost_per_hour
    
    cost_data = pd.DataFrame({
        'Response Type': ['Manual Response', 'AI-Automated Response'],
        'Cost (USD)': [manual_cost, auto_cost],
        'Volume': [total_events - auto_resolved_events, auto_resolved_events]
    })
    
    fig_cost = px.bar(
        cost_data,
        x='Response Type',
        y='Cost (USD)',
        title=f"Monthly Response Costs (${analyst_cost_per_hour}/hour analyst)",
        color='Response Type',
        color_discrete_map={'Manual Response': '#d62728', 'AI-Automated Response': '#2ca02c'}
    )
    
    # Add savings annotation
    savings = manual_cost - auto_cost
    fig_cost.add_annotation(
        x=0.5, y=max(cost_data['Cost (USD)']),
        text=f"Potential Savings: ${savings:,.0f}",
        showarrow=True,
        arrowhead=2,
        bgcolor="yellow",
        bordercolor="black"
    )
    
    st.plotly_chart(fig_cost, use_container_width=True)

with col2:
    st.subheader("📊 Efficiency Metrics")
    
    # Efficiency comparison
    efficiency_data = {
        'Metric': ['Events/Hour', 'Accuracy (%)', 'False Positives (%)', 'Analyst Satisfaction'],
        'Manual': [2.5, 85, 25, 6],
        'AI-Enhanced': [15, 94, 8, 8.5]
    }
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    fig_efficiency = go.Figure()
    fig_efficiency.add_trace(go.Bar(
        name='Manual Process',
        x=efficiency_df['Metric'],
        y=efficiency_df['Manual'],
        marker_color='#d62728'
    ))
    fig_efficiency.add_trace(go.Bar(
        name='AI-Enhanced',
        x=efficiency_df['Metric'],
        y=efficiency_df['AI-Enhanced'],
        marker_color='#2ca02c'
    ))
    
    fig_efficiency.update_layout(
        title="Process Efficiency Comparison",
        xaxis_title="Metrics",
        yaxis_title="Performance",
        barmode='group'
    )
    
    st.plotly_chart(fig_efficiency, use_container_width=True)

# Sidebar - Educational Content
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Learning Objectives")
st.sidebar.markdown("""
- **Explain** how AI automates incident response workflows
- **Analyze** AI-driven threat intelligence correlation  
- **Evaluate** AI's role in attack containment
- **Assess** AI transformation of SOC operations
""")

st.sidebar.markdown("### 🎯 Key Benefits")
st.sidebar.markdown("""
- ⚡ **Speed**: Minutes vs hours response time
- 📈 **Scalability**: Handle 10x more incidents  
- 🎯 **Accuracy**: Reduce human error
- 🔄 **Consistency**: Standardized procedures
- 🛡️ **Proactive**: Predictive threat defense
""")

st.sidebar.markdown("### ⚠️ Implementation Challenges")
st.sidebar.markdown("""
- 📊 Data quality and bias issues
- 🔍 "Black box" explainability problems
- 🤖 Adversarial AI attacks
- 🔧 Integration complexity
- ⚖️ Balancing automation vs human oversight
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Class 8: AI for Incident Response</strong> | 
    AI in Cybersecurity Course | 
    <em>Interactive Dashboard for Educational Purposes</em></p>
    <p>💡 <strong>Key Takeaway:</strong> AI enhances human analysts rather than replacing them, 
    enabling faster, more consistent, and more effective incident response.</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh option
if st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False):
    st.rerun()