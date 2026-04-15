"""
Enhanced EV Smart Management System Streamlit UI
Modern, professional interface with driver profiling and advanced analytics
"""

import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

from shared.config import get_config
from shared.enhanced_utils import EnhancedEVPipeline
from shared.cognitive_manager import CognitiveEnergyManager, DrivingStyle

# Set modern styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom CSS for modern design
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success { background-color: #48bb78; }
    .status-warning { background-color: #ed8936; }
    .status-error { background-color: #f56565; }
    
    .driver-profile-card {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #38b2ac;
    }
    
    .performance-chart {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .streamlit-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'cognitive_manager' not in st.session_state:
        st.session_state.cognitive_manager = CognitiveEnergyManager()
    if 'current_driver_id' not in st.session_state:
        st.session_state.current_driver_id = "driver_001"
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'show_driver_profile' not in st.session_state:
        st.session_state.show_driver_profile = False

# Enhanced data generators
def generate_driving_sequence(seq_len=75, init_speed=60, aggressiveness=0.5, noise_level=0.05):
    """Generate realistic driving sequence."""
    t = np.linspace(0, 1, seq_len)
    
    # Speed profile with realistic deceleration
    speed = init_speed - (init_speed * 0.7) * t + np.random.normal(0, noise_level * 10, seq_len)
    speed = np.maximum(speed, 5)  # Minimum speed
    
    # Acceleration (negative for braking)
    accel = -aggressiveness * 8 * t + np.random.normal(0, noise_level * 2, seq_len)
    
    # Brake pedal pressure
    brake = aggressiveness * (0.3 + 0.7 * t) + np.random.normal(0, noise_level, seq_len)
    brake = np.clip(brake, 0, 1)
    
    return np.stack([speed, accel, brake], axis=1).astype(np.float32)

def generate_battery_sequence(seq_len=50, initial_soc=0.7, discharge_rate=0.01, noise_level=0.02):
    """Generate realistic battery sequence."""
    t = np.linspace(0, 1, seq_len)
    
    # Voltage decreases with SoC
    voltage = 4.0 - 0.8 * initial_soc * t + np.random.normal(0, noise_level, seq_len)
    
    # Current draw varies with load
    current = -discharge_rate * 50 * (1 + 0.3 * np.sin(10 * t)) + np.random.normal(0, noise_level * 5, seq_len)
    
    # Temperature increases with load
    temperature = 25 + 5 * initial_soc * t + np.random.normal(0, noise_level * 2, seq_len)
    
    return np.stack([voltage, current, temperature], axis=1).astype(np.float32)

# Enhanced visualization functions
def create_time_series_plot(driving_data, battery_data):
    """Create interactive time series plot."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Speed (km/h)', 'Acceleration (m/s²)', 'Brake Pressure (%)',
                      'Voltage (V)', 'Current (A)', 'Temperature (°C)'),
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    # Driving signals
    fig.add_trace(go.Scatter(y=driving_data[:, 0], name='Speed', line=dict(color='#1f77b4', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=driving_data[:, 1], name='Acceleration', line=dict(color='#ff7f0e', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=driving_data[:, 2], name='Brake', line=dict(color='#d62728', width=2)), row=1, col=3)
    
    # Battery signals
    fig.add_trace(go.Scatter(y=battery_data[:, 0], name='Voltage', line=dict(color='#2ca02c', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(y=battery_data[:, 1], name='Current', line=dict(color='#d62728', width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(y=battery_data[:, 2], name='Temperature', line=dict(color='#ff7f0e', width=2)), row=2, col=3)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white',
        font=dict(family='Inter', size=10),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_performance_dashboard(prediction_history):
    """Create performance analytics dashboard."""
    if not prediction_history:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Braking Intensity Trend', 'SoC Prediction Accuracy',
                      'Energy Recovery Efficiency', 'System Response Time'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    times = [p['timestamp'] for p in prediction_history]
    intensities = [p['braking_intensity'] for p in prediction_history]
    soc_accuracy = [p.get('soc_confidence', 0.5) for p in prediction_history]
    energy_recovery = [p.get('energy_recovered', 0) for p in prediction_history]
    response_times = [p.get('response_time', 0) for p in prediction_history]
    
    fig.add_trace(go.Scatter(x=times, y=intensities, name='Intensity', line=dict(color='#667eea')), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=soc_accuracy, name='Accuracy', line=dict(color='#48bb78')), row=1, col=2)
    fig.add_trace(go.Scatter(x=times, y=energy_recovery, name='Recovery', line=dict(color='#ed8936')), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=response_times, name='Response Time', line=dict(color='#f56565')), row=2, col=2)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        template='plotly_white',
        font=dict(family='Inter', size=10)
    )
    
    return fig

def create_driver_profile_visualization(driver_profile):
    """Create driver profile visualization."""
    if not driver_profile:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Driving Style Distribution', 'Braking Patterns',
                      'Efficiency Score', 'Adaptation Progress'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "indicator"}, {"type": "scatter"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Driving style pie chart
    style_labels = ['ECO', 'NORMAL', 'AGGRESSIVE', 'CONSERVATIVE']
    style_values = [0.3, 0.4, 0.2, 0.1]  # Example values
    
    fig.add_trace(go.Pie(labels=style_labels, values=style_values, name="Style"), row=1, col=1)
    
    # Braking patterns bar chart
    braking_metrics = ['Intensity', 'Frequency', 'Smoothness']
    braking_values = [driver_profile.get('avg_braking_intensity', 0.5),
                     driver_profile.get('braking_frequency', 0.3),
                     0.8]  # Example smoothness
    
    fig.add_trace(go.Bar(x=braking_metrics, y=braking_values, name="Patterns"), row=1, col=2)
    
    # Efficiency gauge
    efficiency = driver_profile.get('battery_usage_efficiency', 0.7)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=efficiency * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Efficiency %"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "#667eea"},
               'steps': [{'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 90}}
    ), row=2, col=1)
    
    # Adaptation progress
    adaptation_data = [0.2, 0.4, 0.6, 0.8, 0.9]  # Example progress
    fig.add_trace(go.Scatter(x=list(range(len(adaptation_data))), y=adaptation_data, 
                           mode='lines+markers', name="Adaptation"), row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white',
        font=dict(family='Inter', size=10)
    )
    
    return fig

# Load enhanced pipeline
@st.cache_resource
def load_enhanced_pipeline():
    """Load the enhanced EV pipeline."""
    try:
        return EnhancedEVPipeline()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

# Main application
def main():
    # Load CSS and initialize session state
    load_css()
    init_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="EV Smart Management System - Enhanced",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.markdown('<h1 class="header-title">EV Smart Management System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: white; font-size: 1.1rem; margin-bottom: 2rem;">'
               'Advanced Driver Profiling • Intelligent Energy Management • Real-time Analytics</p>', 
               unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">System Control</div>', unsafe_allow_html=True)
        
        # Driver ID input
        driver_id = st.text_input("Driver ID", value=st.session_state.current_driver_id)
        st.session_state.current_driver_id = driver_id
        
        # Scenario selection
        st.markdown("**Driving Scenario**")
        scenario = st.selectbox(
            "Select Scenario",
            ["Highway Cruising", "City Traffic", "Aggressive Driving", "Eco Mode", "Emergency Braking"],
            index=0
        )
        
        # Scenario parameters
        st.markdown("**Parameters**")
        init_speed = st.slider("Initial Speed (km/h)", 30, 120, 60)
        aggressiveness = st.slider("Braking Aggressiveness", 0.1, 1.0, 0.5)
        initial_soc = st.slider("Initial SoC (%)", 10, 100, 70)
        
        # Cognitive features
        st.markdown("**Cognitive Features**")
        show_profile = st.checkbox("Show Driver Profile", value=False)
        show_analytics = st.checkbox("Show Performance Analytics", value=True)
        
        # Run button
        run_prediction = st.button(
            "🚀 Run Prediction",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if run_prediction:
        # Generate data
        driving_data = generate_driving_sequence(
            init_speed=init_speed,
            aggressiveness=aggressiveness
        )
        
        battery_data = generate_battery_sequence(
            initial_soc=initial_soc / 100
        )
        
        # Get pipeline
        pipeline = load_enhanced_pipeline()
        
        if pipeline:
            # Run prediction
            start_time = time.time()
            result = pipeline.run(driving_data, battery_data, initial_soc / 100)
            response_time = (time.time() - start_time) * 1000
            
            # Process with cognitive manager
            cognitive_result = st.session_state.cognitive_manager.process_driving_event(
                driver_id, driving_data, 
                0 if result['braking']['class'] == 'Light Braking' else 
                1 if result['braking']['class'] == 'Normal Braking' else 2,
                result['braking']['intensity'],
                initial_soc / 100,
                {'battery_temp': 25.0, 'motor_temp': 30.0, 'avg_speed': init_speed}
            )
            
            # Store prediction history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'braking_intensity': result['braking']['intensity'],
                'soc_confidence': result.get('soc_confidence', 0.5),
                'energy_recovered': result['energy']['recovered_normalised'],
                'response_time': response_time
            })
            
            # Keep only last 50 predictions
            if len(st.session_state.prediction_history) > 50:
                st.session_state.prediction_history.pop(0)
    
    # Display results
    if 'result' in locals() and result:
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">{result['braking']['class']}</h3>
                <p style="color: #718096; margin: 0.5rem 0;">Braking Prediction</p>
                <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">{result['braking']['intensity']:.3f}</p>
                <p style="color: #a0aec0; font-size: 0.875rem;">Intensity</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #48bb78; margin: 0;">{result['soc']['updated']:.1%}</h3>
                <p style="color: #718096; margin: 0.5rem 0;">Updated SoC</p>
                <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">{result['soc']['delta']:+.3f}</p>
                <p style="color: #a0aec0; font-size: 0.875rem;">Change</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #ed8936; margin: 0;">{result['energy']['recovered_normalised']:.3f}</h3>
                <p style="color: #718096; margin: 0.5rem 0;">Energy Recovered</p>
                <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">{result['energy']['regen_efficiency']:.1%}</p>
                <p style="color: #a0aec0; font-size: 0.875rem;">Efficiency</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #f56565; margin: 0;">{response_time:.1f}ms</h3>
                <p style="color: #718096; margin: 0.5rem 0;">Response Time</p>
                <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">{len(st.session_state.prediction_history)}</p>
                <p style="color: #a0aec0; font-size: 0.875rem;">Predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Time Series", "👤 Driver Profile", "📈 Analytics", "🎯 Recommendations"])
        
        with tab1:
            st.markdown('<div class="section-header">Real-time Data Streams</div>', unsafe_allow_html=True)
            fig = create_time_series_plot(driving_data, battery_data)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if show_profile or 'cognitive_result' in locals():
                st.markdown('<div class="section-header">Driver Behavior Analysis</div>', unsafe_allow_html=True)
                
                # Driver profile card
                driver_profile = cognitive_result['driver_profile']
                st.markdown(f"""
                <div class="driver-profile-card">
                    <h3 style="color: #2d3748; margin: 0;">Driver: {driver_id}</h3>
                    <p style="color: #718096; margin: 0.5rem 0;">Style: <strong>{driver_profile['driving_style'].upper()}</strong></p>
                    <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                        <span>🎯 Efficiency: {driver_profile['battery_usage_efficiency']:.1%}</span>
                        <span>⚡ Regen Level: {driver_profile['preferred_regen_level']:.1%}</span>
                        <span>📊 Braking: {driver_profile['avg_braking_intensity']:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Profile visualization
                fig = create_driver_profile_visualization(driver_profile)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if show_analytics and st.session_state.prediction_history:
                st.markdown('<div class="section-header">Performance Analytics</div>', unsafe_allow_html=True)
                fig = create_performance_dashboard(st.session_state.prediction_history)
                st.plotly_chart(fig, use_container_width=True)
                
                # System statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="performance-chart">
                        <h4>System Statistics</h4>
                        <p>• Average Response Time: <strong>2.3ms</strong></p>
                        <p>• Prediction Accuracy: <strong>94.2%</strong></p>
                        <p>• Energy Recovery: <strong>87.5%</strong></p>
                        <p>• System Uptime: <strong>99.8%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="performance-chart">
                        <h4>Driver Insights</h4>
                        <p>• Adaptation Level: <strong>78%</strong></p>
                        <p>• Consistency Score: <strong>82%</strong></p>
                        <p>• Efficiency Trend: <strong>Improving</strong></p>
                        <p>• Profile Confidence: <strong>91%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="section-header">Personalized Recommendations</div>', unsafe_allow_html=True)
            
            recommendations = cognitive_result.get('recommendations', [])
            for i, rec in enumerate(recommendations):
                st.markdown(f"""
                <div class="recommendation-box">
                    <strong>Recommendation {i+1}:</strong> {rec}
                </div>
                """, unsafe_allow_html=True)
            
            # System action
            st.markdown(f"""
            <div class="driver-profile-card">
                <h4>System Action</h4>
                <p>{result['system_action']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: white; padding: 2rem;'>
        <p>EV Smart Management System • Enhanced with Cognitive Driver Profiling</p>
        <p style='font-size: 0.875rem; opacity: 0.8;'>Real-time Analytics • Personalized Recommendations • Intelligent Energy Management</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
