"""
Enhanced EV Smart Management System UI Launcher
Launches the modern Streamlit interface with driver profiling
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the enhanced UI."""
    print("=" * 60)
    print("ENHANCED EV SMART MANAGEMENT SYSTEM")
    print("=" * 60)
    print("Launching modern Streamlit interface with:")
    print("• Driver Profiling Dashboard")
    print("• Interactive Visualizations")
    print("• Performance Analytics")
    print("• Personalized Recommendations")
    print("=" * 60)
    
    # Check if enhanced dependencies are installed
    try:
        import plotly
        import seaborn
        print("Enhanced dependencies available")
    except ImportError:
        print("Installing enhanced dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "plotly>=5.15.0", "seaborn>=0.12.0"
        ])
    
    # Launch Streamlit
    ui_path = Path(__file__).parent / "ui" / "app_enhanced.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(ui_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main()
