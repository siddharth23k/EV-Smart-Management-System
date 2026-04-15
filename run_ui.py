#!/usr/bin/env python3
"""
EV Smart Management System UI Launcher
Launches Streamlit interface
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch UI."""
    print("=" * 50)
    print("EV SMART MANAGEMENT SYSTEM")
    print("=" * 50)
    print("Launching Streamlit interface...")
    print("=" * 50)
    
    # Launch Streamlit
    ui_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(ui_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main()
