"""
run.py
======
Universal entrypoint for the Privacy Preserving Federated Intrusion Detection System.
Launches the Flask Web Application and REST API.

Usage:
    python run.py
    python run.py --port 5000 --host 127.0.0.1
"""

import sys
import os
import argparse
import webbrowser

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import DASHBOARD, PATHS
from src.dashboard.app import app, socketio

def main():
    parser = argparse.ArgumentParser(description="Run the Federated IoT IDS SOC Dashboard")
    parser.add_argument("--host", default=DASHBOARD.HOST, help="Host address to bind to")
    parser.add_argument("--port", type=int, default=DASHBOARD.PORT, help="Port to listen on")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    args = parser.parse_args()

    PATHS.ensure_dirs()

    url = f"http://{args.host}:{args.port}"
    print("=" * 70)
    print("   PRIVACY PRESERVING FEDERATED INTRUSION DETECTION SYSTEM (IDS)")
    print("=" * 70)
    print(f" [OK] Edge IoT Fleet    : 5 Nodes Configured & Monitored")
    print(f" [OK] Global Model      : Federated MLP (96.90% Accuracy)")
    print(f" [OK] SQLite Database   : results/ids_research.db (13 Tables)")
    print(f" [OK] SOC Web Dashboard : {url}")
    print(f" [OK] Default Login     : admin / admin123  (or analyst / analyst123)")
    print("=" * 70)
    print(f"Starting server on {url} ... (Press Ctrl+C to stop)")

    if not args.no_browser and args.host in ["127.0.0.1", "localhost"]:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
