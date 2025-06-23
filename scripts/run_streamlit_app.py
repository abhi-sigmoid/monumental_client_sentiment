#!/usr/bin/env python3
"""
Launcher script for the Streamlit database viewer application.
"""

import subprocess
import sys
import os


def main():
    """Launch the Streamlit application."""
    # Get the path to the Streamlit app
    app_path = os.path.join("src", "web", "database_viewer.py")

    if not os.path.exists(app_path):
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)

    # Launch Streamlit
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_path,
                "--server.port",
                "8501",
                "--server.address",
                "localhost",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
