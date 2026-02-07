#!/usr/bin/env python3
"""
PaperForge AI - One-Click Launcher
Run this to start everything automatically.
"""
import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

# Configuration
PORT = 5000
URL = f"http://localhost:{PORT}"

def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    required = ['flask', 'openai', 'fitz']  # fitz = PyMuPDF

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[!] Missing packages: {', '.join(missing)}")
        print("[*] Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
        print("[✓] Dependencies installed")
    return True

def check_api_key():
    """Check if OpenAI API key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n" + "="*50)
        print("  ⚠️  OpenAI API Key Required")
        print("="*50)
        print("\nSet your API key:")
        if sys.platform == "win32":
            print("  set OPENAI_API_KEY=your-key-here")
        else:
            print("  export OPENAI_API_KEY='your-key-here'")
        print("\nGet a key at: https://platform.openai.com/api-keys")
        print("="*50 + "\n")

        # Prompt for key
        key = input("Enter API key (or press Enter to skip): ").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            return True
        return False
    return True

def open_browser_delayed():
    """Open browser after short delay."""
    time.sleep(2)
    print(f"\n[*] Opening browser at {URL}")
    webbrowser.open(URL)

def main():
    print("\n" + "="*50)
    print("  🚀 PaperForge AI - Research Paper to MVP")
    print("="*50 + "\n")

    # Change to script directory
    os.chdir(Path(__file__).parent)

    # Check dependencies
    print("[*] Checking dependencies...")
    check_dependencies()
    print("[✓] Dependencies OK")

    # Check API key
    if not check_api_key():
        print("[!] Continuing without API key (uploads will fail)")
    else:
        print("[✓] API key found")

    # Create required directories
    for dir_name in ["uploads", "output", "storage"]:
        Path(dir_name).mkdir(exist_ok=True)

    print(f"\n[*] Starting server at {URL}")
    print("[*] Press Ctrl+C to stop\n")
    print("="*50 + "\n")

    # Open browser in background thread
    import threading
    threading.Thread(target=open_browser_delayed, daemon=True).start()

    # Run Flask app
    from app import app
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[*] Shutting down...")
        # Cleanup running projects
        try:
            from src.runner import runner
            runner.stop_all()
        except:
            pass
        print("[✓] Goodbye!")
