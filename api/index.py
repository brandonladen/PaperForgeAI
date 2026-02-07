import sys
import os

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app  # assuming app.py defines `app = Flask(__name__)`

# Vercel will automatically use `app`
