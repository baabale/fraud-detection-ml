"""
WSGI entry point for the Fraud Detection API.
This file is used by Gunicorn to serve the Flask application.
"""

from app import app

if __name__ == "__main__":
    app.run()
