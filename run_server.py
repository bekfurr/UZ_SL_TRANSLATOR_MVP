#!/usr/bin/env python
import os
import sys
import subprocess

def main():
    """Run the Django server with Daphne for WebSocket support."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sign_language_project.settings')
    
    try:
        # Check if Daphne is installed
        subprocess.run(['daphne', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Starting server with Daphne for WebSocket support...")
        
        # Run Daphne server
        subprocess.run(['daphne', '-b', '0.0.0.0', '-p', '8000', 'sign_language_project.asgi:application'])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Daphne not found. Please install it with: pip install daphne")
        print("Falling back to standard Django development server (WebSockets won't work)...")
        
        # Import Django's execute_from_command_line
        try:
            from django.core.management import execute_from_command_line
        except ImportError as exc:
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            ) from exc
        
        # Run Django development server
        execute_from_command_line(['manage.py', 'runserver', '127.0.0.1:8000'])

if __name__ == '__main__':
    main()
