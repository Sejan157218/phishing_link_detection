#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import dotenv
import pathlib
import signal

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'phishing_link_detection.settings')

    DOT_ENV_PATH = pathlib.Path() / '.env' / 'local.env'
    if DOT_ENV_PATH.exists():
        dotenv.load_dotenv(str(DOT_ENV_PATH))
    else:
        print("## environment variable path not found ##")

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
    # Set up signal handling
    # signal.signal(signal.SIGTERM, handle_sigterm)
    # signal.signal(signal.SIGINT, handle_sigterm)

if __name__ == '__main__':
    main()
