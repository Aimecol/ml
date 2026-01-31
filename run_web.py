#!/usr/bin/env python
"""Run the ML Project Framework web interface.

This script starts a Flask development server for the web interface.

Usage:
    python run_web.py                  # Run with default settings
    python run_web.py --port 8000      # Run on custom port
    python run_web.py --host 0.0.0.0   # Run on all interfaces
    python run_web.py --no-debug       # Run in production mode

"""

import sys
import os
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from web import create_app
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for the web interface."""
    parser = argparse.ArgumentParser(
        description='ML Project Framework Web Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_web.py                      # Default: localhost:5000
  python run_web.py --port 8000          # Custom port
  python run_web.py --host 0.0.0.0       # All interfaces
  python run_web.py --no-debug           # Production mode
        """
    )

    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=True,
        help='Run in debug mode (default: True)'
    )

    parser.add_argument(
        '--no-debug',
        dest='debug',
        action='store_false',
        help='Run in production mode'
    )

    parser.add_argument(
        '--reload',
        action='store_true',
        default=True,
        help='Enable auto-reload on code changes (default: True)'
    )

    parser.add_argument(
        '--no-reload',
        dest='reload',
        action='store_false',
        help='Disable auto-reload'
    )

    args = parser.parse_args()

    # Create Flask app
    logger.info("Creating Flask application...")
    app = create_app()

    # Display startup info
    print("\n" + "="*70)
    print("  ML Project Framework Web Interface")
    print("="*70)
    print(f"\n  Starting server at http://{args.host}:{args.port}")
    print(f"  Debug mode: {args.debug}")
    print(f"  Auto-reload: {args.reload}")
    print("\n  Features:")
    print("    - Dashboard with experiment tracking")
    print("    - Interactive pipeline configuration")
    print("    - Real-time data preview")
    print("    - Model training with custom parameters")
    print("    - Results visualization")
    print("\n  Navigation:")
    print("    - Dashboard: http://{0}:{1}/".format(args.host, args.port))
    print("    - Pipeline: http://{0}:{1}/pipeline".format(args.host, args.port))
    print("    - Results: http://{0}:{1}/results".format(args.host, args.port))
    print("    - Docs: http://{0}:{1}/documentation".format(args.host, args.port))
    print("\n  API Endpoints:")
    print("    - GET  /api/status - System status")
    print("    - GET  /api/config - Configuration")
    print("    - GET  /api/algorithms - Available algorithms")
    print("    - POST /api/upload - Upload dataset")
    print("    - POST /api/preview-data - Preview data")
    print("    - POST /api/train-model - Train model")
    print("    - GET  /api/experiments - List experiments")
    print("    - GET  /api/experiment/<file> - Get experiment details")
    print("\n  Press CTRL+C to stop the server")
    print("="*70 + "\n")

    # Run the app
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=args.reload
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        logger.info("Web interface stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
