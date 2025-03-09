#!/usr/bin/env python3
# server_runner.py

"""
ZerePy Server Runner

This script starts the ZerePy server with the enhanced WebSocket and API functionality.
It handles command-line arguments for configuration and sets up the environment.
"""

import argparse
import os
import sys
import logging
from dotenv import load_dotenv
from pathlib import Path

# Setup logging before imports to catch all log messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('zerepy_server.log')
    ]
)
logger = logging.getLogger("server_runner")

# Try importing enhanced server module
try:
    from enhanced_server import start_server
except ImportError:
    # If not in path, look in src/server
    sys.path.append(str(Path(__file__).parent))
    try:
        from enhanced_server import start_server
    except ImportError as e:
        logger.error(f"Failed to import server module: {e}")
        logger.error("Please ensure enhanced_server.py is in the correct location")
        sys.exit(1)


def main():
    """Parse command-line arguments and start the server."""

    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ZerePy Server with WebSocket and API support')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--mongo-url', help='MongoDB connection URL')
    parser.add_argument('--db-name', help='MongoDB database name')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')

    args = parser.parse_args()

    # Update environment variables if provided as arguments
    if args.mongo_url:
        os.environ['MONGODB_URL'] = args.mongo_url

    if args.db_name:
        os.environ['DATABASE_NAME'] = args.db_name

    # Set log level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")

    # Check for required environment variables
    mongodb_url = os.getenv('MONGODB_URL')
    if not mongodb_url:
        logger.warning("MONGODB_URL not set. Using default: mongodb://localhost:27017")
        os.environ['MONGODB_URL'] = 'mongodb://localhost:27017'

    db_name = os.getenv('DATABASE_NAME')
    if not db_name:
        logger.warning("DATABASE_NAME not set. Using default: zerepy_db")
        os.environ['DATABASE_NAME'] = 'zerepy_db'

    # Start the server
    logger.info(f"Starting ZerePy server on {args.host}:{args.port}")
    try:
        start_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())