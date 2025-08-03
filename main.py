#!/usr/bin/env python3
"""
MedDigest - Main Entry Point

This module serves as the main entry point for the MedDigest application.
It orchestrates the research digest generation and newsletter distribution process.
"""

import logging
import sys

# Local application imports
from newsletter_generator import generate_newsletter_workflow
from newsletter_sender import send_newsletter_workflow


def setup_logging() -> logging.Logger:
    """Configure and return a logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def main() -> None:
    """
    Main entry point for the MedDigest application.
    
    Orchestrates the complete workflow:
    1. Newsletter generation (research digest + newsletter creation)
    2. Newsletter distribution (email sending)
    """
    global logger
    logger = setup_logging()
    
    logger.info("🚀 Starting MedDigest application...")
    
    # Step 1: Generate newsletter
    newsletter_content = generate_newsletter_workflow()
    if not newsletter_content:
        logger.error("❌ Newsletter generation failed. Exiting.")
        return
    
    # Step 2: Send newsletter email
    send_newsletter_workflow(newsletter_content)
    
    logger.info("🎉 MedDigest application completed successfully!")


if __name__ == "__main__":
    main()
