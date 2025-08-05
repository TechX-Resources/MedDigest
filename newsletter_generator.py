#!/usr/bin/env python3
"""
MedDigest - Newsletter Generator

This module handles the research digest generation and newsletter creation process.
It can be run independently to generate newsletters without sending emails.
"""

import logging
import os
import sys
from typing import Optional
from datetime import datetime

# Third-party imports
from dotenv import load_dotenv

# Local application imports
from AI_Processing.research_digest import ResearchDigest
from Output_Generation.newsletter_markdown import NewsletterMarkdown

# Setup logger at module level
logger = logging.getLogger(__name__)


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


def load_environment() -> Optional[str]:
    """
    Load environment variables and validate API key.
    
    Returns:
        API key if valid, None otherwise
    """
    load_dotenv()
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set. Please set it in your .env file.")
        return None
    
    return api_key


def generate_research_digest(api_key: str) -> Optional[ResearchDigest]:
    """
    Generate research digest using the provided API key.
    
    Args:
        api_key: The GROQ API key for AI processing
        
    Returns:
        ResearchDigest instance if successful, None otherwise
    """
    try:
        digest = ResearchDigest(api_key)
        digest_json = digest.generate_digest()
        digest.digest_json = digest_json
        
        # Print token usage summary
        digest.token_monitor.print_usage_summary()
        
        return digest
    except Exception as e:
        logger.error(f"Failed to generate research digest: {e}")
        return None


def generate_newsletter(digest: ResearchDigest) -> Optional[str]:
    """
    Generate newsletter from the research digest.
    
    Args:
        digest: The ResearchDigest instance
        
    Returns:
        Newsletter content if successful, None otherwise
    """
    try:
        newsletter = NewsletterMarkdown(digest)
        return newsletter.generate_newsletter()
    except Exception as e:
        logger.error(f"Failed to generate newsletter: {e}")
        return None


def generate_newsletter_workflow() -> Optional[str]:
    """
    Complete workflow for generating a newsletter.
    
    Returns:
        Newsletter content if successful, None otherwise
    """
    # Ensure logging is configured
    setup_logging()
    logger.info("📝 Starting newsletter generation workflow...")
    
    # Step 1: Load and validate environment
    api_key = load_environment()
    if not api_key:
        logger.error("❌ Environment setup failed.")
        return None
    
    # Step 2: Generate research digest
    digest = generate_research_digest(api_key)
    if not digest:
        logger.error("❌ Research digest generation failed.")
        return None
    
    # Step 3: Generate newsletter
    newsletter_content = generate_newsletter(digest)
    if not newsletter_content:
        logger.error("❌ Newsletter generation failed.")
        return None
    
    logger.info("✅ Newsletter generated successfully!")
    return newsletter_content


def main() -> None:
    """
    Main entry point for newsletter generation.
    
    This can be run independently to generate newsletters without sending emails.
    """
    global logger
    logger = setup_logging()
    
    logger.info("🚀 Starting MedDigest Newsletter Generator...")
    
    newsletter_content = generate_newsletter_workflow()
    
    if newsletter_content:
        logger.info("🎉 Newsletter generation completed successfully!")
        logger.info("📧 Newsletter content is ready for distribution.")
    else:
        logger.error("❌ Newsletter generation failed.")


if __name__ == "__main__":
    main() 