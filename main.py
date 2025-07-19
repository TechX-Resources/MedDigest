# Standard library imports for core functionality
import datetime
import logging
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from urllib.parse import urlencode

# Third-party imports for AI and environment management
from langchain_groq import ChatGroq
import time
from dotenv import load_dotenv
import os
import json
import re

# Local application imports
from Data_Classes.classes import Paper, PaperAnalysis
from Data_Retrieval.data_retrieval import ArxivClient
from AI_Processing.paper_analyzer import PaperAnalyzer
from AI_Processing.research_digest import ResearchDigest

# Load environment variables from .env file
load_dotenv()

# Configure logging with timestamp, level, and message format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the script.
    
    Initializes the ResearchDigest with API credentials and generates
    a research digest of academic papers.
    
    Returns:
        None
    """
    # Retrieve and validate API key from environment variables
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set. Please set it in your .env file.")
        return

    # Initialize and run the research digest generation
    digest = ResearchDigest(api_key)
    digest_json = digest.generate_digest()
    digest.digest_json = digest_json  # Store the digest JSON for Newsletter

    # Print token usage summary
    digest.token_monitor.print_usage_summary()

    # Generate the newsletter
    from Output_Generation.newsletter import Newsletter
    newsletter = Newsletter(digest)
    newsletter.generate_newsletter()

# Entry point guard to ensure script runs only when executed directly
if __name__ == "__main__":
    main()
