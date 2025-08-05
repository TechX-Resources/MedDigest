# Standard library imports for HTTP requests and XML parsing
import time
import requests
import xml.etree.ElementTree as ET
from typing import List
import datetime
from urllib.parse import urlencode
import logging
from dataclasses import dataclass
from Data_Classes.classes import Paper

# Configure logging for this module
logger = logging.getLogger(__name__)


class ArxivClient:
    """
    Client for interacting with the arXiv API.
    
    This class handles the communication with arXiv's API, including:
    - Paper search and retrieval
    - Response parsing
    - Error handling and retries
    """
    
    # Base URL for arXiv API
    BASE_URL = "https://export.arxiv.org/api/query"
    
    def __init__(self):
        """
        Initialize the arXiv client with appropriate headers and session.
        Sets up a persistent session for better performance.
        """
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/xml,application/xhtml+xml,text/html;q=0.9',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def fetch_papers(self, search_query: str, max_results: int = 20) -> List[Paper]:
        """
        Fetch papers from arXiv API with retry logic and exponential backoff.
        
        Args:
            search_query (str): The search query to use for finding papers
            max_results (int): Maximum number of results to return (default: 20)
            
        Returns:
            List[Paper]: List of Paper objects containing the search results
            
        Raises:
            requests.exceptions.RequestException: If all retry attempts fail
        """
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }
        
        url = f"{self.BASE_URL}?{urlencode(params)}"
        
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to fetch papers (attempt {attempt + 1}/{max_retries})...")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return self._parse_response(response.content)
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Request failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch papers after {max_retries} attempts: {str(e)}")
                    raise
        
        return []  # This line should never be reached due to the raise in the loop
    
    def _parse_response(self, content: bytes) -> List[Paper]:
        """
        Parse the XML response from arXiv API into Paper objects.
        
        Args:
            content (bytes): Raw XML response from arXiv API
            
        Returns:
            List[Paper]: List of parsed Paper objects
        """
        root = ET.fromstring(content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        
        for entry in root.findall('atom:entry', ns):
            paper = Paper(
                paper_id=entry.find('atom:id', ns).text.split('/')[-1],
                title=entry.find('atom:title', ns).text.strip(),
                published=datetime.datetime.strptime(
                    entry.find('atom:published', ns).text, 
                    '%Y-%m-%dT%H:%M:%SZ'
                ).replace(tzinfo=datetime.timezone.utc),
                abstract=entry.find('atom:summary', ns).text.strip(),
                authors=[author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                categories=[cat.get('term') for cat in entry.findall('atom:category', ns)],
                conclusion=entry.find('atom:summary', ns).text.strip()
            )
            papers.append(paper)
        
        return papers