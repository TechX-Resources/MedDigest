# Standard library imports for data structures and type hints
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import datetime


@dataclass
class Paper:
    """
    Data class representing a research paper with its metadata and content.
    
    Attributes:
        paper_id (str): Unique identifier for the paper
        title (str): Title of the research paper
        published (datetime): Publication date and time
        abstract (str): Abstract/summary of the paper
        authors (List[str]): List of author names
        categories (List[str]): List of paper categories/topics
        conclusion (str): Main conclusions or findings of the paper
    """
    paper_id: str
    title: str
    published: datetime.datetime
    abstract: str
    authors: List[str]
    categories: List[str]
    conclusion: str


@dataclass
class PaperAnalysis:
    """
    Data class representing the AI-generated analysis of a research paper.
    
    Attributes:
        specialty (str): The main field or specialty of the paper
        keywords (List[str]): Key terms and concepts from the paper
        focus (str): The primary focus or main contribution of the paper
        interest_score (float): AI-generated interest rating from 0.0 to 10.0
        score_breakdown (Optional[Dict[str, Any]]): Detailed breakdown of how the interest score was calculated
    """
    specialty: str
    keywords: List[str]
    focus: str
    interest_score: float
    score_breakdown: Optional[Dict[str, Any]] = None