# Local imports for data structures and AI processing
from Data_Classes.classes import Paper, PaperAnalysis
from langchain_groq import ChatGroq
import logging
import re
import json
import hashlib
from typing import Optional
from utils.token_monitor import TokenMonitor, TokenUsage
from .paper_scorer import PaperScorer
from .prompts_loader import PAPER_ANALYSIS_SYSTEM_ROLE, PAPER_ANALYSIS_PROMPT

# Configure logging for this module
logger = logging.getLogger(__name__)


class PaperAnalyzer:
    """
    Analyzes medical research papers using AI to extract key information and categorize content.
    
    This class uses the Groq LLM to analyze medical papers and extract:
    - Primary medical specialty
    - Key medical concepts and terminology
    - Concise summary of findings
    - Deterministic interest score based on content analysis
    """
    
    # Valid medical specialties for categorization
    VALID_SPECIALTIES = [
        "Cardiology", "Oncology", "Neurology", "Psychiatry", "Pediatrics", "Internal Medicine",
        "Surgery", "Emergency Medicine", "Radiology", "Pathology", "Anesthesiology", "Dermatology",
        "Endocrinology", "Gastroenterology", "Hematology", "Infectious Disease", "Nephrology",
        "Ophthalmology", "Orthopedics", "Otolaryngology", "Pulmonology", "Rheumatology",
        "Urology", "Obstetrics and Gynecology", "Family Medicine", "Preventive Medicine",
        "Public Health", "Epidemiology", "Biostatistics", "Medical Genetics", "Immunology",
        "Pharmacology", "Toxicology", "Medical Education", "Health Policy", "Medical Ethics",
        "Rehabilitation Medicine", "Sports Medicine", "Geriatrics", "Palliative Care",
        "Critical Care", "Intensive Care", "Trauma Surgery", "Plastic Surgery", "Neurosurgery",
        "Cardiothoracic Surgery", "Vascular Surgery", "Transplant Surgery", "Medical Imaging",
        "Nuclear Medicine", "Interventional Radiology", "Radiation Oncology", "Medical Oncology",
        "Surgical Oncology", "Gynecologic Oncology", "Pediatric Oncology", "Hematologic Oncology"
    ]
    
    def __init__(self, api_key: str, token_monitor: Optional[TokenMonitor] = None):
        """
        Initialize the paper analyzer with Groq LLM.
        
        Args:
            api_key (str): API key for Groq LLM service
            token_monitor (Optional[TokenMonitor]): Token monitor instance for rate limiting
        """
        # Use temperature=0.0 for deterministic responses
        self.llm = ChatGroq(api_key=api_key, model="llama3-8b-8192", temperature=0.0)
        self.token_monitor = token_monitor or TokenMonitor(max_tokens_per_minute=15500)
        self.scorer = PaperScorer(self.llm)
    
    def analyze_paper(self, paper: Paper) -> tuple[Optional[PaperAnalysis], Optional[TokenUsage]]:
        """
        Analyze a paper using AI to determine its specialty and key concepts.
        
        Args:
            paper (Paper): The paper to analyze
            
        Returns:
            tuple[Optional[PaperAnalysis], Optional[TokenUsage]]: Analysis results and token usage if successful, (None, None) if analysis fails
            
        Note:
            The analysis includes specialty categorization, keyword extraction,
            and a focused summary of the paper's main findings.
        """

        prompt = f"""
Analyze this medical research paper and provide a JSON response with the exact structure shown below.
    
    Title: {paper.title}
    Abstract: {paper.abstract}
    Conclusion: {paper.conclusion}
    Authors: {', '.join(paper.authors)}
    arXiv Categories: {', '.join(paper.categories)}

{PAPER_ANALYSIS_PROMPT}
"""
        
        try:
            input_text = PAPER_ANALYSIS_SYSTEM_ROLE + prompt
            input_tokens = self.token_monitor.count_tokens(input_text)
            response = self.llm.invoke(
                input=[
                    {"role": "system", "content": PAPER_ANALYSIS_SYSTEM_ROLE},
                    {"role": "user", "content": prompt}
                ]
            )
            output_tokens = self.token_monitor.count_tokens(str(response.content))
            usage = self.token_monitor.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            # Parse the basic analysis
            result = self._parse_analysis_response(str(response.content))
            if result:
                # Calculate deterministic interest score using the PaperScorer
                interest_score, score_breakdown = self.scorer.calculate_interest_score(paper, result)
                # Update the analysis with the calculated score
                result = PaperAnalysis(
                    specialty=result.specialty,
                    keywords=result.keywords,
                    focus=result.focus,
                    interest_score=interest_score,
                    score_breakdown=score_breakdown # Add score breakdown to analysis
                )
            
            return result, usage
        except Exception as e:
            logger.error(f"Error analyzing paper {paper.paper_id}: {str(e)}")
            return None, None
    
    def get_high_interest_papers(self, papers_with_analyses: list) -> list:
        """
        Filter papers to get only those with high interest scores (>= 7.0).
        
        Args:
            papers_with_analyses (list): List of papers with their analysis results
            
        Returns:
            list: Papers with high interest scores, sorted by score (highest first)
        """
        return self.scorer.get_high_interest_papers(papers_with_analyses)
    
    def get_papers_by_interest_range(self, papers_with_analyses: list, min_score: float = 0.0, max_score: float = 10.0) -> list:
        """
        Get papers within a specific interest score range.
        
        Args:
            papers_with_analyses (list): List of papers with their analysis results
            min_score (float): Minimum interest score (inclusive)
            max_score (float): Maximum interest score (inclusive)
            
        Returns:
            list: Papers within the specified interest score range, sorted by score
        """
        return self.scorer.get_papers_by_interest_range(papers_with_analyses, min_score, max_score)
    
    def _parse_analysis_response(self, response: str) -> Optional[PaperAnalysis]:
        """
        Parse the AI response into a PaperAnalysis object.
        
        Args:
            response (str): Raw response from the LLM
            
        Returns:
            Optional[PaperAnalysis]: Parsed analysis if successful, None if parsing fails
            
        Note:
            This method includes validation of the response format,
            required fields, and specialty categorization.
        """
        try:
            # Clean the response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Find JSON object in the response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if not json_match:
                logger.error(f"Could not find JSON in response: {response[:200]}...")
                return None
                
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Validate and sanitize required fields
            summary = data.get('summary')
            if not isinstance(summary, str):
                summary = str(summary) if summary is not None else ''
            
            keywords = data.get('keywords')
            if not isinstance(keywords, list):
                keywords = []
            else:
                # Ensure all keywords are strings
                keywords = [str(kw) for kw in keywords if isinstance(kw, (str, int, float))]
            keywords = keywords[:5]
            
            specialty = data.get('specialty')
            if not isinstance(specialty, str):
                logger.error(f"Invalid specialty type: {type(specialty)}")
                return None
                
            # Try to match specialty with valid ones (case-insensitive)
            specialty_lower = specialty.lower()
            matched_specialty = None
            for valid_specialty in self.VALID_SPECIALTIES:
                if valid_specialty.lower() == specialty_lower:
                    matched_specialty = valid_specialty
                    break
            
            if not matched_specialty:
                # Try partial matching for common variations
                for valid_specialty in self.VALID_SPECIALTIES:
                    if any(word in specialty_lower for word in valid_specialty.lower().split()):
                        matched_specialty = valid_specialty
                        break
                
                if not matched_specialty:
                    logger.error(f"Invalid specialty: {specialty}")
                    return None
            
            # Return analysis without interest score (will be calculated separately)
            return PaperAnalysis(
                specialty=matched_specialty,
                keywords=keywords,
                focus=summary,
                interest_score=0.0  # Placeholder, will be calculated
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response content: {response[:300]}...")
            return None
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            logger.error(f"Response content: {response[:300]}...")
            return None