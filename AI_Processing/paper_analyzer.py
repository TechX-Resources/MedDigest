# Local imports for data structures and AI processing
from Data_Classes.classes import Paper, PaperAnalysis
from langchain_groq import ChatGroq
import logging
import re
import json
from typing import Optional

# Configure logging for this module
logger = logging.getLogger(__name__)


class PaperAnalyzer:
    """
    Analyzes medical research papers using AI to extract key information and categorize content.
    
    This class uses the Groq LLM to analyze medical papers and extract:
    - Primary medical specialty
    - Key medical concepts and terminology
    - Concise summary of findings
    """
    
    # Set of valid medical specialties for categorization
    VALID_SPECIALTIES = {
        'Cardiology', 'Oncology', 'Radiology', 'Neurology', 
        'Surgery', 'Psychiatry', 'Endocrinology', 'General Medicine',
        'Dermatology', 'Gastroenterology', 'Pulmonology', 'Orthopedics',
        'Ophthalmology', 'Urology', 'Gynecology', 'Pediatrics',
        'Emergency Medicine', 'Anesthesiology', 'Pathology', 'Immunology',
        'Infectious Disease', 'Nephrology', 'Hematology', 'Rheumatology',
        'Medical Imaging', 'Biomedical Engineering', 'Medical AI', 'Clinical Research'
    }
    
    # System prompt that defines the AI's role and analysis requirements
    SYSTEM_ROLE = """
    You are an expert medical research analyst with deep knowledge across all medical specialties. 
    Your task is to analyze medical research papers and provide accurate categorization and key insights.
    Focus on:
    - Identifying the primary medical specialty based on the paper's content and methodology
    - Extracting the most relevant medical concepts and terminology
    - Providing a concise but comprehensive summary that captures the key findings
    Be precise and professional in your analysis.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the paper analyzer with Groq LLM.
        
        Args:
            api_key (str): API key for Groq LLM service
        """
        self.llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")
    
    def analyze_paper(self, paper: Paper) -> Optional[PaperAnalysis]:
        """
        Analyze a paper using AI to determine its specialty and key concepts.
        
        Args:
            paper (Paper): The paper to analyze
            
        Returns:
            Optional[PaperAnalysis]: Analysis results if successful, None if analysis fails
            
        Note:
            The analysis includes specialty categorization, keyword extraction,
            and a focused summary of the paper's main findings.
        """
        prompt = self._create_analysis_prompt(paper)
        
        try:
            response = self.llm.invoke(
                input=[
                    {"role": "system", "content": self.SYSTEM_ROLE},
                    {"role": "user", "content": prompt}
                ]
            )
            return self._parse_analysis_response(response.content)
        except Exception as e:
            logger.error(f"Error analyzing paper {paper.paper_id}: {str(e)}")
            return None
    
    def _create_analysis_prompt(self, paper: Paper) -> str:
        """
        Create the prompt for paper analysis.
        
        Args:
            paper (Paper): The paper to analyze
            
        Returns:
            str: Formatted prompt for the LLM
            
        Note:
            The prompt includes paper metadata and specific instructions
            for the analysis format and requirements.
        """
        return f"""
        Analyze this medical research paper and provide a JSON response with the exact structure shown below.
        
        Title: {paper.title}
        Abstract: {paper.abstract}
        Conclusion: {paper.conclusion}
        Authors: {', '.join(paper.authors)}
        arXiv Categories: {', '.join(paper.categories)}
        
        Instructions:
        1. Write a 2-3 sentence summary of the paper's key findings
        2. Identify the primary medical specialty from this list: {', '.join(sorted(self.VALID_SPECIALTIES))}
        3. Extract 5 key medical concepts/terms from this research
        
        IMPORTANT: Return ONLY a valid JSON object with this exact structure:
        {{
            "summary": "2-3 sentence summary of the paper's key findings",
            "specialty": "exact specialty name from the provided list",
            "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
        }}
        
        Do not include any text before or after the JSON object. Ensure all quotes are properly escaped.
        """
    
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
            
            return PaperAnalysis(
                specialty=matched_specialty,
                keywords=keywords,
                focus=summary
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response content: {response[:300]}...")
            return None
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            logger.error(f"Response content: {response[:300]}...")
            return None