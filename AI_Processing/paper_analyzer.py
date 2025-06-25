# Local imports for data structures and AI processing
from Data_Classes.classes import Paper, PaperAnalysis
from langchain_groq import ChatGroq
from utils.token_monitor import TokenMonitor
from pydantic import SecretStr
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
        'Surgery', 'Psychiatry', 'Endocrinology', 'General Medicine'
    }
    
    # System prompt that defines the AI's role and analysis requirements
    SYSTEM_ROLE = """You are an expert medical research analyst with deep knowledge across all medical specialties. 
Your task is to analyze medical research papers and provide accurate categorization and key insights.
Focus on:
- Identifying the primary medical specialty based on the paper's content and methodology
- Extracting the most relevant medical concepts and terminology
- Providing a concise but comprehensive summary that captures the key findings
Be precise and professional in your analysis."""
    
    def __init__(self, api_key: str, token_monitor: Optional[TokenMonitor] = None):
        """
        Initialize the paper analyzer with Groq LLM.
        
        Args:
            api_key (str): API key for Groq LLM service
            token_monitor (Optional[TokenMonitor]): Token monitor for monitoring usage
        """
        self.llm = ChatGroq(api_key=SecretStr(api_key), model="llama3-8b-8192")
        self.token_monitor = token_monitor or TokenMonitor()
    
    def analyze_paper(self, paper: Paper):
        prompt = self._create_analysis_prompt(paper)
        try:
            input_text = self.SYSTEM_ROLE + prompt
            estimated_input_tokens = len(input_text) // 4
            response = self.llm.invoke(
                input=[
                    {"role": "system", "content": self.SYSTEM_ROLE},
                    {"role": "user", "content": prompt}
                ]
            )
            input_tokens = estimated_input_tokens
            output_tokens = len(str(response.content)) // 4
            usage = self.token_monitor.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            result = self._parse_analysis_response(str(response.content))
            return result, usage
        except Exception as e:
            logger.error(f"Error analyzing paper {paper.paper_id}: {str(e)}")
            return None, None
    
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
        Analyze this medical research paper:
        
        Title: {paper.title}
        Abstract: {paper.abstract[:500]}
        Conclusion: {paper.conclusion}
        Authors: {', '.join(paper.authors[:5])}
        arXiv Categories: {', '.join(paper.categories)}
        
        Provide:
        1. Summary of the paper in one paragraph. Write 2-3 sentences about the abstract and 2-3 sentences about the conclusion.
        2. Medical specialty (ONE of: {', '.join(sorted(self.VALID_SPECIALTIES))})
        3. 5 key medical concepts/terms from this research
        
        Format your response as a JSON object with the following structure:
        {{
            "summary": "summary of the paper",
            "specialty": "medical specialty",
            "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
        }}
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
            # Find JSON object in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.error(f"Could not find JSON in response: {response}")
                return None
                
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Validate required fields
            if not all(key in data for key in ['summary', 'specialty', 'keywords']):
                logger.error(f"Missing required fields in response: {data}")
                return None
                
            # Validate specialty
            if data['specialty'] not in self.VALID_SPECIALTIES:
                logger.error(f"Invalid specialty: {data['specialty']}")
                return None
                
            return PaperAnalysis(
                specialty=data['specialty'],
                keywords=data['keywords'][:5],  # Ensure we only take up to 5 keywords
                focus=data['summary']  # Use the summary as the focus
            )
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            return None