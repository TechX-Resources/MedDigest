"""
Prompts loader for AI processing of medical research papers.

This module loads prompts from a JSON file and provides the same interface
as the original prompts.py module for backward compatibility.
"""

import json
import os
from typing import Dict, Any


class PromptsLoader:
    """
    Loads and manages prompts from a JSON file.
    
    This class provides the same interface as the original prompts.py module
    for backward compatibility while loading prompts from a structured JSON file.
    """
    
    def __init__(self, prompts_file_path: str = None):
        """
        Initialize the prompts loader.
        
        Args:
            prompts_file_path (str): Path to the prompts JSON file. 
                                   If None, uses default path in same directory.
        """
        if prompts_file_path is None:
            # Use default path in the same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_file_path = os.path.join(current_dir, "prompts.json")
        
        self.prompts_file_path = prompts_file_path
        self._prompts_data = None
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load prompts from the JSON file."""
        try:
            with open(self.prompts_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._prompts_data = data.get("prompts", {})
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading prompts file: {e}")
    
    def get_prompt(self, prompt_name: str) -> str:
        """
        Get a specific prompt by name.
        
        Args:
            prompt_name (str): The name of the specific prompt
            
        Returns:
            str: The prompt text
            
        Raises:
            KeyError: If prompt_name doesn't exist
        """
        if prompt_name not in self._prompts_data:
            raise KeyError(f"Prompt '{prompt_name}' not found in prompts data")
        
        return self._prompts_data[prompt_name]["prompt"]
    
    def get_prompt_metadata(self, prompt_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific prompt.
        
        Args:
            prompt_name (str): The name of the specific prompt
            
        Returns:
            Dict[str, Any]: The prompt metadata
            
        Raises:
            KeyError: If prompt_name doesn't exist
        """
        if prompt_name not in self._prompts_data:
            raise KeyError(f"Prompt '{prompt_name}' not found in prompts data")
        
        return self._prompts_data[prompt_name]
    
    def get_prompt_variables(self, prompt_name: str) -> list:
        """
        Get variables for a specific prompt.
        
        Args:
            prompt_name (str): The name of the specific prompt
            
        Returns:
            list: The prompt variables
            
        Raises:
            KeyError: If prompt_name doesn't exist
        """
        if prompt_name not in self._prompts_data:
            raise KeyError(f"Prompt '{prompt_name}' not found in prompts data")
        
        return self._prompts_data[prompt_name].get("variables", [])
    
    def list_prompts(self) -> list:
        """
        Get a list of all available prompt names.
        
        Returns:
            list: List of prompt names
        """
        return list(self._prompts_data.keys())
    
    def reload_prompts(self) -> None:
        """Reload prompts from the JSON file."""
        self._load_prompts()


# Create a global instance for backward compatibility
_prompts_loader = PromptsLoader()

# Define the same constants as the original prompts.py for backward compatibility
PAPER_ANALYSIS_SYSTEM_ROLE = _prompts_loader.get_prompt("paper_analysis_system_role")
METHODOLOGY_DETECTION_SYSTEM_PROMPT = _prompts_loader.get_prompt("methodology_detection_system_prompt")

PAPER_ANALYSIS_PROMPT = _prompts_loader.get_prompt("paper_analysis_prompt")
CREATE_PAPER_ANALYSIS_PROMPT = _prompts_loader.get_prompt("create_paper_analysis_prompt")

BATCH_ANALYSIS_PROMPT = _prompts_loader.get_prompt("batch_analysis_prompt")

EXECUTIVE_SUMMARY_PROMPT = _prompts_loader.get_prompt("executive_summary_prompt")
KEY_DISCOVERIES_PROMPT = _prompts_loader.get_prompt("key_discoveries_prompt")
EMERGING_TRENDS_PROMPT = _prompts_loader.get_prompt("emerging_trends_prompt")
MEDICAL_IMPACT_PROMPT = _prompts_loader.get_prompt("medical_impact_prompt")
CROSS_SPECIALTY_INSIGHTS_PROMPT = _prompts_loader.get_prompt("cross_specialty_insights_prompt")
CLINICAL_IMPLICATIONS_PROMPT = _prompts_loader.get_prompt("clinical_implications_prompt")
RESEARCH_GAPS_PROMPT = _prompts_loader.get_prompt("research_gaps_prompt")
FUTURE_DIRECTIONS_PROMPT = _prompts_loader.get_prompt("future_directions_prompt") 