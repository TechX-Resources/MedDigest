# Local imports for data structures and AI processing
from Data_Classes.classes import Paper, PaperAnalysis
from langchain_groq import ChatGroq
import logging
import json
from typing import List, Dict, Tuple
from .prompts_loader import METHODOLOGY_DETECTION_SYSTEM_PROMPT, CREATE_PAPER_ANALYSIS_PROMPT

# Configure logging for this module
logger = logging.getLogger(__name__)


class PaperScorer:
    """
    Handles scoring and ranking of medical research papers based on multiple factors.
    
    This class provides functionality to:
    - Detect methodologies in paper text using AI
    - Calculate deterministic interest scores based on multiple criteria
    - Filter and rank papers by interest scores
    - Provide detailed scoring breakdowns
    """
    
    # High Impact Methodologies/Techniques (Evidence Level 1-2)
    HIGH_IMPACT_METHODOLOGIES = [
        # Clinical Trial Methodologies
        "Randomized Controlled Trial (RCT)",
        "Double-blind Randomized Controlled Trial",
        "Triple-blind Randomized Controlled Trial",
        "Cluster Randomized Controlled Trial",
        "Crossover Randomized Controlled Trial",
        "Adaptive Randomized Trial",
        "Pragmatic Clinical Trial",
        "Phase III Clinical Trial",
        "Phase IV Post-marketing Surveillance",
        "Multi-center Clinical Trial",
        "International Multi-center Trial",
        
        # Meta-analysis and Systematic Reviews
        "Systematic Review with Meta-analysis",
        "Network Meta-analysis",
        "Individual Patient Data Meta-analysis",
        "Cochrane Systematic Review",
        "Living Systematic Review",
        "Umbrella Review",
        "Scoping Review with Meta-analysis",
        
        # Advanced Statistical Methods
        "Bayesian Randomized Controlled Trial",
        "Mendelian Randomization",
        "Propensity Score Matching",
        "Instrumental Variable Analysis",
        "Regression Discontinuity Design",
        "Difference-in-Differences Analysis",
        "Interrupted Time Series Analysis",
        "Causal Inference Methods",
        
        # Genomics and Precision Medicine
        "Genome-Wide Association Study (GWAS)",
        "Whole Genome Sequencing",
        "Whole Exome Sequencing",
        "Pharmacogenomics Study",
        "Polygenic Risk Score Analysis",
        "Multi-omics Integration",
        "Single-cell RNA Sequencing",
        "Spatial Transcriptomics",
        
        # Advanced AI/ML in Medicine
        "Deep Learning for Medical Imaging",
        "Federated Learning in Healthcare",
        "Explainable AI in Clinical Decision Making",
        "Large Language Models for Medical Tasks",
        "Computer Vision for Pathology",
        "Reinforcement Learning for Treatment Optimization",
        "Transfer Learning in Medical AI",
        "Multi-modal AI for Healthcare",
        
        # Regulatory and Implementation
        "FDA Breakthrough Therapy Designation",
        "Real-World Evidence (RWE) Study",
        "Comparative Effectiveness Research",
        "Health Technology Assessment",
        "Implementation Science Study",
        "Pragmatic-Explanatory Continuum Indicator Summary (PRECIS-2)",
    ]

    # Medium Impact Methodologies/Techniques (Evidence Level 3-4)
    MEDIUM_IMPACT_METHODOLOGIES = [
        # Observational Studies
        "Prospective Cohort Study",
        "Retrospective Cohort Study",
        "Case-Control Study",
        "Nested Case-Control Study",
        "Cross-sectional Study",
        "Longitudinal Study",
        "Population-based Study",
        "Registry-based Study",
        "Electronic Health Record (EHR) Study",
        
        # Phase I/II Trials
        "Phase I Clinical Trial",
        "Phase II Clinical Trial",
        "Dose-escalation Study",
        "Pilot Clinical Trial",
        "Feasibility Study",
        "Proof-of-Concept Study",
        "First-in-Human Study",
        
        # Diagnostic Studies
        "Diagnostic Accuracy Study",
        "Biomarker Validation Study",
        "Screening Study",
        "Predictive Model Development",
        "Prognostic Model Validation",
        "Risk Stratification Study",
        
        # Experimental Methods
        "In Vivo Animal Study",
        "Preclinical Study",
        "Translational Research",
        "Mechanistic Study",
        "Pharmacokinetic/Pharmacodynamic Study",
        "Toxicology Study",
        
        # Survey and Qualitative Methods
        "Cross-sectional Survey",
        "Longitudinal Survey",
        "Mixed Methods Study",
        "Qualitative Study",
        "Ethnographic Study",
        "Phenomenological Study",
        "Grounded Theory Study",
        
        # Intermediate AI/ML Methods
        "Machine Learning for Risk Prediction",
        "Natural Language Processing for Clinical Notes",
        "Computer-Aided Diagnosis",
        "Radiomics Analysis",
        "Predictive Modeling",
        "Time Series Analysis for Healthcare",
        "Survival Analysis with ML",
        
        # Specialized Techniques
        "Proteomics Study",
        "Metabolomics Study",
        "Microbiome Analysis",
        "Epigenetic Study",
        "Immunophenotyping",
        "Flow Cytometry Analysis",
        "Mass Spectrometry Analysis",
        "Pharmacovigilance Study",
        
        # Health Services Research
        "Health Economic Evaluation",
        "Quality of Life Assessment",
        "Patient-Reported Outcome Measures (PROMs)",
        "Cost-effectiveness Analysis",
        "Budget Impact Analysis",
        "Markov Modeling",
    ]

    # Low Impact Methodologies/Techniques (Evidence Level 5 and below)
    LOW_IMPACT_METHODOLOGIES = [
        # Case Studies and Reports
        "Case Report",
        "Case Series",
        "Single Case Study",
        "Multiple Case Report",
        "Case-based Review",
        
        # Basic Laboratory Methods
        "In Vitro Study",
        "Cell Culture Study",
        "Tissue Culture Study",
        "Biochemical Assay",
        "Enzyme Activity Assay",
        "Protein Expression Analysis",
        "Western Blot Analysis",
        "PCR Analysis",
        "qPCR Analysis",
        "ELISA",
        "Immunohistochemistry",
        "Histological Analysis",
        
        # Basic Animal Models
        "Mouse Model Study",
        "Rat Model Study",
        "Cell Line Study",
        "Xenograft Model",
        "Organoid Study",
        
        # Descriptive Studies
        "Descriptive Study",
        "Ecological Study",
        "Correlation Study",
        "Prevalence Study",
        "Incidence Study",
        "Mortality Study",
        
        # Reviews and Commentaries
        "Literature Review",
        "Narrative Review",
        "Editorial",
        "Commentary",
        "Opinion Piece",
        "Perspective Article",
        "Letter to Editor",
        "Short Communication",
        
        # Basic Statistical Methods
        "Descriptive Statistics",
        "Correlation Analysis",
        "Simple Linear Regression",
        "Chi-square Test",
        "T-test Analysis",
        "ANOVA",
        "Basic Survival Analysis",
        
        # Theoretical and Conceptual
        "Theoretical Model",
        "Conceptual Framework",
        "Hypothesis Generation",
        "Mathematical Model",
        "Simulation Study",
        "Modeling Study",
        
        # Basic Surveys
        "Cross-sectional Survey (Small Sample)",
        "Convenience Sample Survey",
        "Online Survey",
        "Questionnaire Study",
        "Interview Study",
        "Focus Group Study",
        
        # Basic Imaging
        "Imaging Study (Descriptive)",
        "Radiological Case Series",
        "Ultrasound Study",
        "CT Scan Analysis",
        "MRI Analysis",
        "X-ray Analysis",
        
        # Preliminary Work
        "Pilot Study (Small N)",
        "Preliminary Results",
        "Feasibility Assessment",
        "Method Development",
        "Protocol Development",
        "Validation Study (Small Scale)",
    ]
    
    def __init__(self, llm: ChatGroq):
        """
        Initialize the paper scorer with the LLM instance.
        
        Args:
            llm (ChatGroq): The LLM instance to use for methodology detection
        """
        self.llm = llm
    
    def detect_methodologies(self, paper_text: str, methodology_list: List[str]) -> List[Dict]:
        """
        Detect which methodologies from a given list are present in the paper text.
        
        Args:
            paper_text (str): The text of the paper to analyze
            methodology_list (List[str]): List of methodologies to check for
            
        Returns:
            List[Dict]: List of dictionaries with methodology name and presence indicator
        """
        try:
            prompt = CREATE_PAPER_ANALYSIS_PROMPT.format(
                methodology_list=methodology_list,
                paper_text=paper_text
            )

            response = self.llm.invoke(
                input=[
                    {"role": "system", "content": METHODOLOGY_DETECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            detected_methodologies = json.loads(str(response.content))
            
            # Validate the response format
            if not isinstance(detected_methodologies, list):
                logger.error("Invalid response format: expected list")
                return []
                
            return detected_methodologies
            
        except Exception as e:
            logger.error(f"Error detecting methodologies: {str(e)}")
            return []
    
    def calculate_paper_score(self, detected_methodologies: Dict) -> float:
        """
        Calculate score based on detected methodologies.
    
        Args:
            detected_methodologies (Dict): Dictionary with keys 'high', 'medium', 'low'
                                        and values as lists of detected methodologies
                                        
        Returns:
            float: Calculated methodology score
        """
        score = 0.0

        # Calculate high impact score (max 3.0)
        high_score = min(len(detected_methodologies.get('high', [])) * 1.0, 3.0)
        
        # Calculate medium impact score (max 1.5)
        moderate_score = min(len(detected_methodologies.get('medium', [])) * 0.5, 1.5)
        
        # Calculate low impact penalty (max -1.0)
        low_penalty = -min(len(detected_methodologies.get('low', [])) * 0.3, 1.0)

        score = high_score + moderate_score + low_penalty

        return score
    
    def calculate_interest_score(self, paper: Paper, analysis: PaperAnalysis) -> Tuple[float, Dict]:
        """
        Calculate a deterministic interest score for a paper based on multiple factors.
        
        Args:
            paper (Paper): The paper to score
            analysis (PaperAnalysis): The analysis results
            
        Returns:
            Tuple[float, Dict]: Interest score (0-10) and breakdown of scoring factors
        """
        score = 0.0
        breakdown = {}
        
        # 1. Methodology-based scoring (0-4 points)
        try:
            # Detect methodologies in the paper
            high_methodologies = self.detect_methodologies(paper.abstract + " " + paper.conclusion, self.HIGH_IMPACT_METHODOLOGIES)
            medium_methodologies = self.detect_methodologies(paper.abstract + " " + paper.conclusion, self.MEDIUM_IMPACT_METHODOLOGIES)
            low_methodologies = self.detect_methodologies(paper.abstract + " " + paper.conclusion, self.LOW_IMPACT_METHODOLOGIES)
            
            # Filter detected methodologies
            detected_high = [m for m in high_methodologies if m.get('present', 0) == 1]
            detected_medium = [m for m in medium_methodologies if m.get('present', 0) == 1]
            detected_low = [m for m in low_methodologies if m.get('present', 0) == 1]
            
            methodology_data = {
                'high': detected_high,
                'medium': detected_medium,
                'low': detected_low
            }
            
            methodology_score = self.calculate_paper_score(methodology_data)
            score += methodology_score
            breakdown['methodology_score'] = methodology_score
            breakdown['detected_methodologies'] = methodology_data
            
        except Exception as e:
            logger.error(f"Error calculating methodology score: {str(e)}")
            breakdown['methodology_score'] = 0.0
        
        # 2. Content length scoring (0-1 point)
        content_length = len(paper.abstract) + len(paper.conclusion)
        if content_length > 2000:
            content_score = 1.0
        elif content_length > 1000:
            content_score = 0.7
        elif content_length > 500:
            content_score = 0.4
        else:
            content_score = 0.1
        score += content_score
        breakdown['content_score'] = content_score
        
        # 3. Author count scoring (0-1 point)
        author_count = len(paper.authors)
        if author_count >= 10:
            author_score = 1.0
        elif author_count >= 5:
            author_score = 0.8
        elif author_count >= 3:
            author_score = 0.6
        elif author_count >= 2:
            author_score = 0.4
        else:
            author_score = 0.2
        score += author_score
        breakdown['author_score'] = author_score
        
        # 4. Category relevance scoring (0-2 points)
        medical_categories = ['q-bio', 'stat.ML', 'cs.AI', 'cs.LG', 'cs.CV', 'cs.CL']
        relevant_categories = [cat for cat in paper.categories if any(med_cat in cat for med_cat in medical_categories)]
        if len(relevant_categories) >= 3:
            category_score = 2.0
        elif len(relevant_categories) >= 2:
            category_score = 1.5
        elif len(relevant_categories) >= 1:
            category_score = 1.0
        else:
            category_score = 0.5
        score += category_score
        breakdown['category_score'] = category_score
        
        # 5. Keyword relevance scoring (0-2 points)
        if analysis.keywords and len(analysis.keywords) >= 3:
            keyword_score = 2.0
        elif analysis.keywords and len(analysis.keywords) >= 1:
            keyword_score = 1.0
        else:
            keyword_score = 0.0
        score += keyword_score
        breakdown['keyword_score'] = keyword_score
        
        # Ensure score is within 0-10 range
        score = max(0.0, min(10.0, score))
        
        return score, breakdown
    
    def get_high_interest_papers(self, papers_with_analyses: List[Dict]) -> List[Dict]:
        """
        Filter papers to get only those with high interest scores (>= 7.0).
        
        Args:
            papers_with_analyses (List[Dict]): List of papers with their analysis results
            
        Returns:
            List[Dict]: Papers with high interest scores, sorted by score (highest first)
        """
        high_interest = []
        for paper_data in papers_with_analyses:
            if 'interest_score' in paper_data and paper_data['interest_score'] >= 7.0:
                high_interest.append(paper_data)
        
        # Sort by interest score (highest first)
        high_interest.sort(key=lambda x: x['interest_score'], reverse=True)
        return high_interest
    
    def get_papers_by_interest_range(self, papers_with_analyses: List[Dict], min_score: float = 0.0, max_score: float = 10.0) -> List[Dict]:
        """
        Get papers within a specific interest score range.
        
        Args:
            papers_with_analyses (List[Dict]): List of papers with their analysis results
            min_score (float): Minimum interest score (inclusive)
            max_score (float): Maximum interest score (inclusive)
            
        Returns:
            List[Dict]: Papers within the specified interest score range, sorted by score
        """
        filtered_papers = []
        for paper_data in papers_with_analyses:
            if 'interest_score' in paper_data and min_score <= paper_data['interest_score'] <= max_score:
                filtered_papers.append(paper_data)
        
        # Sort by interest score (highest first)
        filtered_papers.sort(key=lambda x: x['interest_score'], reverse=True)
        return filtered_papers 