from Data_Classes.classes import Paper, PaperAnalysis
from Data_Retrieval.data_retrieval import ArxivClient
from AI_Processing.paper_analyzer import PaperAnalyzer
from typing import List, Dict
import logging
import time
import datetime
import json

logger = logging.getLogger(__name__)


class ResearchDigest:
    """
    Main class for generating medical research digests.
    
    This class orchestrates the process of:
    1. Fetching medical research papers from arXiv via data_retrieval.py
    2. Analyzing papers using AI to extract key information via paper_analyzer.py
    3. Analyzing the results of the AI analysis to generate a comprehensive summary and insights in batches of 20 papers at the time.
    4. Generating comprehensive summaries and insights across all batches.
    5. Saving the results to a JSON file.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the research digest generator.
        
        Args:
            api_key (str): API key for the AI service
        """
        self.arxiv_client = ArxivClient()
        self.analyzer = PaperAnalyzer(api_key)
        self.llm = self.analyzer.llm  # Get LLM instance from analyzer
        self.specialty_data: Dict[str, Dict] = {}
        self.batch_analyses: Dict[int, Dict] = {}

    def generate_digest(self, search_query: str = "all:medical") -> Dict:
        """
        Generate a research digest for medical papers in batches of 20 papers at the time.
        
        Args:
            search_query (str): The search query to use for finding papers
            
        Returns:
            Dict: The digest summary as a dictionary

        Note:
            The digest includes paper analysis, specialty categorization,
            and AI-generated summaries of key findings.
        """
        logger.info("Fetching papers from arXiv...")
        papers = self.arxiv_client.fetch_papers(search_query)
        logger.info(f"Found {len(papers)} papers")
        
        self._analyze_papers(papers)
        self._batch_analyze_papers(self.specialty_data)
        
        # Wait until the beginning of the next minute to avoid rate limiting
        now = datetime.datetime.now()
        next_minute = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
        wait_seconds = (next_minute - now).total_seconds()
        time.sleep(wait_seconds)

        # Generate and store the digest
        self.digest_json = self._digest_summary()
        return self.digest_json
    
    def _analyze_papers(self, papers: List[Paper]) -> None:
        """
        Analyze the fetched papers using AI.
        
        Args:
            papers (List[Paper]): List of papers to analyze
        """
        logger.info("Analyzing papers with AI...")
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"Analyzing paper {i}/{len(papers)}: {paper.title[:80]}...")
            
            analysis = self.analyzer.analyze_paper(paper)
            if not analysis:
                continue
                
            self._update_specialty_data(paper, analysis)
    
    def _update_specialty_data(self, paper: Paper, analysis: PaperAnalysis) -> None:
        """
        Update the specialty data with the paper analysis.
        
        Args:
            paper (Paper): The analyzed paper
            analysis (PaperAnalysis): The AI analysis results
            
        Note:
            This method organizes papers by specialty and maintains
            metadata including keywords and author networks.
        """
        if analysis.specialty not in self.specialty_data:
            self.specialty_data[analysis.specialty] = {
                "papers": [],
                "all_keywords": [],
                "author_network": set()
            }
        
        self.specialty_data[analysis.specialty]["papers"].append({
            "id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "keywords": analysis.keywords,
            "focus": analysis.focus,
            "date": paper.published.strftime("%Y-%m-%d")
        })
        self.specialty_data[analysis.specialty]["all_keywords"].extend(analysis.keywords)
        self.specialty_data[analysis.specialty]["author_network"].update(paper.authors)

    def _batch_analyze_papers(self, specialty_data: Dict[str, Dict]) -> None:
        """
        Analyze the fetched papers using AI in batches of 10 papers at the time.
        
        Args:
            specialty_data (Dict[str, Dict]): Dictionary containing specialty data
        """
        logger.info("Analyzing papers with AI in batches of 10 papers at the time...")
        
        # Collect all papers from all specialties
        all_papers = []
        for specialty, data in specialty_data.items():
            for paper in data['papers']:
                paper['specialty'] = specialty  # Add specialty info to each paper
                all_papers.append(paper)
        
        total_papers = len(all_papers)
        total_batches = (total_papers + 9) // 10  # Calculate total number of batches (reduced from 20 to 10)
        
        logger.info(f"Total papers to analyze: {total_papers} in {total_batches} batches")
        
        for i in range(0, total_papers, 10):
            batch_num = i // 10 + 1
            batch = all_papers[i:i+10]
            logger.info(f"Analyzing batch {batch_num} of {total_batches} ({len(batch)} papers)...")
            
            # Create detailed batch information for the prompt (simplified to reduce token count)
            batch_info = []
            for j, paper in enumerate(batch, 1):
                # Truncate abstract to reduce token count
                abstract = paper['abstract'][:500] + "..." if len(paper['abstract']) > 500 else paper['abstract']
                batch_info.append(f"""
                Paper {j}:
                Title: {paper['title']}
                Abstract: {abstract}
                Specialty: {paper['specialty']}
                Keywords: {', '.join(paper['keywords'][:3])}  # Limit to 3 keywords
                """)
            
            batch_text = "\n".join(batch_info)
            
            prompt = f"""
            Analyze the following batch of {len(batch)} medical research papers:
            
            {batch_text}
            
            Provide a comprehensive analysis in the following JSON format. IMPORTANT: Return ONLY valid JSON, no additional text:
            {{
                "batch_summary": "2-3 paragraph summary focusing on key findings and implications for current medical practices",
                "significant_findings": ["List of top 5 most significant findings across all papers in this batch"],
                "major_trends": ["List of 2-3 major trends or patterns identified across multiple papers in this batch"],
                "medical_impact": "Brief analysis of potential impact on medical practice and patient care",
                "cross_specialty_insights": "Brief analysis of cross-specialty implications and connections",
                "medical_keywords": ["List of 10-15 relevant medical keywords across all research papers in this batch"],
                "papers_analyzed": {len(batch)},
                "batch_number": {batch_num},
                "specialties_covered": ["List of medical specialties represented in this batch"]
            }}
            
            Ensure the analysis is comprehensive and provides sufficient detail for later integration into a complete newsletter digest.
            Return ONLY the JSON object, no additional text or explanations.
            """
            
            response = None
            try:
                # Get AI analysis for this batch
                response = self.llm.invoke(prompt)
                
                # Check if response is empty or invalid
                if not response.content or response.content.strip() == "":
                    logger.error(f"Empty response from LLM for batch {batch_num}")
                    raise ValueError("Empty response from LLM")
                
                # Parse the JSON response
                batch_analysis = json.loads(response.content)
                
                # Store the batch analysis
                self.batch_analyses[batch_num] = {
                    "papers": batch,
                    "analysis": batch_analysis,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                logger.info(f"Successfully analyzed batch {batch_num}")
                
                # Rate limiting: wait until the beginning of the next minute between batches to avoid API limits
                if batch_num < total_batches:
                    logger.info("Waiting until the beginning of the next minute before next batch...")
                    now = datetime.datetime.now()
                    next_minute = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
                    wait_seconds = (next_minute - now).total_seconds()
                    time.sleep(wait_seconds)
                    
            except Exception as e:
                logger.error(f"Error analyzing batch {batch_num}: {str(e)}")
                if response:
                    logger.error(f"Response content: {getattr(response, 'content', 'No content')}")
                # Store error information for this batch
                self.batch_analyses[batch_num] = {
                    "papers": batch,
                    "error": str(e),
                    "timestamp": datetime.datetime.now().isoformat()
                }
        
        logger.info(f"Completed batch analysis. Processed {len(self.batch_analyses)} batches.")

    def _generate_executive_summary(self) -> str:
        """
        Generate an AI-generated executive summary from the batch analyses.
        
        Returns:
            str: The executive summary
        """
        print("Generating executive summary...")

        # Extract only the analysis results from each batch
        batch_analysis_results = []
        for batch_num, batch_data in self.batch_analyses.items():
            if "analysis" in batch_data:
                batch_analysis_results.append({
                    "batch_number": batch_num,
                    "analysis": batch_data["analysis"]
                })
            else:
                logger.warning(f"Batch {batch_num} has no analysis results, skipping...")

        # Create a list of all batch analyses
        prompt = f"""
        Generate an AI-generated executive summary from the following batch analysis results:
        {batch_analysis_results}
        
        The summary should be 2-3 paragraphs long and focus on the key findings and implications of the research papers across all batches.
        """
        response = self.llm.invoke(prompt)
        
        # Check if response is empty or invalid
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from LLM for executive summary")
            return "No executive summary available due to processing errors."
        
        return response.content
        
    def _generate_key_discoveries(self) -> list:
        """
        Generate a list of key discoveries from the batch analyses.
        
        Returns:
            list: The key discoveries
        """
        print("Generating key discoveries...")

        # Extract only the analysis results from each batch
        batch_analysis_results = []
        for batch_num, batch_data in self.batch_analyses.items():
            if "analysis" in batch_data:
                batch_analysis_results.append({
                    "batch_number": batch_num,
                    "analysis": batch_data["analysis"]
                })
            else:
                logger.warning(f"Batch {batch_num} has no analysis results, skipping...")

        prompt = f"""
        Generate a list of key discoveries from the following batch analysis results:
        {batch_analysis_results}
        
        Return the response as a JSON array of exactly 10 key discoveries across all batches.
        Format: ["discovery 1", "discovery 2", ..., "discovery 10"]
        """
        response = self.llm.invoke(prompt)
        
        try:
            # Check if response is empty or invalid
            if not response.content or response.content.strip() == "":
                logger.error("Empty response from LLM for key discoveries")
                return []
            
            # Parse the JSON response to get a list
            key_discoveries = json.loads(response.content)
            if isinstance(key_discoveries, list):
                return key_discoveries
            else:
                logger.error("LLM response is not a list, returning empty list")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {response.content}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in key discoveries: {e}")
            return []

    def _generate_emerging_trends(self) -> str:
        """
        Generate 1-2 paragraphs on emerging trends from the batch analyses.
        
        Returns:
            str: The emerging trends
        """
        print("Generating emerging trends...")
        
        # Extract only the analysis results from each batch
        batch_analysis_results = []
        for batch_num, batch_data in self.batch_analyses.items():
            if "analysis" in batch_data:
                batch_analysis_results.append({
                    "batch_number": batch_num,
                    "analysis": batch_data["analysis"]
                })
            else:
                logger.warning(f"Batch {batch_num} has no analysis results, skipping...")

        prompt = f"""
        Generate 1-2 paragraphs on emerging trends from the following batch analysis results:
        {batch_analysis_results}
        
        The trends should be 1-2 paragraphs long and focus on the emerging trends in the research papers across all batches.
        """
        response = self.llm.invoke(prompt)
        
        # Check if response is empty or invalid
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from LLM for emerging trends")
            return "No emerging trends available due to processing errors."
        
        return response.content
        
    def _generate_medical_impact(self) -> str:
        """
        Generate 1 paragraph on the potential impact of the research papers on medical practice and patient care.
        
        Returns:
            str: The medical impact
        """
        print("Generating medical impact...")
        
        # Extract only the analysis results from each batch
        batch_analysis_results = []
        for batch_num, batch_data in self.batch_analyses.items():
            if "analysis" in batch_data:
                batch_analysis_results.append({
                    "batch_number": batch_num,
                    "analysis": batch_data["analysis"]
                })
            else:
                logger.warning(f"Batch {batch_num} has no analysis results, skipping...")

        prompt = f"""
        Generate 1 paragraph on the potential impact of the research papers on medical practice and patient care from the following batch analysis results:
        {batch_analysis_results}
        
        The impact should be 1 paragraph long and focus on the potential impact of the research papers on medical practice and patient care across all batches.
        """
        response = self.llm.invoke(prompt)
        
        # Check if response is empty or invalid
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from LLM for medical impact")
            return "No medical impact analysis available due to processing errors."
        
        return response.content
    
    def _generate_cross_specialty_insights(self) -> str:
        """
        Generate 1 paragraph on the cross-specialty implications of the research papers.
        
        Returns:
            str: The cross-specialty implications
        """
        print("Generating cross-specialty implications...")

        # Extract only the analysis results from each batch
        batch_analysis_results = []
        for batch_num, batch_data in self.batch_analyses.items():
            if "analysis" in batch_data:
                batch_analysis_results.append({
                    "batch_number": batch_num,
                    "analysis": batch_data["analysis"]
                })
            else:
                logger.warning(f"Batch {batch_num} has no analysis results, skipping...")

        prompt = f"""
        Generate 1-2 paragraphs on the cross-specialty implications of the research papers from the following batch analysis results:
        {batch_analysis_results}
        
        The implications should be 1-2 paragraphs long and focus on the cross-specialty implications of the research papers across all batches.
        """
        response = self.llm.invoke(prompt)
        
        # Check if response is empty or invalid
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from LLM for cross-specialty insights")
            return "No cross-specialty insights available due to processing errors."
        
        return response.content
    
    def _generate_clinical_implications(self) -> str:
        """
        Generate 1-2 paragraphs on the clinical implications of the research papers.
        
        Returns:
            str: The clinical implications
        """
        print("Generating clinical implications...")

        # Extract only the analysis results from each batch
        batch_analysis_results = []
        for batch_num, batch_data in self.batch_analyses.items():
            if "analysis" in batch_data:
                batch_analysis_results.append({
                    "batch_number": batch_num,
                    "analysis": batch_data["analysis"]
                })
            else:
                logger.warning(f"Batch {batch_num} has no analysis results, skipping...")

        prompt = f"""
        Generate 1-2 paragraphs on the clinical implications of the research papers from the following batch analysis results:
        {batch_analysis_results}
        
        The implications should be 1-2 paragraphs long and focus on the clinical implications of the research papers across all batches.
        """
        response = self.llm.invoke(prompt)
        
        # Check if response is empty or invalid
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from LLM for clinical implications")
            return "No clinical implications available due to processing errors."
        
        return response.content
    
    def _generate_research_gaps(self) -> str:
        """
        Generate 1 paragraph on the research gaps in the research papers.
        
        Returns:
            str: The research gaps
        """
        print("Generating research gaps...")

        # Extract only the analysis results from each batch
        batch_analysis_results = []
        for batch_num, batch_data in self.batch_analyses.items():
            if "analysis" in batch_data:
                batch_analysis_results.append({
                    "batch_number": batch_num,
                    "analysis": batch_data["analysis"]
                })
            else:
                logger.warning(f"Batch {batch_num} has no analysis results, skipping...")

        prompt = f"""
        Generate 1 paragraph on the research gaps in the research papers from the following batch analysis results:
        {batch_analysis_results}
        
        The gaps should be 1 paragraph long and focus on the research gaps in the research papers across all batches.
        """
        response = self.llm.invoke(prompt)
        
        # Check if response is empty or invalid
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from LLM for research gaps")
            return "No research gaps analysis available due to processing errors."
        
        return response.content
    
    def _generate_future_directions(self) -> str:
        """
        Generate 1 paragraph on the future directions of the research papers.
        
        Returns:
            str: The future directions
        """
        print("Generating future directions...")

        # Extract only the analysis results from each batch
        batch_analysis_results = []
        for batch_num, batch_data in self.batch_analyses.items():
            if "analysis" in batch_data:
                batch_analysis_results.append({
                    "batch_number": batch_num,
                    "analysis": batch_data["analysis"]
                })
            else:
                logger.warning(f"Batch {batch_num} has no analysis results, skipping...")

        prompt = f"""
        Generate 1 paragraph on the future directions of the research papers from the following batch analysis results:
        {batch_analysis_results}
        
        The directions should be 1 paragraph long and focus on the future directions of the research papers across all batches.
        """
        response = self.llm.invoke(prompt)
        
        # Check if response is empty or invalid
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from LLM for future directions")
            return "No future directions analysis available due to processing errors."
        
        return response.content
    
    def _digest_summary(self) -> dict:
        """
        Generate a comprehensive digest JSON summary from the batch analyses.
        
        Returns:
            dict: The digest
        """
        print("Generating digest summary JSON...")

        # Generate the digest
        digest = {
            "executive_summary": self._generate_executive_summary(),
            "key_discoveries": self._generate_key_discoveries(),
            "emerging_trends": self._generate_emerging_trends(),
            "medical_impact": self._generate_medical_impact(),
            "cross_specialty_insights": self._generate_cross_specialty_insights(),
            "clinical_implications": self._generate_clinical_implications(),
            "research_gaps": self._generate_research_gaps(),
            "future_directions": self._generate_future_directions()
        }
        
        # Add missing required fields
        digest["date_generated"] = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Calculate total papers from specialty data
        total_papers = sum(len(data["papers"]) for data in self.specialty_data.values())
        digest["total_papers"] = total_papers
        
        # Add papers list for newsletter compatibility
        digest["papers"] = []
        for specialty, data in self.specialty_data.items():
            for paper in data["papers"]:
                paper_copy = paper.copy()
                paper_copy["specialty"] = specialty
                digest["papers"].append(paper_copy)

        return digest
