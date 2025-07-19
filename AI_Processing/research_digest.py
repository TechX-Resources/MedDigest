from Data_Classes.classes import Paper, PaperAnalysis
from Data_Retrieval.data_retrieval import ArxivClient
from AI_Processing.paper_analyzer import PaperAnalyzer
from utils.token_monitor import TokenMonitor
from typing import List, Dict
import logging
import datetime
import json
import re
import time
from .prompts import (
    create_batch_analysis_prompt,
    create_executive_summary_prompt,
    create_key_discoveries_prompt,
    create_emerging_trends_prompt,
    create_medical_impact_prompt,
    create_cross_specialty_insights_prompt,
    create_clinical_implications_prompt,
    create_research_gaps_prompt,
    create_future_directions_prompt
)

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
        self.token_monitor = TokenMonitor(max_tokens_per_minute=16000)
        self.analyzer = PaperAnalyzer(api_key, token_monitor=self.token_monitor)
        self.llm = self.analyzer.llm
        self.specialty_data: Dict[str, Dict] = {}
        self.batch_analyses: Dict[int, Dict] = {}
        self.rate_limit_threshold = 14000  # Sleep when approaching 14k tokens (leaving 2k buffer)
        self.current_minute_tokens = 0
        self.minute_start_time = time.time()

    def _check_rate_limit(self, estimated_tokens: int) -> None:
        """
        Check if we're approaching the rate limit and sleep if necessary.
        
        Args:
            estimated_tokens (int): Estimated tokens for the next operation
        """
        current_time = time.time()
        
        # Reset counter if a new minute has started
        if current_time - self.minute_start_time >= 60:
            self.current_minute_tokens = 0
            self.minute_start_time = current_time
        
        # Check if adding these tokens would exceed our threshold
        if self.current_minute_tokens + estimated_tokens > self.rate_limit_threshold:
            # Calculate how long to sleep until the next minute
            time_elapsed = current_time - self.minute_start_time
            sleep_time = 60 - time_elapsed
            
            if sleep_time > 0:
                logger.info(f"Approaching rate limit ({self.current_minute_tokens} tokens used). Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.current_minute_tokens = 0
                self.minute_start_time = time.time()
        
        # Update token count for this minute
        self.current_minute_tokens += estimated_tokens

    def generate_digest(self, search_query: str = "all:medical") -> Dict:
        """
        Generate a research digest for medical papers in batches of 10 papers at the time.
        
        Args:
            search_query (str): The search query to use for finding papers
            
        Returns:
            Dict: The digest summary as a dictionary

        Note:
            The digest includes paper analysis, specialty categorization,
            and AI-generated summaries of key findings.
        """
        logger.info("\nFetching papers from arXiv...")
        papers = self.arxiv_client.fetch_papers(search_query)
        logger.info(f"Found {len(papers)} papers")
        
        self._analyze_papers(papers)
        self._batch_analyze_papers(self.specialty_data)
        
        # Token monitor handles rate limiting automatically, no need for manual sleep

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
            
            # Estimate tokens for this paper analysis (rough approximation)
            estimated_tokens = len(paper.title + paper.abstract + paper.conclusion) // 4 + 500  # Add buffer for prompt and response
            self._check_rate_limit(estimated_tokens)
            
            result = self.analyzer.analyze_paper(paper)
            if not result or result[0] is None:
                continue
                
            analysis, usage = result
            if analysis is not None:
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
            "interest_score": analysis.interest_score,
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
        
        logger.info(f"Total papers to analyze: {total_papers} in {total_batches} {'batch' if total_batches == 1 else 'batches'}")
        
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
            
            # Estimate tokens for batch analysis
            estimated_tokens = len(batch_text) // 4 + 2000  # Add buffer for prompt and response
            self._check_rate_limit(estimated_tokens)
            
            prompt = create_batch_analysis_prompt(batch, batch_text, batch_num)
            
            response = None
            try:
                # Estimate input tokens (rough approximation: 1 token ≈ 4 characters)
                input_tokens = len(prompt) // 4
                
                # Get AI analysis for this batch
                response = self.llm.invoke(prompt)
                
                # Estimate output tokens
                output_tokens = len(response.content) // 4
                
                # Record token usage with enhanced tracking
                self.token_monitor.record_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    call_type="batch_analysis",
                    prompt_length=len(prompt),
                    response_length=len(response.content)
                )
                
                # Check if response is empty or invalid
                if not response.content or response.content.strip() == "":
                    logger.error(f"Empty response from LLM for batch {batch_num}")
                    raise ValueError("Empty response from LLM")
                
                # Extract JSON from the response
                batch_analysis = self._extract_json_from_response(response.content, "object")
                if batch_analysis is None:
                    raise ValueError("Failed to parse JSON response")
                
                # Store the batch analysis
                self.batch_analyses[batch_num] = {
                    "papers": batch,
                    "analysis": batch_analysis,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                logger.info(f"Successfully analyzed batch {batch_num}")
                
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

    def get_high_interest_papers_summary(self) -> dict:
        """
        Generate a summary of high-interest papers (score >= 7.0) across all specialties.
        
        Returns:
            dict: Summary of high-interest papers with statistics and top papers
        """
        all_papers = []
        for specialty, data in self.specialty_data.items():
            for paper in data['papers']:
                paper['specialty'] = specialty
                all_papers.append(paper)
        
        # Filter high-interest papers (score >= 7.0)
        high_interest_papers = [p for p in all_papers if p.get('interest_score', 0) >= 7.0]
        high_interest_papers.sort(key=lambda x: x['interest_score'], reverse=True)
        
        # Calculate statistics
        total_papers = len(all_papers)
        high_interest_count = len(high_interest_papers)
        avg_interest_score = sum(p.get('interest_score', 0) for p in all_papers) / total_papers if total_papers > 0 else 0
        
        # Group by specialty
        specialty_breakdown = {}
        for paper in high_interest_papers:
            specialty = paper['specialty']
            if specialty not in specialty_breakdown:
                specialty_breakdown[specialty] = []
            specialty_breakdown[specialty].append(paper)
        
        return {
            "total_papers": total_papers,
            "high_interest_count": high_interest_count,
            "high_interest_percentage": (high_interest_count / total_papers * 100) if total_papers > 0 else 0,
            "average_interest_score": round(avg_interest_score, 2),
            "top_papers": high_interest_papers[:10],  # Top 10 highest scoring papers
            "specialty_breakdown": specialty_breakdown,
            "interest_score_distribution": {
                "9-10": len([p for p in all_papers if 9.0 <= p.get('interest_score', 0) <= 10.0]),
                "7-8.9": len([p for p in all_papers if 7.0 <= p.get('interest_score', 0) < 9.0]),
                "5-6.9": len([p for p in all_papers if 5.0 <= p.get('interest_score', 0) < 7.0]),
                "3-4.9": len([p for p in all_papers if 3.0 <= p.get('interest_score', 0) < 5.0]),
                "0-2.9": len([p for p in all_papers if 0.0 <= p.get('interest_score', 0) < 3.0])
            }
        }

    def _extract_json_from_response(self, response_content: str, expected_type: str = "object") -> any:
        """
        Extract JSON from LLM response, handling cases where the response includes explanatory text.
        
        Args:
            response_content (str): The raw response from the LLM
            expected_type (str): Expected JSON type - "object" for dict, "array" for list
            
        Returns:
            any: Parsed JSON object or list, or None if parsing fails
        """
        try:
            # Clean the response - remove any markdown formatting
            cleaned_response = response_content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Look for JSON pattern based on expected type
            if expected_type == "array":
                json_pattern = r'\[.*\]'
            else:  # object
                json_pattern = r'\{.*\}'
                
            json_match = re.search(json_pattern, cleaned_response, re.DOTALL)
            
            if json_match:
                json_content = json_match.group(0)
                return json.loads(json_content)
            else:
                # Try to parse the entire response as JSON
                return json.loads(cleaned_response)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response content: {response_content[:300]}...")
            return None
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            return None

    def _make_llm_call_with_monitoring(self, prompt: str, call_type: str = "summary_generation") -> str:
        """
        Make an LLM call with token monitoring and rate limiting.
        
        Args:
            prompt (str): The prompt to send to the LLM
            call_type (str): Type of call for tracking purposes
            
        Returns:
            str: The LLM response content
        """
        # Estimate input tokens and check rate limit
        estimated_tokens = len(prompt) // 4 + 1000  # Add buffer for response
        self._check_rate_limit(estimated_tokens)
        
        # Estimate input tokens (rough approximation: 1 token ≈ 4 characters)
        input_tokens = len(prompt) // 4
        
        response = self.llm.invoke(prompt)
        
        # Estimate output tokens
        output_tokens = len(response.content) // 4
        
        # Record token usage with enhanced tracking
        self.token_monitor.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_type=call_type,
            prompt_length=len(prompt),
            response_length=len(response.content)
        )
        
        return response.content

    def _generate_executive_summary(self) -> str:
        """
        Generate an AI-generated executive summary from the batch analyses.
        
        Returns:
            str: The executive summary
        """
        print("\nGenerating executive summary...")

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

        prompt = create_executive_summary_prompt(batch_analysis_results)
        
        try:
            response_content = self._make_llm_call_with_monitoring(prompt, "executive_summary")
            
            # Check if response is empty or invalid
            if not response_content or response_content.strip() == "":
                logger.error("Empty response from LLM for executive summary")
                return "No executive summary available due to processing errors."
            
            return response_content
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return "No executive summary available due to processing errors."
        
    def _generate_key_discoveries(self) -> list:
        """
        Generate a list of key discoveries from the batch analyses.
        
        Returns:
            list: The key discoveries
        """
        print("\nGenerating key discoveries...")

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

        prompt = create_key_discoveries_prompt(batch_analysis_results)
        
        try:
            response_content = self._make_llm_call_with_monitoring(prompt, "key_discoveries")
            
            # Check if response is empty or invalid
            if not response_content or response_content.strip() == "":
                logger.error("Empty response from LLM for key discoveries")
                return []
            
            # Extract JSON array from the response
            key_discoveries = self._extract_json_from_response(response_content, "array")
            if key_discoveries is None:
                logger.error("Failed to parse JSON response for key discoveries")
                return []
            
            if isinstance(key_discoveries, list):
                return key_discoveries
            else:
                logger.error("LLM response is not a list, returning empty list")
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
        print("\nGenerating emerging trends...")
        
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

        prompt = create_emerging_trends_prompt(batch_analysis_results)
        
        try:
            response_content = self._make_llm_call_with_monitoring(prompt, "emerging_trends")
            
            # Check if response is empty or invalid
            if not response_content or response_content.strip() == "":
                logger.error("Empty response from LLM for emerging trends")
                return "No emerging trends available due to processing errors."
            
            return response_content
        except Exception as e:
            logger.error(f"Error generating emerging trends: {str(e)}")
            return "No emerging trends available due to processing errors."
        
    def _generate_medical_impact(self) -> str:
        """
        Generate 1 paragraph on the potential impact of the research papers on medical practice and patient care.
        
        Returns:
            str: The medical impact
        """
        print("\nGenerating medical impact...")
        
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

        prompt = create_medical_impact_prompt(batch_analysis_results)
        
        try:
            response_content = self._make_llm_call_with_monitoring(prompt, "medical_impact")
            
            # Check if response is empty or invalid
            if not response_content or response_content.strip() == "":
                logger.error("Empty response from LLM for medical impact")
                return "No medical impact analysis available due to processing errors."
            
            return response_content
        except Exception as e:
            logger.error(f"Error generating medical impact: {str(e)}")
            return "No medical impact analysis available due to processing errors."
    
    def _generate_cross_specialty_insights(self) -> str:
        """
        Generate 1 paragraph on the cross-specialty implications of the research papers.
        
        Returns:
            str: The cross-specialty implications
        """
        print("\nGenerating cross-specialty implications...")

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

        prompt = create_cross_specialty_insights_prompt(batch_analysis_results)
        
        try:
            response_content = self._make_llm_call_with_monitoring(prompt, "cross_specialty_insights")
            
            # Check if response is empty or invalid
            if not response_content or response_content.strip() == "":
                logger.error("Empty response from LLM for cross-specialty insights")
                return "No cross-specialty insights available due to processing errors."
            
            return response_content
        except Exception as e:
            logger.error(f"Error generating cross-specialty insights: {str(e)}")
            return "No cross-specialty insights available due to processing errors."
    
    def _generate_clinical_implications(self) -> str:
        """
        Generate 1-2 paragraphs on the clinical implications of the research papers.
        
        Returns:
            str: The clinical implications
        """
        print("\nGenerating clinical implications...")

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

        prompt = create_clinical_implications_prompt(batch_analysis_results)
        
        try:
            response_content = self._make_llm_call_with_monitoring(prompt, "clinical_implications")
            
            # Check if response is empty or invalid
            if not response_content or response_content.strip() == "":
                logger.error("Empty response from LLM for clinical implications")
                return "No clinical implications available due to processing errors."
            
            return response_content
        except Exception as e:
            logger.error(f"Error generating clinical implications: {str(e)}")
            return "No clinical implications available due to processing errors."
    
    def _generate_research_gaps(self) -> str:
        """
        Generate 1 paragraph on the research gaps in the research papers.
        
        Returns:
            str: The research gaps
        """
        print("\nGenerating research gaps...")

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

        prompt = create_research_gaps_prompt(batch_analysis_results)
        
        try:
            response_content = self._make_llm_call_with_monitoring(prompt, "research_gaps")
            
            # Check if response is empty or invalid
            if not response_content or response_content.strip() == "":
                logger.error("Empty response from LLM for research gaps")
                return "No research gaps analysis available due to processing errors."
            
            return response_content
        except Exception as e:
            logger.error(f"Error generating research gaps: {str(e)}")
            return "No research gaps analysis available due to processing errors."
    
    def _generate_future_directions(self) -> str:
        """
        Generate 1 paragraph on the future directions of the research papers.
        
        Returns:
            str: The future directions
        """
        print("\nGenerating future directions...")

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

        prompt = create_future_directions_prompt(batch_analysis_results)
        
        try:
            response_content = self._make_llm_call_with_monitoring(prompt, "future_directions")
            
            # Check if response is empty or invalid
            if not response_content or response_content.strip() == "":
                logger.error("Empty response from LLM for future directions")
                return "No future directions analysis available due to processing errors."
            
            return response_content
        except Exception as e:
            logger.error(f"Error generating future directions: {str(e)}")
            return "No future directions analysis available due to processing errors."
    
    def _print_highest_rated_papers_per_specialty(self) -> None:
        """
        Print the title of the highest rated paper per specialty to the console.
        """
        print("\n" + "="*80)
        print("HIGHEST RATED PAPERS PER SPECIALTY")
        print("="*80)
        
        for specialty, data in self.specialty_data.items():
            if not data["papers"]:
                continue
                
            # Find the paper with the highest interest score in this specialty
            highest_rated_paper = max(data["papers"], key=lambda p: p.get('interest_score', 0))
            
            print(f"\n{specialty.upper()}:")
            print(f"  Title: {highest_rated_paper['title']}")
            print(f"  Interest Score: {highest_rated_paper.get('interest_score', 'N/A')}")
            print(f"  Authors: {', '.join(highest_rated_paper['authors'][:3])}{'...' if len(highest_rated_paper['authors']) > 3 else ''}")
        
        print("\n" + "="*80)

    def _digest_summary(self) -> dict:
        """
        Generate a comprehensive digest JSON summary from the batch analyses.
        
        Returns:
            dict: The digest
        """
        print("\nGenerating digest summary JSON...")

        # Print highest rated papers per specialty
        self._print_highest_rated_papers_per_specialty()

        # Generate the digest
        digest = {
            "executive_summary": self._generate_executive_summary(),
            "key_discoveries": self._generate_key_discoveries(),
            "emerging_trends": self._generate_emerging_trends(),
            "medical_impact": self._generate_medical_impact(),
            "cross_specialty_insights": self._generate_cross_specialty_insights(),
            "clinical_implications": self._generate_clinical_implications(),
            "research_gaps": self._generate_research_gaps(),
            "future_directions": self._generate_future_directions(),
            "high_interest_papers": self.get_high_interest_papers_summary()
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
