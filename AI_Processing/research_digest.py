from Data_Classes.classes import Paper, PaperAnalysis
from Data_Retrieval.data_retrieval import ArxivClient
from AI_Processing.paper_analyzer import PaperAnalyzer
from utils.token_monitor import TokenMonitor
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
    1. Fetching medical research papers from arXiv
    2. Analyzing papers using AI to extract key information
    3. Organizing findings by medical specialty
    4. Generating comprehensive summaries and insights
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the research digest generator.
        
        Args:
            api_key (str): API key for the AI service
        """
        self.arxiv_client = ArxivClient()
        self.token_monitor = TokenMonitor(max_tokens_per_minute=16000)  # Groq Llama 3 8B limit
        self.analyzer = PaperAnalyzer(api_key, token_monitor=self.token_monitor)
        self.llm = self.analyzer.llm  # Get LLM instance from analyzer
        self.specialty_data: Dict[str, Dict] = {}
    
    def generate_digest(self, search_query: str = "all:medical", max_results: int = 10) -> None:
        """
        Generate a research digest for medical papers.
        
        Args:
            search_query (str): The search query to use for finding papers
            max_results (int): Maximum number of papers to analyze
            
        Note:
            The digest includes paper analysis, specialty categorization,
            and AI-generated summaries of key findings.
        """
        logger.info("Fetching papers from arXiv...")
        papers = self.arxiv_client.fetch_papers(search_query)
        logger.info(f"Found {len(papers)} papers")
        
        self._analyze_papers(papers)
        self._display_summary()
        
        # Display token usage statistics
        self.token_monitor.print_usage_summary()
        
        time.sleep(60) 
        self._digest_summary()
    
    def _analyze_papers(self, papers: List[Paper]) -> None:
        """
        Analyze the fetched papers using AI.
        
        Args:
            papers (List[Paper]): List of papers to analyze
            
        Note:
            Each paper is analyzed to determine its specialty,
            key concepts, and main findings.
        """
        logger.info("Analyzing papers with AI...")
        self.per_paper_usage = []  
        for i, paper in enumerate(papers, 1):
            logger.info(f"Analyzing paper {i}/{len(papers)}: {paper.title[:80]}...")
            analysis, usage = self.analyzer.analyze_paper(paper)
            if not analysis:
                continue
            self._update_specialty_data(paper, analysis)
            self.per_paper_usage.append((paper.title, usage))
        # Print per-paper usage after all analyses
        logger.info("\nPER-PAPER TOKEN USAGE:")
        for title, usage in self.per_paper_usage:
            if usage:
                logger.info(f"Paper: {title}")
                logger.info(f"  Input tokens: {usage.input_tokens}")
                logger.info(f"  Output tokens: {usage.output_tokens}")
                logger.info(f"  Total tokens: {usage.total_tokens}")
                logger.info(f"  Cost: ${usage.cost_usd:.4f}")
            else:
                logger.info(f"Paper: {title} (usage unavailable)")
    
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
            "keywords": analysis.keywords,
            "focus": analysis.focus,
            "date": paper.published.strftime("%Y-%m-%d")
        })
        self.specialty_data[analysis.specialty]["all_keywords"].extend(analysis.keywords)
        self.specialty_data[analysis.specialty]["author_network"].update(paper.authors)
    
    def _display_summary(self) -> None:
        """
        Display the research summary organized by specialty.
        
        Note:
            This method provides a detailed breakdown of:
            - Number of papers per specialty
            - Recent papers and their focus
            - Top research terms and their frequency
        """
        logger.info("\n" + "="*60)
        logger.info("MEDICAL RESEARCH SUMMARY BY SPECIALTY:")
        logger.info("="*60)
        
        for specialty, data in sorted(
            self.specialty_data.items(), 
            key=lambda x: len(x[1]["papers"]), 
            reverse=True
        ):
            num_papers = len(data["papers"])
            num_authors = len(data["author_network"])
            
            logger.info(f"\n{specialty.upper()} ({num_papers} papers, {num_authors} unique authors)")
            logger.info("-"*50)
            
            # Recent papers
            logger.info("\nRecent Papers:")
            for j, paper in enumerate(data["papers"][:3], 1):
                logger.info(f"  {j}. [{paper['date']}] {paper['title'][:60]}...")
                focus_text = paper.get('focus', 'No focus available')
                if not isinstance(focus_text, str) or not focus_text:
                    focus_text = 'No focus available'
                logger.info(f"     Focus: {focus_text[:80]}...")
            
            # Top keywords
            keyword_freq = {}
            for kw in data["all_keywords"]:
                keyword_freq[kw.lower()] = keyword_freq.get(kw.lower(), 0) + 1
            
            logger.info("\nTop Research Terms:")
            top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:8]
            for keyword, count in top_keywords:
                logger.info(f"  • {keyword} ({count} papers)")

    def _digest_summary(self) -> None:
        """
        Generate a concise AI-powered summary of key findings across all papers.
        Outputs JSON that can be used by the Newsletter class.
        """

        # Collect all paper information for comprehensive analysis
        all_papers_info = []
        specialty_breakdown = {}

        for specialty, data in self.specialty_data.items():
            specialty_breakdown[specialty] = len(data["papers"])
            for paper in data["papers"]:
                paper_summary = {
                    "title": paper["title"],
                    "specialty": specialty,
                    "focus": paper["focus"],
                    "date": paper["date"],
                    "authors": paper["authors"],
                    "keywords": paper["keywords"]
                }
                all_papers_info.append(paper_summary)

        # Create coprehensive prompt for combined analysis
        try:
            # Prepare paper summaries for prompt
            papers_text = []
            for i, paper in enumerate(all_papers_info, 1):
                paper_text = f"""
                Paper {i}:
                Title: {paper['title']}
                Specialty: {paper['specialty']}
                Focus: {paper['focus']}
                Date: {paper['date']}
                Authors: {paper['authors']}
                Keywords: {paper['keywords']}
                """
                papers_text.append(paper_text)

            combined_papers = '\n'.join(papers_text)

            # System prompt that defines the AI's role and analysis requirements
            SYSTEM_ROLE = """
            You are an expert medical research analyst with deep knowledge across all medical specialties. 
            Your task is to summarize and present the key findings of recently published papers across multiple medical specialties how they interact with eachother.
            """

            prompt = f"""
            Analyze this collection of {len(all_papers_info)} medical research papers published this past week and provide a comprehensive digest:
            
            PAPERS TO ANALYZE:
            {combined_papers}
            
            SPECIALTY BREAKDOWN:
            {', '.join([f"{spec}: {count} papers" for spec, count in specialty_breakdown.items()])}
            
            Please provide a comprehensive analysis with the following sections:
            
            1. EXECUTIVE SUMMARY: A 2-3 paragraph overview of the entire research collection
            2. KEY DISCOVERIES: Top 10 most significant findings across all papers (bullet points)
            3. EMERGING TRENDS: 3-4 major trends or patterns identified across multiple papers if any
            4. CROSS-SPECIALTY INSIGHTS: How different medical fields are interconnecting in this research
            5. CLINICAL IMPLICATIONS: Potential impact on medical practice and patient care
            6. RESEARCH GAPS: Areas that need further investigation
            7. FUTURE DIRECTIONS: Where this collective research is heading
            
            Format your response as a JSON object and provide actionable insights that synthesize information across all papers rather than discussing individual studies. 
            Only provide insights based on the material you have been provided with.

            JSON FORMAT:
            {{
                "executive_summary": "EXECUTIVE SUMMARY",
                "key_discoveries": "KEY DISCOVERIES",
                "emerging_trends": "EMERGING TRENDS",
                "cross_specialty_insights": "CROSS-SPECIALTY INSIGHTS",
                "clinical_implications": "CLINICAL IMPLICATIONS",
                "research_gaps": "RESEARCH GAPS",
                "future_directions": "FUTURE DIRECTIONS",
            }}
            """

            response = self.llm.invoke(
                input=[
                    {"role": "system", "content": SYSTEM_ROLE},
                    {"role": "user", "content": prompt}
                ]
            )

            # Record token usage for digest summary
            input_text = SYSTEM_ROLE + prompt
            estimated_input_tokens = len(input_text) // 4
            output_tokens = len(str(response.content)) // 4
            
            self.token_monitor.record_usage(
                input_tokens=estimated_input_tokens,
                output_tokens=output_tokens
            )

            # Try to parse the response as JSON
            try:
                if hasattr(response, 'content'):
                    response_text = response.content
                elif isinstance(response, dict) and 'content' in response:
                    response_text = response['content']
                else:
                    response_text = str(response)

                digest_json = json.loads(response_text)
                print(json.dumps(digest_json, indent=2))

            except json.JSONDecodeError as e:
                print("[ERROR] Could not parse LLM response as JSON:", e)
                print("Raw response:", response_text)
                
        except Exception as e:
            print(f"[ERROR] Exception during digest summary: {e}")