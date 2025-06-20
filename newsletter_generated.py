import datetime
import logging
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from urllib.parse import urlencode
from langchain_groq import ChatGroq
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Data class representing a research paper."""
    paper_id: str
    title: str
    published: datetime.datetime
    abstract: str
    authors: List[str]
    categories: List[str]
    conclusion: str

@dataclass
class PaperAnalysis:
    """Data class representing the analysis of a paper."""
    specialty: str
    keywords: List[str]
    focus: str

class ArxivClient:
    """Client for interacting with the arXiv API."""
    
    BASE_URL = "https://export.arxiv.org/api/query"
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/xml,application/xhtml+xml,text/html;q=0.9',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def fetch_papers(self, search_query: str, max_results: int = 20) -> List[Paper]:
        """
        Fetch papers from arXiv API with retry logic.
        
        Args:
            search_query: The search query to use
            max_results: Maximum number of results to return
            
        Returns:
            List of Paper objects
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
        """Parse the XML response from arXiv."""
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

class PaperAnalyzer:
    """Analyzes medical research papers using AI."""
    
    VALID_SPECIALTIES = {
        'Cardiology', 'Oncology', 'Radiology', 'Neurology', 
        'Surgery', 'Psychiatry', 'Endocrinology', 'General Medicine'
    }
    
    def __init__(self, api_key: str):
        self.llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")
    
    def analyze_paper(self, paper: Paper) -> Optional[PaperAnalysis]:
        """
        Analyze a paper using AI to determine its specialty and key concepts.
        
        Args:
            paper: The paper to analyze
            
        Returns:
            PaperAnalysis object if successful, None if analysis fails
        """
        prompt = self._create_analysis_prompt(paper)
        
        try:
            response = self.llm.invoke(prompt)
            time.sleep(2)
            return self._parse_analysis_response(response.content)
        except Exception as e:
            logger.error(f"Error analyzing paper {paper.paper_id}: {str(e)}")
            return None
    
    def _create_analysis_prompt(self, paper: Paper) -> str:
        """Create the prompt for paper analysis."""
        return f"""
        Analyze this medical research paper:
        
        Title: {paper.title}
        Abstract: {paper.abstract[:500]}
        Conclusion: {paper.conclusion}
        Authors: {', '.join(paper.authors[:5])}
        arXiv Categories: {', '.join(paper.categories)}
        
        Provide:
        1. Medical specialty (ONE of: {', '.join(sorted(self.VALID_SPECIALTIES))})
        2. 5 key medical concepts/terms from this research
        3. Main research focus in one sentence
        
        Format your response EXACTLY like this:
        Specialty: [specialty]
        Keywords: [keyword1], [keyword2], [keyword3], [keyword4], [keyword5]
        Focus: [one sentence description]
        """
    
    def _parse_analysis_response(self, response: str) -> Optional[PaperAnalysis]:
        """Parse the AI response into a PaperAnalysis object."""
        lines = response.strip().split('\n')
        specialty = None
        keywords = []
        focus = ""
        
        for line in lines:
            if line.startswith("Specialty:"):
                specialty = line.replace("Specialty:", "").strip()
            elif line.startswith("Keywords:"):
                keywords_text = line.replace("Keywords:", "").strip()
                keywords = [k.strip() for k in keywords_text.split(',')][:5]
            elif line.startswith("Focus:"):
                focus = line.replace("Focus:", "").strip()
        
        if not specialty or specialty not in self.VALID_SPECIALTIES:
            return None
            
        return PaperAnalysis(specialty=specialty, keywords=keywords, focus=focus)

class ResearchDigest:
    """Main class for generating medical research digests."""
    
    def __init__(self, api_key: str):
        self.arxiv_client = ArxivClient()
        self.analyzer = PaperAnalyzer(api_key)
        self.llm = self.analyzer.llm  # Get LLM instance from analyzer
        self.specialty_data: Dict[str, Dict] = {}
    
    def generate_digest(self, search_query: str = "all:medical", max_results: int = 20) -> None:
        """
        Generate a research digest for medical papers.
        
        Args:
            search_query: The search query to use
            max_results: Maximum number of papers to analyze
        """
        logger.info("Fetching papers from arXiv...")
        papers = self.arxiv_client.fetch_papers(search_query, max_results)
        logger.info(f"Found {len(papers)} papers")
        
        self._analyze_papers(papers)
        self._display_summary()
        self._digest_summary()
    
    def _analyze_papers(self, papers: List[Paper]) -> None:
        """Analyze the fetched papers."""
        logger.info("Analyzing papers with AI...")
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"Analyzing paper {i}/{len(papers)}: {paper.title[:80]}...")
            
            analysis = self.analyzer.analyze_paper(paper)
            if not analysis:
                continue
                
            self._update_specialty_data(paper, analysis)
    
    def _update_specialty_data(self, paper: Paper, analysis: PaperAnalysis) -> None:
        """Update the specialty data with the paper analysis."""
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
        """Display the research summary by specialty."""
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
                logger.info(f"     Focus: {paper['focus'][:80]}...")
            
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
        papers_with_summaries = []
        
        for specialty, data in self.specialty_data.items():
            for paper in data["papers"]:
                paper_info = {
                    "title": paper["title"],
                    "specialty": specialty,
                    "focus": paper["focus"],
                    "date": paper["date"],
                    "authors": paper["authors"],
                    "summary": "",
                    "main_discovery": "",
                    "implications": "",
                    "challenges": "",
                    "looking_forward": ""
                }
                
                # Generate analysis for each paper
                try:
                    prompt = f"""
                    Analyze this medical research paper and provide the following information:
                    
                    Title: {paper['title']}
                    Specialty: {specialty}
                    Focus: {paper['focus']}
                    Keywords: {', '.join(paper['keywords'])}
                    
                    Please provide:
                    1. Summary: A comprehensive 5-6 sentence overview of the research
                    2. Main Discovery: The key breakthrough or finding in one sentence
                    3. Implications: The potential impact on medical practice in 1-2 sentences
                    4. Challenges: Main obstacles or limitations identified in one sentence
                    5. Looking Forward: Future directions or next steps in one sentence
                    
                    Format your response exactly as:
                    Summary: [your 5-6 sentence summary]
                    Main Discovery: [key finding]
                    Implications: [medical impact]
                    Challenges: [obstacles]
                    Looking Forward: [future directions]
                    """
                    
                    response = self.llm.invoke(input=prompt)
                    content = response.content.strip()
                    
                    # Parse the response
                    lines = content.split('\n')
                    for line in lines:
                        if line.startswith('Summary:'):
                            paper_info["summary"] = line.replace('Summary:', '').strip()
                        elif line.startswith('Main Discovery:'):
                            paper_info["main_discovery"] = line.replace('Main Discovery:', '').strip()
                        elif line.startswith('Implications:'):
                            paper_info["implications"] = line.replace('Implications:', '').strip()
                        elif line.startswith('Challenges:'):
                            paper_info["challenges"] = line.replace('Challenges:', '').strip()
                        elif line.startswith('Looking Forward:'):
                            paper_info["looking_forward"] = line.replace('Looking Forward:', '').strip()
                    
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate analysis for paper: {str(e)}")
                    # Fallback content
                    paper_info["summary"] = (f"This research focuses on {paper['focus']}. "
                                        f"The study investigates key aspects of {specialty} "
                                        f"with emphasis on {paper['keywords'][0] if paper['keywords'] else 'innovative approaches'}. "
                                        f"The research aims to advance our understanding in this field. "
                                        f"It contributes to the growing body of knowledge in {specialty}. "
                                        f"The findings have potential applications in clinical practice.")
                    paper_info["main_discovery"] = f"Advances in {paper['keywords'][0] if paper['keywords'] else 'medical research'}."
                    paper_info["implications"] = f"Potential to improve {specialty} practices."
                    paper_info["challenges"] = "Further validation and implementation required."
                    paper_info["looking_forward"] = "Future studies needed to expand on these findings."
                
                papers_with_summaries.append(paper_info)
        
        if not papers_with_summaries:
            logger.info("\nNo papers were analyzed. Please ensure papers were successfully fetched and analyzed.")
            return
        
        # Create simplified output structure for newsletter
        output_json = {
            "date_generated": datetime.datetime.now().strftime("%B %d, %Y"),
            "papers": papers_with_summaries,
            "total_papers": len(papers_with_summaries)
        }
        
        logger.info("\n" + "="*60)
        logger.info("JSON DIGEST SUMMARY:")
        logger.info("="*60)
        
        print(json.dumps(output_json, indent=2, ensure_ascii=False))
        
        self.digest_json = output_json

class Newsletter:
    """Generates a newsletter from the research digest."""
    
    def __init__(self, digest: ResearchDigest):
        self.digest = digest
    
    def generate_newsletter(self) -> None:
        """Generate a newsletter from the research digest JSON."""
        if not hasattr(self.digest, 'digest_json'):
            logger.error("No digest JSON available. Run generate_digest() first.")
            return
        
        data = self.digest.digest_json
        
        newsletter_lines = []
        
        # Header
        newsletter_lines.append("="*80)
        newsletter_lines.append("MedDigest Newsletter")
        newsletter_lines.append("="*80)
        newsletter_lines.append(f"\nDate: {data['date_generated']}")
        newsletter_lines.append(f"Total Papers Analyzed: {data['total_papers']}")
        newsletter_lines.append("\n" + "="*80)
        
        # Group papers by specialty
        papers_by_specialty = {}
        for paper in data['papers']:
            specialty = paper['specialty']
            if specialty not in papers_by_specialty:
                papers_by_specialty[specialty] = []
            papers_by_specialty[specialty].append(paper)
        
        # Table of Contents
        newsletter_lines.append("\nTABLE OF CONTENTS:")
        newsletter_lines.append("-"*30)
        for specialty, papers in sorted(papers_by_specialty.items()):
            newsletter_lines.append(f"• {specialty} ({len(papers)} papers)")
        newsletter_lines.append("\n" + "="*80)
        
        # Write each specialty section
        for specialty, papers in sorted(papers_by_specialty.items()):
            newsletter_lines.append(f"\n\n{specialty.upper()}")
            newsletter_lines.append("="*80)
            newsletter_lines.append(f"Number of papers: {len(papers)}\n")
            
            for i, paper in enumerate(papers, 1):
                newsletter_lines.append(f"\nPaper {i}: {paper['title']}")
                newsletter_lines.append("-"*70)
                
                # Paper metadata
                newsletter_lines.append(f"Date: {paper['date']}")
                newsletter_lines.append(f"Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                newsletter_lines.append(f"Focus: {paper['focus']}")
                
                # Main content sections with fallback for empty fields
                if paper.get('summary'):
                    newsletter_lines.append(f"\nSUMMARY:")
                    newsletter_lines.append(paper['summary'])
                
                if paper.get('main_discovery'):
                    newsletter_lines.append(f"\nMAIN DISCOVERY:")
                    newsletter_lines.append(paper['main_discovery'])
                
                if paper.get('implications'):
                    newsletter_lines.append(f"\nIMPLICATIONS:")
                    newsletter_lines.append(paper['implications'])
                
                if paper.get('challenges'):
                    newsletter_lines.append(f"\nCHALLENGES:")
                    newsletter_lines.append(paper['challenges'])
                
                if paper.get('looking_forward'):
                    newsletter_lines.append(f"\nLOOKING FORWARD:")
                    newsletter_lines.append(paper['looking_forward'])
                
                newsletter_lines.append("")  
        
        # Footer
        newsletter_lines.append("\n" + "="*80)
        newsletter_lines.append("END OF MEDICAL RESEARCH DIGEST")
        newsletter_lines.append("="*80)
        newsletter_lines.append(f"\nGenerated on: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        
        # Save to file
        filename = f"newsletter_{datetime.datetime.now().strftime('%m_%d_%Y')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(newsletter_lines))
        
        logger.info(f"\nNewsletter saved to: {filename}")
           

def main():
    """Main entry point for the script."""
    API_KEY = "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"
    
    try:
        digest = ResearchDigest(API_KEY)
        digest.generate_digest()

        # Generate the newsletter
        newsletter = Newsletter(digest)
        newsletter.generate_newsletter()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()