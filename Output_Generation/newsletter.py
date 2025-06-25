from AI_Processing.research_digest import ResearchDigest
import logging
import datetime

logger = logging.getLogger(__name__)


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