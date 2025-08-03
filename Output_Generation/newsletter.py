from AI_Processing.research_digest import ResearchDigest
import logging
import datetime

logger = logging.getLogger(__name__)


class Newsletter:
    """Generates a newsletter from the research digest."""

    def __init__(self, digest: ResearchDigest):
        self.digest = digest
        self.newsletter_content = None

    def generate_newsletter(self) -> str:
        """Generate a newsletter from the research digest JSON."""
        if not hasattr(self.digest, 'digest_json'):
            logger.error("No digest JSON available. Run generate_digest() first.")
            return None

        data = self.digest.digest_json

        newsletter_lines = []

        # Professional Header
        newsletter_lines.append("="*80)
        newsletter_lines.append(" "*20 + "MedDigest Weekly Research Newsletter")
        newsletter_lines.append("="*80)
        newsletter_lines.append(" "*18 + "Curated Insights from the Latest Medical Literature")
        newsletter_lines.append("="*80)
        newsletter_lines.append(f"\nDate: {data.get('date_generated', 'Unknown')}   |   Total Papers Analyzed: {data.get('total_papers', 0)}")
        newsletter_lines.append("="*80)

        # Executive Summary
        newsletter_lines.append("\nEXECUTIVE SUMMARY")
        newsletter_lines.append("-"*80)
        newsletter_lines.append(data.get('executive_summary', 'No executive summary available.'))
        newsletter_lines.append("\n" + "="*80)

        # Key Discoveries
        newsletter_lines.append("\nKEY DISCOVERIES")
        newsletter_lines.append("-"*80)
        key_discoveries = data.get('key_discoveries', [])
        if key_discoveries:
            for i, discovery in enumerate(key_discoveries, 1):
                newsletter_lines.append(f"{i}. {discovery}")
        else:
            newsletter_lines.append("No key discoveries available.")
        newsletter_lines.append("\n" + "="*80)

        # Emerging Trends
        newsletter_lines.append("\nEMERGING TRENDS")
        newsletter_lines.append("-"*80)
        emerging_trends = data.get('emerging_trends', '')
        if emerging_trends:
            if isinstance(emerging_trends, list):
                for i, trend in enumerate(emerging_trends, 1):
                    newsletter_lines.append(f"{i}. {trend}")
            else:
                newsletter_lines.append(emerging_trends)
        else:
            newsletter_lines.append("No emerging trends available.")
        newsletter_lines.append("\n" + "="*80)

        # Cross-Specialty Insights
        newsletter_lines.append("\nCROSS-SPECIALTY INSIGHTS")
        newsletter_lines.append("-"*80)
        cross_specialty_insights = data.get('cross_specialty_insights', [])
        if cross_specialty_insights:
            if isinstance(cross_specialty_insights, list):
                for i, insight in enumerate(cross_specialty_insights, 1):
                    newsletter_lines.append(f"{i}. {insight}")
            else:
                newsletter_lines.append(cross_specialty_insights)
        else:
            newsletter_lines.append("No cross-specialty insights available.")
        newsletter_lines.append("\n" + "="*80)

        # Clinical Implications
        newsletter_lines.append("\nCLINICAL IMPLICATIONS")
        newsletter_lines.append("-"*80)
        clinical_implications = data.get('clinical_implications', '')
        if clinical_implications:
            if isinstance(clinical_implications, list):
                for i, implication in enumerate(clinical_implications, 1):
                    newsletter_lines.append(f"{i}. {implication}")
            else:
                newsletter_lines.append(clinical_implications)
        else:
            newsletter_lines.append("No clinical implications available.")
        newsletter_lines.append("\n" + "="*80)

        # Research Gaps
        newsletter_lines.append("\nRESEARCH GAPS")
        newsletter_lines.append("-"*80)
        research_gaps = data.get('research_gaps', '')
        if research_gaps:
            if isinstance(research_gaps, list):
                for i, gap in enumerate(research_gaps, 1):
                    newsletter_lines.append(f"{i}. {gap}")
            else:
                newsletter_lines.append(research_gaps)
        else:
            newsletter_lines.append("No research gaps available.")
        newsletter_lines.append("\n" + "="*80)

        # Future Directions
        newsletter_lines.append("\nFUTURE DIRECTIONS")
        newsletter_lines.append("-"*80)
        future_directions = data.get('future_directions', '')
        if future_directions:
            if isinstance(future_directions, list):
                for i, direction in enumerate(future_directions, 1):
                    newsletter_lines.append(f"{i}. {direction}")
            else:
                newsletter_lines.append(future_directions)
        else:
            newsletter_lines.append("No future directions available.")
        newsletter_lines.append("\n" + "="*80)

        # Group papers by specialty from specialty_data
        papers_by_specialty = {}
        if hasattr(self.digest, 'specialty_data'):
            for specialty, specialty_info in self.digest.specialty_data.items():
                papers_by_specialty[specialty] = specialty_info.get('papers', [])
        else:
            # Fallback: try to get papers from digest JSON if available
            papers = data.get('papers', [])
            if papers:
                for paper in papers:
                    specialty = paper.get('specialty', 'Unknown')
                    if specialty not in papers_by_specialty:
                        papers_by_specialty[specialty] = []
                    papers_by_specialty[specialty].append(paper)

        # Table of Contents
        newsletter_lines.append("\nTABLE OF CONTENTS")
        newsletter_lines.append("-"*80)
        if papers_by_specialty:
            for specialty, papers in sorted(papers_by_specialty.items()):
                newsletter_lines.append(f"• {specialty} ({len(papers)} papers)")
        else:
            newsletter_lines.append("• No papers available for this digest")
        newsletter_lines.append("="*80)

        # Specialty Sections
        if papers_by_specialty:
            for specialty, papers in sorted(papers_by_specialty.items()):
                newsletter_lines.append(f"\n{specialty.upper()}")
                newsletter_lines.append("-"*80)
                newsletter_lines.append(f"Number of papers this week: {len(papers)}")
                newsletter_lines.append(" ")
                newsletter_lines.append("(See full paper details in the online supplement)")
                newsletter_lines.append("="*80)
        else:
            newsletter_lines.append("\nPAPER ANALYSIS")
            newsletter_lines.append("-"*80)
            newsletter_lines.append("No papers available for detailed analysis.")
            newsletter_lines.append("="*80)

        # Footer
        newsletter_lines.append("\n" + "="*80)
        newsletter_lines.append("Thank you for reading MedDigest! For more information, visit https://meddigest.example.com or contact us at info@meddigest.example.com")
        newsletter_lines.append("© " + datetime.datetime.now().strftime('%Y') + " MedDigest. All rights reserved.")
        newsletter_lines.append("="*80)
        newsletter_lines.append(f"\nGenerated on: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}")

        # Store the newsletter content for email sending
        self.newsletter_content = '\n'.join(newsletter_lines)

        # Create Newsletters directory if it doesn't exist
        import os
        os.makedirs("Newsletters", exist_ok=True)

        # Save to file
        filename = f"Newsletters/newsletter_{datetime.datetime.now().strftime('%m_%d_%Y')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.newsletter_content)

        logger.info(f"\nNewsletter saved to: {filename}")
        
        # Return the newsletter content for email sending
        return self.newsletter_content