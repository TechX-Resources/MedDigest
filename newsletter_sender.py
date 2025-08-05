#!/usr/bin/env python3
"""
MedDigest - Newsletter Sender

This module handles the email distribution of newsletters to subscribers.
It can be run independently to send newsletters without regenerating content.
If no newsletter content is provided, it will fetch the latest from the database.
"""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Local application imports
from Email_System.send_email import send_bulk_emails
from Firebase.firebase_client import FirebaseClient
from Firebase.firebase_config import FirebaseConfig
from Output_Generation.newsletter_markdown import NewsletterMarkdown


def setup_logging() -> logging.Logger:
    """Configure and return a logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def initialize_firebase() -> Optional[FirebaseClient]:
    """
    Initialize Firebase client for accessing user subscriptions.
    
    Returns:
        FirebaseClient instance if successful, None otherwise
    """
    try:
        firebase_config = FirebaseConfig.from_env()
        firebase_client = FirebaseClient(firebase_config)
        return firebase_client
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")
        return None


def get_latest_newsletter_from_db(firebase_client: FirebaseClient) -> Optional[str]:
    """
    Fetch the latest newsletter data from the database and format it.
    
    Args:
        firebase_client: The Firebase client instance
        
    Returns:
        Formatted newsletter content if successful, None otherwise
    """
    try:
        # Get the latest digest from the database
        latest_digest = firebase_client.get_latest_digest()
        
        if not latest_digest:
            logger.warning("⚠️ No newsletter data found in the database.")
            return None
        
        logger.info(f"📊 Found latest newsletter from {latest_digest.get('date_generated', 'Unknown date')}")
        
        # Transform database data into the format expected by Newsletter class
        formatted_data = {
            'id': latest_digest.get('id'),
            'date_generated': latest_digest.get('date_generated'),
            'total_papers': latest_digest.get('total_papers'),
            'specialty_data': {}
        }
        
        # Process specialty data
        if 'specialty_data' in latest_digest:
            for specialty, spec_info in latest_digest['specialty_data'].items():
                if 'papers' in spec_info:
                    formatted_data['specialty_data'][specialty] = {'papers': spec_info['papers']}
        
        # Process batch analyses (new format)
        if 'batch_analyses' in latest_digest and latest_digest['batch_analyses']:
            batch_key = list(latest_digest['batch_analyses'].keys())[0]
            batch_analysis = latest_digest['batch_analyses'][batch_key]
            
            if 'analysis' in batch_analysis:
                analysis = batch_analysis['analysis']
                
                formatted_data['executive_summary'] = analysis.get('batch_summary', '')
                formatted_data['key_discoveries'] = analysis.get('significant_findings', [])
                formatted_data['emerging_trends'] = '. '.join(analysis.get('major_trends', [])) + '.'
                formatted_data['cross_specialty_insights'] = analysis.get('cross_specialty_insights', '')
                formatted_data['clinical_implications'] = analysis.get('medical_impact', '')
                formatted_data['research_gaps'] = analysis.get('research_gaps', '')
                formatted_data['future_directions'] = analysis.get('future_directions', '')
        
        # Process digest summary (fallback format)
        elif 'digest_summary' in latest_digest and latest_digest['digest_summary']:
            summary_data = latest_digest['digest_summary']
            formatted_data.update(summary_data)
        
        # Create a mock ResearchDigest object to use with Newsletter class
        class MockDigest:
            def __init__(self, digest_data):
                self.digest_json = digest_data
                # Add specialty_data attribute for Newsletter class
                self.specialty_data = digest_data.get('specialty_data', {})
        
        mock_digest = MockDigest(formatted_data)
        
        # Generate newsletter using the existing Newsletter class
        newsletter = NewsletterMarkdown(mock_digest)
        newsletter_content = newsletter.generate_newsletter()
        
        if newsletter_content:
            logger.info("✅ Successfully formatted newsletter from database")
            return newsletter_content
        else:
            logger.error("❌ Failed to format newsletter from database data")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to fetch newsletter from database: {e}")
        return None


def get_all_user_subscriptions(firebase_client: FirebaseClient) -> Optional[Dict[str, Any]]:
    """
    Get all user subscriptions from Firestore.
    
    Args:
        firebase_client: The Firebase client instance
        
    Returns:
        User subscriptions dictionary if successful, None otherwise
    """
    try:
        return firebase_client.get_all_user_subscriptions()
    except Exception as e:
        logger.error(f"Failed to get all user subscriptions: {e}")
        return None


def send_newsletter_email_safely(newsletter_content: str, user_subscriptions: Dict[str, Any]) -> bool:
    """
    Safely attempt to send the newsletter email.
    
    Args:
        newsletter_content: The newsletter content to send
        user_subscriptions: The user subscriptions to send the newsletter to
        
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Convert user subscriptions to the format expected by send_bulk_emails
        recipients = []
        for subscription_data in user_subscriptions.values():
            if 'email' in subscription_data:
                recipients.append({'email': subscription_data['email']})
        
        if not recipients:
            logger.warning("⚠️ No user subscriptions found with email addresses.")
            logger.info("📧 Newsletter was generated and saved locally. You can find it in the Newsletters directory.")
            return False
        
        newsletter_date = datetime.now().strftime("%Y-%m-%d")
        subject = f"MedDigest Newsletter - {newsletter_date}"
        result = send_bulk_emails(recipients, subject, newsletter_content, is_markdown=True)
        
        if result and result.get('successful'):
            logger.info("✅ Newsletter email sent successfully!")
            return True
        else:
            logger.warning("⚠️ Email sending failed, but newsletter was generated and saved locally.")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Email sending failed with error: {e}")
        logger.info("📧 Newsletter was generated and saved locally. You can find it in the Newsletters directory.")
        return False


def send_newsletter_workflow(newsletter_content: Optional[str] = None) -> bool:
    """
    Complete workflow for sending a newsletter.
    
    Args:
        newsletter_content: The newsletter content to send. If None, will fetch from database.
        
    Returns:
        True if email sent successfully, False otherwise
    """
    logger.info("📧 Starting newsletter sending workflow...")
    
    # Step 1: Initialize Firebase
    firebase_client = initialize_firebase()
    if not firebase_client:
        logger.error("❌ Firebase initialization failed.")
        return False
    
    # Step 2: Get newsletter content (from parameter or database)
    if not newsletter_content:
        logger.info("📊 No newsletter content provided. Fetching latest from database...")
        newsletter_content = get_latest_newsletter_from_db(firebase_client)
        
        if not newsletter_content:
            logger.error("❌ Failed to get newsletter content from database.")
            return False
    else:
        logger.info("📝 Using provided newsletter content.")
    
    # Step 3: Get user subscriptions
    user_subscriptions = get_all_user_subscriptions(firebase_client)
    if user_subscriptions is None:
        logger.error("❌ Failed to get user subscriptions.")
        return False
    
    # Step 4: Send newsletter email
    success = send_newsletter_email_safely(newsletter_content, user_subscriptions)
    
    if success:
        logger.info("✅ Newsletter sending workflow completed successfully!")
    else:
        logger.warning("⚠️ Newsletter sending workflow completed with warnings.")
    
    return success


def main() -> None:
    """
    Main entry point for newsletter sending.
    
    This can be run independently to send newsletters without regenerating content.
    If no newsletter content is provided, it will automatically fetch the latest from the database.
    """
    global logger
    logger = setup_logging()
    
    logger.info("🚀 Starting MedDigest Newsletter Sender...")
    logger.info("📧 Newsletter sender will fetch latest content from database if not provided.")
    
    # Run the workflow without providing content - it will fetch from database
    success = send_newsletter_workflow()
    
    if success:
        logger.info("🎉 Newsletter sending completed successfully!")
    else:
        logger.error("❌ Newsletter sending failed.")


if __name__ == "__main__":
    main() 