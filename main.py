#!/usr/bin/env python3
"""
MedDigest - Main Entry Point

This module serves as the main entry point for the MedDigest application.
It orchestrates the research digest generation and newsletter distribution process.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional
from datetime import datetime
# Third-party imports
from dotenv import load_dotenv

# Local application imports
from AI_Processing.research_digest import ResearchDigest
from Output_Generation.newsletter import Newsletter
from Email_System.send_email import send_bulk_emails
from Firebase.firebase_client import FirebaseClient
from Firebase.firebase_config import FirebaseConfig


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


def load_environment() -> Optional[str]:
    """
    Load environment variables and validate API key.
    
    Returns:
        API key if valid, None otherwise
    """
    load_dotenv()
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set. Please set it in your .env file.")
        return None
    
    return api_key


def generate_research_digest(api_key: str) -> Optional[ResearchDigest]:
    """
    Generate research digest using the provided API key.
    
    Args:
        api_key: The GROQ API key for AI processing
        
    Returns:
        ResearchDigest instance if successful, None otherwise
    """
    try:
        digest = ResearchDigest(api_key)
        digest_json = digest.generate_digest()
        digest.digest_json = digest_json
        
        # Print token usage summary
        digest.token_monitor.print_usage_summary()
        
        return digest
    except Exception as e:
        logger.error(f"Failed to generate research digest: {e}")
        return None


def generate_newsletter(digest: ResearchDigest) -> Optional[str]:
    """
    Generate newsletter from the research digest.
    
    Args:
        digest: The ResearchDigest instance
        
    Returns:
        Newsletter content if successful, None otherwise
    """
    try:
        newsletter = Newsletter(digest)
        return newsletter.generate_newsletter()
    except Exception as e:
        logger.error(f"Failed to generate newsletter: {e}")
        return None
    
def get_all_user_subscriptions(firebase_client: FirebaseClient) -> Optional[Dict[str, Any]]:
    """
    Get all user subscriptions from Firestore.
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
        result = send_bulk_emails(recipients, subject, newsletter_content)
        
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


def main() -> None:
    """
    Main entry point for the MedDigest application.
    
    Orchestrates the complete workflow:
    1. Environment setup and validation
    2. Research digest generation
    3. Newsletter creation
    4. Email distribution
    """
    global logger
    logger = setup_logging()
    
    logger.info("🚀 Starting MedDigest application...")
    
    # Step 1: Load and validate environment
    api_key = load_environment()
    if not api_key:
        logger.error("❌ Environment setup failed. Exiting.")
        return
    
    # Step 2: Initialize Firebase client
    try:
        firebase_config = FirebaseConfig.from_env()
        firebase_client = FirebaseClient(firebase_config)
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")
        return
    
    # Step 3: Generate research digest
    digest = generate_research_digest(api_key)
    if not digest:
        logger.error("❌ Research digest generation failed. Exiting.")
        return
    
    # Step 4: Generate newsletter
    newsletter_content = generate_newsletter(digest)
    if not newsletter_content:
        logger.error("❌ Newsletter generation failed. Exiting.")
        return
    
    # Step 5: Get all user subscriptions
    user_subscriptions = get_all_user_subscriptions(firebase_client)
    if user_subscriptions is None:
        logger.error("❌ Failed to get user subscriptions. Exiting.")
        return
    
    # Step 6: Send newsletter email
    send_newsletter_email_safely(newsletter_content, user_subscriptions)
    
    logger.info("🎉 MedDigest application completed successfully!")


if __name__ == "__main__":
    main()
