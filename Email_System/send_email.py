#!/usr/bin/env python3
"""
Email Sender Script
Uses the existing email configuration to send emails via Gmail API.
"""

from email_config import gmail_send_message


def send_newsletter_email(to_email, subject, body):
    """
    Send a newsletter email using the Gmail API.
    
    Args:
        to_email (str): Recipient email address
        subject (str): Email subject line
        body (str): Email body content
    
    Returns:
        dict: Response from Gmail API or None if error
    """
    try:
        result = gmail_send_message(to_email, subject, body)
        if result:
            print(f"✅ Email sent successfully to {to_email}")
            return result
        else:
            print(f"❌ Failed to send email to {to_email}")
            return None
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return None


def send_bulk_emails(recipients, subject, body):
    """
    Send emails to multiple recipients.
    
    Args:
        recipients (list): List of email addresses
        subject (str): Email subject line
        body (str): Email body content
    
    Returns:
        dict: Summary of results
    """
    results = {
        'successful': [],
        'failed': []
    }
    
    for email in recipients:
        result = send_newsletter_email(email, subject, body)
        if result:
            results['successful'].append(email)
        else:
            results['failed'].append(email)
    
    print(f"\n📊 Email Summary:")
    print(f"✅ Successful: {len(results['successful'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    
    return results 