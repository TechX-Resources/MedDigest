import base64
import pickle
from email.message import EmailMessage
from pathlib import Path
import logging

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)
SCOPES = ['https://www.googleapis.com/auth/gmail.send']


def get_gmail_service():
    """Get Gmail service with OAuth2 authentication."""
    creds = None
    token_path = Path(__file__).parent / 'token.pickle'
    credentials_path = Path(__file__).parent / 'credentials.json'
    
    logger.info(f"Looking for credentials at: {credentials_path}")
    logger.info(f"Looking for token at: {token_path}")
    
    if token_path.exists():
        try:
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
            logger.info("Found existing token file")
        except Exception as e:
            logger.error(f"Error loading token file: {e}")
            creds = None
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials...")
            try:
                creds.refresh(Request())
                logger.info("Credentials refreshed successfully")
            except Exception as e:
                logger.error(f"Error refreshing credentials: {e}")
                creds = None
        else:
            if not credentials_path.exists():
                error_msg = (
                    f"credentials.json not found at {credentials_path}. "
                    "Please download OAuth2 credentials from Google Cloud Console."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info("Starting OAuth2 flow...")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES)
                creds = flow.run_local_server(
                    port=8080,
                    open_browser=True
                )
                logger.info("OAuth2 authentication completed successfully")
            except Exception as e:
                logger.error(f"Error during OAuth2 flow: {e}")
                raise
        
        try:
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
            logger.info("Token saved successfully")
        except Exception as e:
            logger.error(f"Error saving token: {e}")
    
    try:
        service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail service built successfully")
        return service
    except Exception as e:
        logger.error(f"Error building Gmail service: {e}")
        raise


def gmail_send_message(to_email, subject, body):
    """Create and send an email message.
    
    Args:
        to_email (str): Recipient email address
        subject (str): Email subject line
        body (str): Email body content
    
    Returns:
        dict: Message object, including message id, or None if error
    """
    try:
        logger.info(f"Attempting to send email to: {to_email}")
        logger.info(f"Subject: {subject}")
        logger.info(f"Body length: {len(body)} characters")
        
        # Clean the body content to ensure it's safe for email
        if body is None:
            logger.error("Body content is None")
            return None
            
        # Ensure body is a string
        body = str(body)
        
        # Remove any problematic characters that might cause encoding issues
        body = body.replace('\x00', '')  # Remove null bytes
        body = body.replace('\r\n', '\n')  # Normalize line endings
        body = body.replace('\r', '\n')  # Normalize line endings
        
        service = get_gmail_service()
        if service is None:
            logger.error("Failed to get Gmail service")
            return None
            
        message = EmailMessage()

        message.set_content(body)
        message["To"] = to_email
        message["From"] = "meddigest.newsletter@gmail.com"
        message["Subject"] = subject

        logger.info("Encoding message...")
        try:
            message_bytes = message.as_bytes()
            logger.info(f"Message bytes length: {len(message_bytes)}")
            
            # Check if message is too large (Gmail has limits)
            if len(message_bytes) > 25 * 1024 * 1024:  # 25MB limit
                logger.error(f"Message too large: {len(message_bytes)} bytes")
                return None
                
            encoded_message = base64.urlsafe_b64encode(message_bytes).decode()
            logger.info(f"Encoded message length: {len(encoded_message)}")
        except Exception as e:
            logger.error(f"Error encoding message: {e}")
            return None

        create_message = {"raw": encoded_message}
        
        logger.info("Sending message via Gmail API...")
        try:
            send_message = (
                service.users()
                .messages()
                .send(userId="me", body=create_message)
                .execute()
            )
            logger.info(f'Message sent successfully! Message Id: {send_message["id"]}')
            return send_message
        except Exception as e:
            logger.error(f"Error in Gmail API call: {e}")
            logger.error(f"Error type: {type(e)}")
            return None
            
    except HttpError as error:
        logger.error(f"Gmail API HTTP error: {error}")
        logger.error(f"HTTP error details: {error.resp.status} {error.content}")
        return None
    except Exception as error:
        logger.error(f"Unexpected error sending email: {error}")
        logger.error(f"Error type: {type(error)}")
        logger.error(f"Error details: {str(error)}")
        return None
