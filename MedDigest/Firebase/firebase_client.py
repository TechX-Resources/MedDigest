"""
Firebase client for MedDigest.

This module provides Firebase Firestore operations for storing research digests.
"""

import logging
from typing import Dict, Any, Optional, List
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

from .firebase_config import FirebaseConfig

logger = logging.getLogger(__name__)


class FirebaseClient:
    """
    Firebase client for storing research digests in Firestore.
    """
    
    def __init__(self, config: FirebaseConfig):
        """
        Initialize Firebase connection.
        
        Args:
            config (FirebaseConfig): Firebase configuration
            
        Raises:
            ValueError: If configuration is invalid
            FirebaseError: If Firebase initialization fails
        """
        if not config.validate():
            raise ValueError("Invalid Firebase configuration")
        
        self.config = config
        self.db = None
        self._initialize_firebase()
    
    def _initialize_firebase(self) -> None:
        """
        Initialize Firebase Admin SDK and Firestore connection.
        
        Raises:
            FirebaseError: If Firebase initialization fails
        """
        try:
            # Check if Firebase app is already initialized
            if not firebase_admin._apps:
                if self.config.service_account_path:
                    # Use service account credentials
                    cred = credentials.Certificate(self.config.service_account_path)
                    firebase_admin.initialize_app(cred, {
                        'projectId': self.config.project_id
                    })
                else:
                    # Use default credentials (for development/testing)
                    firebase_admin.initialize_app()
            
            self.db = firestore.client()
            logger.info(f"Connected to Firebase project: {self.config.project_id}")
            
        except FirebaseError as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing Firebase: {str(e)}")
            raise
    
    def store_digest(self, digest_data: Dict[str, Any], digest_id: str) -> bool:
        """
        Store a research digest in Firestore.
        
        Args:
            digest_data (Dict[str, Any]): Digest data as dictionary
            digest_id (str): Unique ID for the digest
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            digest_ref = self.db.collection('research_digests').document(digest_id)
            digest_ref.set(digest_data)
            logger.info(f"Stored research digest: {digest_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store research digest: {str(e)}")
            return False
        
    def store_user_subscription(self, user_id: str, subscription_data: Dict[str, Any]) -> bool:
        """
        Store a user subscription in Firestore.
        """
        try:
            subscription_ref = self.db.collection('user_subscriptions').document(user_id)
            subscription_ref.set(subscription_data)
            logger.info(f"Stored user subscription: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store user subscription: {str(e)}")
            return False
        
    def store_paper_analysis(self, paper_id: str, analysis: Dict[str, Any]) -> bool:
        """
        Store a paper analysis in Firestore.
        
        Args:
            paper_id (str): Unique identifier for the paper
            analysis (Dict[str, Any]): Analysis data to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Store in paper_analyses collection with paper_id as document ID
            analysis_ref = self.db.collection('paper_analyses').document(paper_id)
            
            # Add timestamp for when the analysis was stored
            analysis['stored_at'] = firestore.SERVER_TIMESTAMP
            
            analysis_ref.set(analysis)
            logger.info(f"Stored paper analysis for paper ID: {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store paper analysis for paper ID {paper_id}: {str(e)}")
            return False
    
    def get_paper_analysis(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a paper analysis from Firestore.
        
        Args:
            paper_id (str): ID of the paper to retrieve analysis for
            
        Returns:
            Optional[Dict[str, Any]]: Analysis data if found, None otherwise
        """
        try:
            analysis_ref = self.db.collection('paper_analyses').document(paper_id)
            analysis_doc = analysis_ref.get()
            
            if analysis_doc.exists:
                return analysis_doc.to_dict()
            else:
                logger.warning(f"Paper analysis not found: {paper_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve paper analysis {paper_id}: {str(e)}")
            return None
    
    def get_analyses_by_specialty(self, specialty: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve paper analyses filtered by specialty.
        
        Args:
            specialty (str): Medical specialty to filter by
            limit (int): Maximum number of analyses to return
            
        Returns:
            List[Dict[str, Any]]: List of analysis data
        """
        try:
            analyses_ref = self.db.collection('paper_analyses')
            query = analyses_ref.where('specialty', '==', specialty).limit(limit)
            docs = query.stream()
            
            analyses = []
            for doc in docs:
                analysis_data = doc.to_dict()
                analysis_data['paper_id'] = doc.id
                analyses.append(analysis_data)
            
            logger.info(f"Retrieved {len(analyses)} analyses for specialty: {specialty}")
            return analyses
            
        except Exception as e:
            logger.error(f"Failed to retrieve analyses for specialty {specialty}: {str(e)}")
            return []
    
    def get_high_interest_analyses(self, min_score: float = 7.0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve paper analyses with high interest scores.
        
        Args:
            min_score (float): Minimum interest score threshold
            limit (int): Maximum number of analyses to return
            
        Returns:
            List[Dict[str, Any]]: List of high-interest analysis data
        """
        try:
            analyses_ref = self.db.collection('paper_analyses')
            query = analyses_ref.where('interest_score', '>=', min_score).order_by('interest_score', direction=firestore.Query.DESCENDING).limit(limit)
            docs = query.stream()
            
            analyses = []
            for doc in docs:
                analysis_data = doc.to_dict()
                analysis_data['paper_id'] = doc.id
                analyses.append(analysis_data)
            
            logger.info(f"Retrieved {len(analyses)} high-interest analyses (score >= {min_score})")
            return analyses
            
        except Exception as e:
            logger.error(f"Failed to retrieve high-interest analyses: {str(e)}")
            return []
    
    def get_digest(self, digest_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a research digest from Firestore.
        
        Args:
            digest_id (str): ID of the digest to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Digest data if found, None otherwise
        """
        try:
            digest_ref = self.db.collection('research_digests').document(digest_id)
            digest_doc = digest_ref.get()
            
            if digest_doc.exists:
                return digest_doc.to_dict()
            else:
                logger.warning(f"Digest not found: {digest_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve digest {digest_id}: {str(e)}")
            return None
    
    def get_user_subscription(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user subscription from Firestore.
        """
        try:
            subscription_ref = self.db.collection('user_subscriptions').document(user_id)
            subscription_doc = subscription_ref.get()
            
            if subscription_doc.exists:
                return subscription_doc.to_dict()
            else:
                logger.warning(f"User subscription not found: {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve user subscription {user_id}: {str(e)}")
            return None