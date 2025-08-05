"""
Firebase configuration for MedDigest.

This module handles Firebase Admin SDK configuration for server-side operations.
"""

import os
import json
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FirebaseConfig:
    """
    Firebase Admin SDK configuration.
    
    Attributes:
        project_id (str): Firebase project ID
        service_account_path (Optional[str]): Path to service account JSON file
        service_account_json (Optional[str]): Service account JSON as string (for Vercel)
    """
    project_id: str
    service_account_path: Optional[str] = None
    service_account_json: Optional[str] = None
        
    @classmethod
    def from_env(cls) -> 'FirebaseConfig':
        """
        Create Firebase configuration from environment variables.
        
        Returns:
            FirebaseConfig: Configuration object with values from environment
            
        Raises:
            ValueError: If required environment variables are missing
        """
        project_id = os.getenv('FIREBASE_PROJECT_ID')
        if not project_id:
            raise ValueError("FIREBASE_PROJECT_ID environment variable is required")
        
        service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
        service_account_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
        
        return cls(
            project_id=project_id,
            service_account_path=service_account_path,
            service_account_json=service_account_json
        )
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        return bool(self.project_id and isinstance(self.project_id, str))
    
    def get_service_account_dict(self) -> Optional[dict]:
        """
        Get service account credentials as dictionary.
        
        Returns:
            Optional[dict]: Service account credentials or None
        """
        if self.service_account_json:
            try:
                return json.loads(self.service_account_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse service account JSON: {e}")
                return None
        
        if self.service_account_path and os.path.exists(self.service_account_path):
            try:
                with open(self.service_account_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read service account file: {e}")
                return None
        
        return None 