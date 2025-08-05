#!/usr/bin/env python3
"""
Test script to verify that research gaps and future directions are being generated correctly.
"""

import os
from dotenv import load_dotenv
from AI_Processing.research_digest import ResearchDigest

def test_digest_generation():
    """Test the digest generation to ensure research gaps and future directions are included."""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        return
    
    print("Testing digest generation...")
    
    # Create research digest instance
    digest = ResearchDigest(api_key)
    
    # Generate digest with a small search query to test
    print("Generating digest...")
    digest_json = digest.generate_digest("all:medical AND (cancer OR oncology) AND 2024")
    
    # Check if research gaps and future directions are present
    print("\n" + "="*50)
    print("CHECKING DIGEST CONTENT")
    print("="*50)
    
    if 'research_gaps' in digest_json:
        print(f"✓ Research gaps found: {len(digest_json['research_gaps'])} characters")
        print(f"  Preview: {digest_json['research_gaps'][:100]}...")
    else:
        print("✗ Research gaps NOT found in digest")
    
    if 'future_directions' in digest_json:
        print(f"✓ Future directions found: {len(digest_json['future_directions'])} characters")
        print(f"  Preview: {digest_json['future_directions'][:100]}...")
    else:
        print("✗ Future directions NOT found in digest")
    
    # Print all available keys
    print(f"\nAll keys in digest: {list(digest_json.keys())}")
    
    # Test storing in Firebase
    if digest.firebase_available:
        print("\n" + "="*50)
        print("TESTING FIREBASE STORAGE")
        print("="*50)
        
        try:
            digest_data = digest.to_dict()
            print(f"Digest data keys: {list(digest_data.keys())}")
            
            if 'digest_summary' in digest_data:
                summary_keys = list(digest_data['digest_summary'].keys())
                print(f"Digest summary keys: {summary_keys}")
                
                if 'research_gaps' in digest_data['digest_summary']:
                    print("✓ Research gaps found in Firebase digest_summary")
                else:
                    print("✗ Research gaps NOT found in Firebase digest_summary")
                    
                if 'future_directions' in digest_data['digest_summary']:
                    print("✓ Future directions found in Firebase digest_summary")
                else:
                    print("✗ Future directions NOT found in Firebase digest_summary")
            
            success = digest.firebase_client.store_digest(digest_data, digest.id)
            if success:
                print("✓ Digest stored successfully in Firebase")
            else:
                print("✗ Failed to store digest in Firebase")
                
        except Exception as e:
            print(f"✗ Error storing digest: {str(e)}")
    else:
        print("Firebase not available, skipping storage test")
    
    print("\n" + "="*50)
    print("TEST COMPLETE")
    print("="*50)

if __name__ == "__main__":
    test_digest_generation() 