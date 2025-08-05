from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
import os
import json

load_dotenv()

from Firebase.firebase_config import FirebaseConfig
from Firebase.firebase_client import FirebaseClient

app = FastAPI()

# Updated CORS settings to allow the deployed frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://med-digest-osln.vercel.app",
        "https://med-digest-osln.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

firebase_client = None
try:
    firebase_config = FirebaseConfig.from_env()
    firebase_client = FirebaseClient(firebase_config)
    print("Firebase client initialized successfully")
except Exception as e:
    print(f"Firebase not available: {str(e)}")
    firebase_client = None

class UserSignup(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    medical_interests: list[str]

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "MedDigest API"}

@app.get("/api/newsletter")
def get_newsletter():
    if not firebase_client:
        return {"error": "Firebase service unavailable"}
    
    try:
        digest_data = firebase_client.get_latest_digest()
        if not digest_data:
            return {"error": "No newsletter data found"}
        
        response_data = {
            'id': digest_data.get('id'),
            'date_generated': digest_data.get('date_generated'),
            'total_papers': digest_data.get('total_papers'),
            'specialty_data': {}
        }
        
        if 'specialty_data' in digest_data:
            for specialty, spec_info in digest_data['specialty_data'].items():
                if 'papers' in spec_info:
                    response_data['specialty_data'][specialty] = {'papers': spec_info['papers']}
        
        if 'digest_summary' in digest_data and digest_data['digest_summary']:
            summary_data = digest_data['digest_summary']
            response_data.update(summary_data)
        
        elif 'batch_analyses' in digest_data and digest_data['batch_analyses']:
            batch_key = list(digest_data['batch_analyses'].keys())[0]
            batch_analysis = digest_data['batch_analyses'][batch_key]
            
            if 'analysis' in batch_analysis:
                analysis = batch_analysis['analysis']
                
                response_data['executive_summary'] = analysis.get('batch_summary', '')
                response_data['key_discoveries'] = analysis.get('significant_findings', [])
                # Preserve line breaks in emerging trends instead of joining with periods
                emerging_trends = analysis.get('major_trends', [])
                if isinstance(emerging_trends, list):
                    response_data['emerging_trends'] = '\n\n'.join(emerging_trends)
                else:
                    response_data['emerging_trends'] = emerging_trends
                response_data['cross_specialty_insights'] = analysis.get('cross_specialty_insights', '')
                response_data['clinical_implications'] = analysis.get('medical_impact', '')
                response_data['research_gaps'] = analysis.get('research_gaps', '')
                response_data['future_directions'] = analysis.get('future_directions', '')
        
        elif 'digest_summary' in digest_data and digest_data['digest_summary']:
            summary_data = digest_data['digest_summary']
            response_data.update(summary_data)
        
        return response_data
    except Exception as e:
        return {"error": f"Failed to fetch newsletter data: {str(e)}"}

@app.post("/api/signup")
def simple_signup(signup: UserSignup):
    if not firebase_client:
        return {"success": False, "message": "Service unavailable"}
    
    try:
        success = firebase_client.store_user_signup(
            email=signup.email,
            first_name=signup.first_name,
            last_name=signup.last_name,
            medical_interests=signup.medical_interests
        )
        
        if success:
            return {"success": True, "message": f"Thanks {signup.first_name}! You're signed up."}
        else:
            return {"success": False, "message": "Signup failed. Please try again."}
    except Exception as e:
        return {"success": False, "message": f"Signup failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 