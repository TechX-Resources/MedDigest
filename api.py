from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

load_dotenv()

from Firebase.firebase_config import FirebaseConfig
from Firebase.firebase_client import FirebaseClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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

@app.get("/api/newsletter")
def get_newsletter():
    if not firebase_client:
        return {"error": "Firebase service unavailable"}
    
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
    
    if 'batch_analyses' in digest_data and digest_data['batch_analyses']:
        batch_key = list(digest_data['batch_analyses'].keys())[0]
        batch_analysis = digest_data['batch_analyses'][batch_key]
        
        if 'analysis' in batch_analysis:
            analysis = batch_analysis['analysis']
            
            response_data['executive_summary'] = analysis.get('batch_summary', '')
            response_data['key_discoveries'] = analysis.get('significant_findings', [])
            response_data['emerging_trends'] = '. '.join(analysis.get('major_trends', [])) + '.'
            response_data['cross_specialty_insights'] = analysis.get('cross_specialty_insights', '')
            response_data['clinical_implications'] = analysis.get('medical_impact', '')
            response_data['research_gaps'] = 'Research gaps analysis is being processed.'
            response_data['future_directions'] = 'Future directions analysis is being processed.'
    
    elif 'digest_summary' in digest_data and digest_data['digest_summary']:
        summary_data = digest_data['digest_summary']
        response_data.update(summary_data)
    
    return response_data

@app.post("/api/signup")
def simple_signup(signup: UserSignup):
    if not firebase_client:
        return {"success": False, "message": "Service unavailable"}
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000) 