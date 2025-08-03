# Firebase Setup Guide for MedDigest

This guide will help you set up Firebase integration for storing and retrieving research digests.

## Prerequisites

1. **Firebase Project**: You need a Firebase project with Firestore database enabled
2. **Service Account Key**: A service account JSON file for server-side authentication

## Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project" or select an existing project
3. Enable Firestore Database:
   - Go to Firestore Database in the left sidebar
   - Click "Create database"
   - Choose "Start in test mode" (for development)
   - Select a location for your database

## Step 2: Create Service Account

1. In Firebase Console, go to Project Settings (gear icon)
2. Go to "Service accounts" tab
3. Click "Generate new private key"
4. Download the JSON file and save it securely
5. **Important**: Never commit this file to version control!

## Step 3: Environment Configuration

Add these environment variables to your `.env` file:

```bash
# Firebase Configuration
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/your/serviceAccountKey.json

# Existing variables
GROQ_API_KEY=your-groq-api-key
```

### Example `.env` file:
```bash
GROQ_API_KEY=your-groq-api-key-here
FIREBASE_PROJECT_ID=meddigest-12345
FIREBASE_SERVICE_ACCOUNT_PATH=/Users/yourname/Desktop/TechX/firebase-service-account.json
```

## Step 4: Install Dependencies

Make sure you have the Firebase Admin SDK installed:

```bash
pip install firebase-admin
```

## Step 5: Test the Setup

1. **Run the Python script** to generate and store a digest:
   ```bash
   python main.py
   ```

2. **Start the API server**:
   ```bash
   python minimal_api.py
   ```

3. **Test the API endpoint**:
   ```bash
   curl http://localhost:8000/api/newsletter
   ```

## Step 6: Frontend Integration

Your Next.js frontend will automatically work with Firebase through the API. The frontend doesn't need any changes - it will fetch data from the same `/api/newsletter` endpoint.

## Troubleshooting

### Common Issues:

1. **"Firebase not available" error**:
   - Check that your `.env` file has the correct variables
   - Verify the service account JSON file path is correct
   - Ensure the Firebase project ID matches your project

2. **Permission denied errors**:
   - Make sure Firestore rules allow read/write operations
   - For development, you can use test mode rules

3. **Service account file not found**:
   - Verify the file path in `FIREBASE_SERVICE_ACCOUNT_PATH`
   - Make sure the file exists and is readable

### Firestore Security Rules (Development)

For development, you can use these permissive rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} {
      allow read, write: if true;
    }
  }
}
```

**Note**: These rules allow anyone to read/write. For production, implement proper authentication and authorization.

## Data Structure

The digest data is stored in Firestore with this structure:

```
research_digests/
  ├── {digest_id}/
  │   ├── id: string
  │   ├── date_generated: string (ISO format)
  │   ├── total_papers: number
  │   ├── specialty_data: object
  │   ├── batch_analyses: object
  │   └── digest_summary: object
```

## Benefits of Firebase Integration

1. **Real-time updates**: Multiple users can see the same data
2. **Scalability**: Handles large amounts of data efficiently
3. **Backup and recovery**: Automatic data backup
4. **Security**: Built-in authentication and authorization
5. **Analytics**: Track usage and performance
6. **No file management**: No need to handle JSON files manually

## Fallback Mode

If Firebase is not configured, the system will fall back to reading JSON files from the local filesystem, ensuring backward compatibility. 