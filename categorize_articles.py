import datetime
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import time
from langchain_groq import ChatGroq

now_utc = datetime.datetime.now(datetime.timezone.utc)
one_week_ago = now_utc - datetime.timedelta(days=7)

BASE_URL = "https://export.arxiv.org/api/query"
search_query = "all:medical"
params = {
    "search_query": search_query,
    "sortBy": "submittedDate",
    "sortOrder": "descending",
    "max_results": 20,
}

# Proper URL encoding
url = f"{BASE_URL}?{urlencode(params)}"

# Add headers to make request more browser-like
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/xml,application/xhtml+xml,text/html;q=0.9',
}

print("Fetching papers from arXiv...")
try:
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Failed to fetch papers: {str(e)}")
    exit(1)

# Parse XML
root = ET.fromstring(response.content)
ns = {'atom': 'http://www.w3.org/2005/Atom'}

# Extract papers
papers = []
entries = root.findall('atom:entry', ns)

for entry in entries:
    paper = {
        "paper_id": entry.find('atom:id', ns).text.split('/')[-1],
        "title": entry.find('atom:title', ns).text.strip(),
        "published": datetime.datetime.strptime(
            entry.find('atom:published', ns).text, 
            '%Y-%m-%dT%H:%M:%SZ'
        ).replace(tzinfo=datetime.timezone.utc),
        "abstract": entry.find('atom:summary', ns).text.strip(),
        "authors": [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
        "categories": [cat.get('term') for cat in entry.findall('atom:category', ns)]
    }
    papers.append(paper)

print(f"Found {len(papers)} papers\n")

# Setup Groq
llm = ChatGroq(api_key="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", model="llama3-8b-8192")

# Analyze papers
specialty_data = {}

print("ANALYZING PAPERS WITH AI:")
print("="*60)

for i, paper in enumerate(papers, 1):
    print(f"\n{i}. {paper['title'][:80]}...")
    print(f"   Authors: {', '.join(paper['authors'][:3])}")
    if len(paper['authors']) > 3:
        print(f"            + {len(paper['authors']) - 3} more")
    
    # prompt
    prompt = f"""
    Analyze this medical research paper:
    
    Title: {paper['title']}
    Abstract: {paper['abstract'][:500]}
    Authors: {', '.join(paper['authors'][:5])}
    arXiv Categories: {', '.join(paper['categories'])}
    
    Provide:
    1. Medical specialty (ONE of: Cardiology, Oncology, Radiology, Neurology, Surgery, Psychiatry, Endocrinology, General Medicine)
    2. 5 key medical concepts/terms from this research
    3. Main research focus in one sentence
    
    Format your response EXACTLY like this:
    Specialty: [specialty]
    Keywords: [keyword1], [keyword2], [keyword3], [keyword4], [keyword5]
    Focus: [one sentence description]
    """
    
    try:
        response = llm.invoke(prompt)
        output = response.content
        
        # response
        lines = output.strip().split('\n')
        specialty = None
        keywords = []
        focus = ""
        
        for line in lines:
            if line.startswith("Specialty:"):
                specialty = line.replace("Specialty:", "").strip()
            elif line.startswith("Keywords:"):
                keywords_text = line.replace("Keywords:", "").strip()
                keywords = [k.strip() for k in keywords_text.split(',')][:5]
            elif line.startswith("Focus:"):
                focus = line.replace("Focus:", "").strip()
        
        # results
        if specialty:
            if specialty not in specialty_data:
                specialty_data[specialty] = {
                    "papers": [],
                    "all_keywords": [],
                    "author_network": set()
                }
            
            specialty_data[specialty]["papers"].append({
                "id": paper['paper_id'],
                "title": paper['title'],
                "authors": paper['authors'],
                "keywords": keywords,
                "focus": focus,
                "date": paper['published'].strftime("%Y-%m-%d")
            })
            specialty_data[specialty]["all_keywords"].extend(keywords)
            specialty_data[specialty]["author_network"].update(paper['authors'])
            
            print(f"   Specialty: {specialty}")
            print(f"   Keywords: {', '.join(keywords)}")
            print(f"   Focus: {focus}")
            
    except Exception as e:
        print(f"   Error analyzing paper: {str(e)}")

# Display summary
print("\n" + "="*60)
print("MEDICAL RESEARCH SUMMARY BY SPECIALTY:")
print("="*60)

for specialty, data in sorted(specialty_data.items(), key=lambda x: len(x[1]["papers"]), reverse=True):
    num_papers = len(data["papers"])
    num_authors = len(data["author_network"])
    
    print(f"\n{specialty.upper()} ({num_papers} papers, {num_authors} unique authors)")
    print("-"*50)
    
    # Recent papers
    print("\nRecent Papers:")
    for j, paper in enumerate(data["papers"][:3], 1):
        print(f"  {j}. [{paper['date']}] {paper['title'][:60]}...")
        print(f"     Focus: {paper['focus'][:80]}...")
    
    # Top keywords
    keyword_freq = {}
    for kw in data["all_keywords"]:
        keyword_freq[kw.lower()] = keyword_freq.get(kw.lower(), 0) + 1
    
    print("\nTop Research Terms:")
    top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:8]
    for keyword, count in top_keywords:
        print(f"  • {keyword} ({count} papers)")

print("\n" + "="*60)