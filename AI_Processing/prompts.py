"""
Prompts for AI processing of medical research papers.

This module contains all the prompts used by the AI processing components
for analyzing medical research papers and calculating interest scores.
"""

# System prompt that defines the AI's role and analysis requirements
PAPER_ANALYSIS_SYSTEM_ROLE = """
You are an expert medical research analyst with deep knowledge across all medical specialties. 
Your task is to analyze medical research papers and provide accurate categorization and key insights.
Focus on:
- Identifying the primary medical specialty based on the paper's content and methodology
- Extracting the most relevant medical concepts and terminology
- Providing a concise but comprehensive summary that captures the key findings
- Identifying key characteristics for interest scoring (study type, sample size, clinical relevance, etc.)

Be precise and professional in your analysis. Use consistent terminology and avoid subjective language.
"""

# System prompt for methodology detection
METHODOLOGY_DETECTION_SYSTEM_PROMPT = """
You are an expert medical research analyst. Analyze the paper text and identify which methodologies from the provided list are present.
Return a JSON array where each object contains the methodology name and whether it's present (1 for present, 0 for not present).
"""

def create_paper_analysis_prompt(paper, valid_specialties):
    """
    Create the prompt for paper analysis.
    
    Args:
        paper: The paper to analyze
        valid_specialties: List of valid medical specialties
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    Analyze this medical research paper and provide a JSON response with the exact structure shown below.
    
    Title: {paper.title}
    Abstract: {paper.abstract}
    Conclusion: {paper.conclusion}
    Authors: {', '.join(paper.authors)}
    arXiv Categories: {', '.join(paper.categories)}
    
    Instructions:
    1. Write a 2-3 sentence summary of the paper's key findings
    2. Identify the primary medical specialty from this list: {', '.join(sorted(valid_specialties))}
    3. Extract 5 key medical concepts/terms from this research
    4. Identify study characteristics (study type, sample size, clinical relevance, etc.)
    
    IMPORTANT: Return ONLY a valid JSON object with this exact structure:
    {{
        "summary": "2-3 sentence summary of the paper's key findings",
        "specialty": "exact specialty name from the provided list",
        "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
        "study_type": "type of study (e.g., clinical trial, observational study, etc.)",
        "sample_size_indicator": "indication of sample size (e.g., large, small, not specified)",
        "clinical_relevance": "level of clinical relevance (high, moderate, low)"
    }}
    
    Do not include any text before or after the JSON object. Ensure all quotes are properly escaped.
    """

def create_methodology_detection_prompt(paper_text, methodology_list):
    """
    Create the prompt for methodology detection.
    
    Args:
        paper_text: The text of the paper to analyze
        methodology_list: List of methodologies to check for
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    Analyze this medical research paper and identify which methodologies from the list are utilized.
    
    Methodology List: {methodology_list}
    Paper Text: {paper_text}
    
    Return a JSON array with this exact structure:
    [
        {{"methodology": "methodology name", "present": 1}},
        {{"methodology": "methodology name", "present": 0}},
        ...
    ]
    
    IMPORTANT: Return ONLY a valid JSON array. Do not include any text before or after.
    """

def create_batch_analysis_prompt(batch, batch_text, batch_num):
    """
    Create the prompt for batch analysis.
    
    Args:
        batch: List of papers in the batch
        batch_text: Formatted text of all papers in the batch
        batch_num: Batch number
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    You are a medical research analyst tasked with analyzing a batch of {len(batch)} medical research papers. Your goal is to provide a comprehensive analysis that will be used in a medical research digest newsletter.

    PAPERS TO ANALYZE:
    {batch_text}

    ANALYSIS REQUIREMENTS:
    1. Read each paper carefully, focusing on methodology, findings, and clinical implications
    2. Identify connections and patterns across multiple papers in the batch
    3. Consider the broader impact on medical practice and patient care
    4. Note any cross-specialty implications or interdisciplinary connections

    PROVIDE YOUR ANALYSIS IN THE FOLLOWING JSON FORMAT. Return ONLY valid JSON, no additional text:

    {{
        "batch_summary": "2-3 paragraph summary focusing on key findings and implications for current medical practices",
        "significant_findings": ["List of top 5 most significant findings across all papers in this batch"],
        "major_trends": ["List of 2-3 major trends or patterns identified across multiple papers in this batch"],
        "medical_impact": "Brief analysis of potential impact on medical practice and patient care",
        "cross_specialty_insights": "Brief analysis of cross-specialty implications and connections",
        "medical_keywords": ["List of 10-15 relevant medical keywords across all research papers in this batch"],
        "papers_analyzed": {len(batch)},
        "batch_number": {batch_num},
        "specialties_covered": ["List of medical specialties represented in this batch"]
    }}
    
    Ensure the analysis is comprehensive and provides sufficient detail for later integration into a complete newsletter digest.
    Return ONLY the JSON object, no additional text or explanations.
    """

def create_executive_summary_prompt(batch_analysis_results):
    """
    Create the prompt for executive summary generation.
    
    Args:
        batch_analysis_results: Results from batch analyses
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    You are a senior medical research analyst creating an executive summary for a medical research digest newsletter. Your audience includes healthcare professionals, researchers, and medical administrators who need to quickly understand the most important developments in medical research.

    RESEARCH DATA:
    {batch_analysis_results}

    TASK: Generate a compelling executive summary that synthesizes the key insights from all research findings.

    EXECUTIVE SUMMARY REQUIREMENTS:
    1. Start with the most impactful or surprising finding that will grab readers' attention
    2. Identify and discuss 2-3 major themes that emerge across multiple research areas
    3. Emphasize how these findings could change medical practice or improve patient care
    4. Briefly mention what these trends suggest about the future of medicine
    5. Write for an educated medical audience using appropriate terminology

    FORMAT: 2-3 well-structured paragraphs (approximately 300-400 words total)

    CONTENT FOCUS:
    - Prioritize findings with immediate clinical relevance
    - Highlight breakthrough discoveries or novel approaches
    - Emphasize cross-specialty connections and integrated care implications
    - Include specific examples where possible
    - Avoid technical jargon that would confuse non-specialists

    CRITICAL INSTRUCTION: Write the executive summary directly without any introductory phrases like "Here is..." or "This summary..." or "Based on the research data...". Start immediately with the content as if it's the first paragraph of the newsletter.

    Write a clear, engaging executive summary that would make a busy healthcare professional want to read the full digest.
    """

def create_key_discoveries_prompt(batch_analysis_results):
    """
    Create the prompt for key discoveries generation.
    
    Args:
        batch_analysis_results: Results from batch analyses
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    You are a medical research analyst tasked with identifying the most significant discoveries from a comprehensive analysis of medical research papers.

    RESEARCH DATA:
    {batch_analysis_results}

    TASK: Extract and synthesize the 10 most important discoveries across all research findings.

    KEY DISCOVERY CRITERIA:
    - Findings that could change medical practice or improve patient outcomes
    - Novel methodologies, breakthrough technologies, or paradigm shifts
    - Discoveries that have implications across multiple medical fields
    - Well-supported findings with robust methodology
    - Discoveries that can be implemented in clinical settings

    FORMAT REQUIREMENTS:
    - Return exactly 10 discoveries as a JSON array
    - Each discovery should be 1-2 sentences long
    - Be specific and actionable
    - Include the medical specialty or context where relevant
    - Use clear, professional medical terminology

    EXAMPLE FORMAT:
    ["Specific finding with clinical context and impact",
     "Specific finding with clinical context and impact",
     ...]

    IMPORTANT: Return ONLY the JSON array, no additional text or explanations.
    CRITICAL INSTRUCTION: Write the key discoveries directly without any introductory phrases like "Here is..." or "This summary..." or "Based on the research data...". Start immediately with the content as if it's the first paragraph of the newsletter.
    """

def create_emerging_trends_prompt(batch_analysis_results):
    """
    Create the prompt for emerging trends generation.
    
    Args:
        batch_analysis_results: Results from batch analyses
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    You are a medical research analyst specializing in trend analysis and pattern recognition in medical research.

    RESEARCH DATA:
    {batch_analysis_results}

    TASK: Identify and analyze emerging trends that are shaping the future of medical research and practice.

    TREND ANALYSIS FRAMEWORK:
    1. New research approaches, technologies, or analytical methods
    2. Shifts in treatment approaches, diagnostic methods, or care delivery
    3. Convergence of different medical specialties or integration with other fields
    4. Focus on personalized medicine, patient outcomes, or patient experience
    5. AI, machine learning, digital health, or precision medicine advances

    ANALYSIS REQUIREMENTS:
    - Identify 2-3 most significant emerging trends
    - Explain why these trends are important and where they're leading
    - Discuss potential implications for healthcare delivery
    - Consider both opportunities and challenges
    - Include specific examples from the research data

    FORMAT: 1-2 well-structured paragraphs (approximately 200-300 words)
    - Start with the most impactful trend
    - Connect trends to practical implications
    - Use clear, professional medical terminology
    - Focus on actionable insights for healthcare professionals

    CRITICAL INSTRUCTION: Write the emerging trends directly without any introductory phrases like "Here is..." or "This summary..." or "Based on the research data...". Start immediately with the content as if it's the first paragraph of the newsletter.

    Write an analysis that helps readers understand the direction of medical research and its implications for the future of healthcare.
    """

def create_medical_impact_prompt(batch_analysis_results):
    """
    Create the prompt for medical impact generation.
    
    Args:
        batch_analysis_results: Results from batch analyses
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    You are a medical research analyst specializing in translating research findings into clinical practice implications.

    RESEARCH DATA:
    {batch_analysis_results}

    TASK: Analyze the potential impact of these research findings on medical practice and patient care.

    IMPACT ANALYSIS FRAMEWORK:
    1. Immediate clinical applications and practice changes
    2. Patient outcomes and quality of care improvements
    3. Healthcare system efficiency and cost implications
    4. Training and education needs for healthcare professionals
    5. Regulatory and policy considerations

    ANALYSIS REQUIREMENTS:
    - Focus on practical, actionable implications
    - Consider both positive impacts and potential challenges
    - Address implementation considerations and timelines
    - Include specific examples of how findings could be applied
    - Consider different healthcare settings and patient populations

    FORMAT: 1 well-structured paragraph (approximately 150-200 words)
    - Start with the most significant impact
    - Use clear, professional medical terminology
    - Focus on concrete, measurable outcomes
    - Address both opportunities and implementation considerations

    CRITICAL INSTRUCTION: Write the medical impact analysis directly without any introductory phrases like "Here is..." or "This summary..." or "Based on the research data...". Start immediately with the content as if it's the first paragraph of the newsletter.

    Write an analysis that helps healthcare professionals understand how these research findings could change their practice and improve patient care.
    """

def create_cross_specialty_insights_prompt(batch_analysis_results):
    """
    Create the prompt for cross-specialty insights generation.
    
    Args:
        batch_analysis_results: Results from batch analyses
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    You are a medical research analyst specializing in interdisciplinary medicine and cross-specialty collaboration.

    RESEARCH DATA:
    {batch_analysis_results}

    TASK: Identify and analyze cross-specialty insights and interdisciplinary connections from the research findings.

    CROSS-SPECIALTY ANALYSIS FRAMEWORK:
    1. Research findings that span multiple medical specialties
    2. Methodologies or technologies that can be applied across different fields
    3. Patient care approaches that require multi-specialty collaboration
    4. Shared challenges or opportunities across different medical domains
    5. Integration of different medical perspectives and approaches

    ANALYSIS REQUIREMENTS:
    - Identify 2-3 most significant cross-specialty connections
    - Explain how different specialties can learn from each other
    - Discuss collaborative opportunities and integrated care models
    - Consider how findings in one specialty might inform practice in another
    - Address barriers to cross-specialty collaboration and potential solutions

    FORMAT: 1-2 well-structured paragraphs (approximately 200-300 words)
    - Start with the most impactful cross-specialty connection
    - Use clear, professional medical terminology
    - Focus on practical collaboration opportunities
    - Include specific examples of interdisciplinary applications

    CRITICAL INSTRUCTION: Write the cross-specialty insights directly without any introductory phrases like "Here is..." or "This summary..." or "Based on the research data...". Start immediately with the content as if it's the first paragraph of the newsletter.

    Write an analysis that encourages healthcare professionals to think beyond their specialty boundaries and explore collaborative opportunities.
    """

def create_clinical_implications_prompt(batch_analysis_results):
    """
    Create the prompt for clinical implications generation.
    
    Args:
        batch_analysis_results: Results from batch analyses
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    You are a medical research analyst specializing in clinical translation and evidence-based practice.

    RESEARCH DATA:
    {batch_analysis_results}

    TASK: Analyze the clinical implications of these research findings and their potential to change clinical practice.

    CLINICAL IMPLICATIONS FRAMEWORK:
    1. Direct clinical applications and practice recommendations
    2. Changes to diagnostic approaches and treatment protocols
    3. Patient management strategies and care pathways
    4. Risk assessment and prevention strategies
    5. Quality improvement and patient safety implications

    ANALYSIS REQUIREMENTS:
    - Focus on evidence-based clinical recommendations
    - Consider the strength of evidence and confidence in findings
    - Address implementation challenges and practical considerations
    - Include specific clinical scenarios and patient populations
    - Consider both immediate and long-term clinical implications

    FORMAT: 1-2 well-structured paragraphs (approximately 200-300 words)
    - Start with the most clinically significant implications
    - Use clear, professional medical terminology
    - Focus on actionable clinical recommendations
    - Address both benefits and potential risks or limitations

    CRITICAL INSTRUCTION: Write the clinical implications directly without any introductory phrases like "Here is..." or "This summary..." or "Based on the research data...". Start immediately with the content as if it's the first paragraph of the newsletter.

    Write an analysis that helps clinicians understand how these research findings should influence their practice and patient care decisions.
    """

def create_research_gaps_prompt(batch_analysis_results):
    """
    Create the prompt for research gaps generation.
    
    Args:
        batch_analysis_results: Results from batch analyses
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    You are a medical research analyst specializing in research methodology and identifying knowledge gaps in medical science.

    RESEARCH DATA:
    {batch_analysis_results}

    TASK: Identify and analyze research gaps and areas where additional investigation is needed.

    RESEARCH GAPS ANALYSIS FRAMEWORK:
    1. Questions that remain unanswered by current research
    2. Methodological limitations in existing studies
    3. Populations or conditions that are understudied
    4. Gaps in clinical translation and implementation research
    5. Areas where conflicting evidence exists and needs resolution

    ANALYSIS REQUIREMENTS:
    - Identify 2-3 most critical research gaps
    - Explain why these gaps are important to address
    - Suggest specific research approaches or methodologies needed
    - Consider the priority and feasibility of addressing each gap
    - Discuss the potential impact of filling these gaps

    FORMAT: 1-2 well-structured paragraphs (approximately 200-300 words)
    - Start with the most significant research gap
    - Use clear, professional medical terminology
    - Focus on actionable research priorities
    - Include specific suggestions for future research directions

    CRITICAL INSTRUCTION: Write the research gaps analysis directly without any introductory phrases like "Here is..." or "This summary..." or "Based on the research data...". Start immediately with the content as if it's the first paragraph of the newsletter.

    Write an analysis that helps researchers and funding agencies understand where to focus future research efforts for maximum impact.
    """

def create_future_directions_prompt(batch_analysis_results):
    """
    Create the prompt for future directions generation.
    
    Args:
        batch_analysis_results: Results from batch analyses
        
    Returns:
        str: Formatted prompt for the LLM
    """
    return f"""
    You are a medical research analyst specializing in forecasting and strategic planning in medical research.

    RESEARCH DATA:
    {batch_analysis_results}

    TASK: Analyze the research findings to predict future directions and emerging opportunities in medical research and practice.

    FUTURE DIRECTIONS ANALYSIS FRAMEWORK:
    1. Emerging technologies and methodologies that show promise
    2. Shifts in research priorities and funding focus areas
    3. Integration of different research approaches and disciplines
    4. Evolution of clinical practice models and healthcare delivery
    5. Long-term implications for patient care and population health

    ANALYSIS REQUIREMENTS:
    - Identify 2-3 most promising future directions
    - Explain the rationale and evidence supporting these predictions
    - Discuss the timeline and feasibility of these developments
    - Consider both opportunities and potential challenges
    - Include specific recommendations for researchers and healthcare professionals

    FORMAT: 1-2 well-structured paragraphs (approximately 200-300 words)
    - Start with the most promising future direction
    - Use clear, professional medical terminology
    - Focus on actionable insights and recommendations
    - Include specific examples and potential applications

    CRITICAL INSTRUCTION: Write the future directions analysis directly without any introductory phrases like "Here is..." or "This summary..." or "Based on the research data...". Start immediately with the content as if it's the first paragraph of the newsletter.

    Write an analysis that helps readers understand the trajectory of medical research and prepare for future developments in healthcare.
    """ 