import streamlit as st
import re
import os
import tempfile
import subprocess
import PyPDF2
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import textwrap
import pdfplumber
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="CV Optimizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
    }
    .comparison-container {
        display: flex;
        gap: 20px;
    }
    .comparison-column {
        flex: 1;
        padding: 10px;
        border-radius: 5px;
        background-color: #f7f7f7;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'original_latex' not in st.session_state:
    st.session_state.original_latex = ""
if 'optimized_latex' not in st.session_state:
    st.session_state.optimized_latex = ""
if 'ats_score' not in st.session_state:
    st.session_state.ats_score = 0
if 'keywords' not in st.session_state:
    st.session_state.keywords = []
if 'modified_sections' not in st.session_state:
    st.session_state.modified_sections = {}

# Function to load NLP models
@st.cache_resource
def load_nlp_models():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    keyword_extractor = pipeline("feature-extraction", model="distilbert-base-uncased")
    return tokenizer, model, summarizer, keyword_extractor

# Mean Pooling for BERT embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to get embeddings
def get_embedding(texts, tokenizer, model):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings.numpy()

# Extract keywords from text
def extract_keywords(text, keyword_extractor, top_n=10):
    # Use a simple CountVectorizer approach for keywords
    n_gram_range = (1, 2)
    stop_words = "english"
    
    # Extract candidate words/phrases
    count_vectorizer = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words)
    count = count_vectorizer.fit_transform([text])
    
    candidates = count_vectorizer.get_feature_names_out()
    
    # Get word embeddings
    candidate_embeddings = keyword_extractor(list(candidates))
    candidate_embeddings = [np.mean(emb, axis=0) for emb in candidate_embeddings]
    
    # Calculate distances and extract keywords
    distances = cosine_similarity(candidate_embeddings)
    
    # Get top keywords based on similarity
    keywords = [(candidates[idx], float(np.mean(distances[idx]))) 
                for idx in range(len(candidates))]
    
    # Sort by score and select top N
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    return keywords[:top_n]

# Parse LaTeX file
def parse_latex(latex_content):
    sections = {}
    current_section = None
    
    # Pattern to match section commands
    section_pattern = re.compile(r'\\(section|subsection)\{([^}]+)\}')
    
    lines = latex_content.split('\n')
    buffer = []
    
    for line in lines:
        match = section_pattern.search(line)
        if match:
            # If we were working on a section, save it
            if current_section:
                sections[current_section] = '\n'.join(buffer)
                buffer = []
            
            current_section = match.group(2)
            buffer.append(line)
        else:
            buffer.append(line)
    
    # Save the last section
    if current_section and buffer:
        sections[current_section] = '\n'.join(buffer)
    
    # Add preamble (everything before the first section)
    preamble_end = latex_content.find('\\section') if '\\section' in latex_content else latex_content.find('\\begin{document}')
    if preamble_end > 0:
        sections['preamble'] = latex_content[:preamble_end]
    
    return sections

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Optimize CV content based on job description
def optimize_cv(latex_content, jd_text, tokenizer, model, summarizer, keyword_extractor):
    # Extract key information
    jd_keywords = extract_keywords(jd_text, keyword_extractor, top_n=15)
    jd_keyword_phrases = [kw[0] for kw in jd_keywords]
    
    # Parse the LaTeX file
    sections = parse_latex(latex_content)
    optimized_sections = sections.copy()
    modified_sections = {}
    
    # Process each section
    for section_name, section_content in sections.items():
        if section_name.lower() in ['skills', 'experience', 'education', 'projects', 'work experience']:
            # Extract actual content without LaTeX commands for analysis
            clean_content = re.sub(r'\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?', ' ', section_content)
            clean_content = re.sub(r'[{}\\]', ' ', clean_content)
            
            # Calculate relevance to JD
            section_embedding = get_embedding([clean_content], tokenizer, model)
            jd_embedding = get_embedding([jd_text], tokenizer, model)
            similarity = cosine_similarity(section_embedding, jd_embedding)[0][0]
            
            # If similarity is low, try to optimize
            if similarity < 0.75:  # Threshold for optimization
                # For skills section: emphasize matching skills
                if section_name.lower() == 'skills':
                    skills_pattern = re.compile(r'\\item\s+([^\\]+)')
                    skills = skills_pattern.findall(section_content)
                    
                    # Reorder skills to prioritize ones in JD
                    prioritized_skills = []
                    other_skills = []
                    
                    for skill in skills:
                        skill = skill.strip()
                        if any(kw.lower() in skill.lower() for kw in jd_keyword_phrases):
                            prioritized_skills.append(f"\\item \\textbf{{{skill}}}")
                        else:
                            other_skills.append(f"\\item {skill}")
                    
                    # Reconstruct the skills section
                    skills_list = '\n'.join(prioritized_skills + other_skills)
                    new_section = re.sub(r'(\\begin\{itemize\}).*?(\\end\{itemize\})', 
                                        f'\\1\n{skills_list}\n\\2', 
                                        section_content, flags=re.DOTALL)
                    
                    optimized_sections[section_name] = new_section
                    modified_sections[section_name] = True
                
                # For experience sections: emphasize relevant experiences
                elif 'experience' in section_name.lower() or 'project' in section_name.lower():
                    # Find each experience item
                    experience_pattern = re.compile(r'\\item.*?(?=\\item|\\end\{itemize\})', re.DOTALL)
                    experiences = experience_pattern.findall(section_content + "\\end{itemize}")
                    
                    optimized_experiences = []
                    for exp in experiences:
                        # Calculate relevance
                        clean_exp = re.sub(r'\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?', ' ', exp)
                        clean_exp = re.sub(r'[{}\\]', ' ', clean_exp)
                        
                        # Count keyword matches
                        keyword_matches = sum(1 for kw in jd_keyword_phrases if kw.lower() in clean_exp.lower())
                        
                        # If experience matches keywords, emphasize it
                        if keyword_matches > 0:
                            # Add bold to keywords
                            for kw in jd_keyword_phrases:
                                if kw.lower() in clean_exp.lower() and len(kw) > 3:  # Avoid very short keywords
                                    pattern = re.compile(re.escape(kw), re.IGNORECASE)
                                    exp = pattern.sub(f"\\\\textbf{{{kw}}}", exp)
                            
                            optimized_experiences.append(exp)
                        else:
                            optimized_experiences.append(exp)
                    
                    # Sort experiences by relevance (more keyword matches first)
                    # For simplicity, we keep the original order here
                    
                    # Reconstruct the experience section
                    experiences_text = '\n'.join(optimized_experiences)
                    new_section = re.sub(r'(\\begin\{itemize\}).*?(?=\\end\{itemize\})', 
                                        f'\\1\n{experiences_text}', 
                                        section_content, flags=re.DOTALL)
                    
                    optimized_sections[section_name] = new_section
                    modified_sections[section_name] = True
    
    # Reconstruct the full LaTeX document
    optimized_latex = ""
    for section_name, section_content in optimized_sections.items():
        if section_name == 'preamble':
            optimized_latex = section_content + optimized_latex
        else:
            optimized_latex += section_content + "\n\n"
    
    return optimized_latex, modified_sections

# Calculate ATS score based on keyword matching and format
def calculate_ats_score(cv_text, jd_text, tokenizer, model):
    # Extract clean text from LaTeX
    clean_cv = re.sub(r'\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?', ' ', cv_text)
    clean_cv = re.sub(r'[{}\\]', ' ', clean_cv)
    
    # Get embeddings
    cv_embedding = get_embedding([clean_cv], tokenizer, model)
    jd_embedding = get_embedding([jd_text], tokenizer, model)
    
    # Calculate similarity
    semantic_similarity = cosine_similarity(cv_embedding, jd_embedding)[0][0]
    
    # Calculate keyword match percentage
    cv_words = set(clean_cv.lower().split())
    jd_words = set(jd_text.lower().split())
    common_words = cv_words.intersection(jd_words)
    keyword_match = len(common_words) / max(1, len(jd_words))
    
    # Check for formatting issues in LaTeX
    format_score = 1.0
    # Deduct points for potential formatting issues
    if '\\tiny' in cv_text or '\\scriptsize' in cv_text:
        format_score -= 0.1  # Font too small
    if cv_text.count('\\section') < 3:
        format_score -= 0.1  # Too few sections
    
    # Final ATS score (weighted)
    ats_score = (semantic_similarity * 0.5) + (keyword_match * 0.3) + (format_score * 0.2)
    return min(1.0, max(0.0, ats_score)) * 100  # Convert to percentage

# Generate LaTeX for different formats
def generate_formatted_latex(latex_content, format_type):
    if format_type == "side_panel":
        # Check if already using a two-column package
        if "\\usepackage{multicol}" not in latex_content and "\\begin{multicols}" not in latex_content:
            # Add necessary packages for side panel format
            if "\\documentclass" in latex_content:
                latex_content = re.sub(r'(\\documentclass.*?})', 
                                     r'\1\n\\usepackage{multicol}\n\\usepackage{geometry}\n\\geometry{margin=1in}', 
                                     latex_content)
                
                # Modify document environment to use multicols
                latex_content = re.sub(r'(\\begin{document})', 
                                     r'\1\n\\begin{multicols}{2}', 
                                     latex_content)
                latex_content = re.sub(r'(\\end{document})', 
                                     r'\\end{multicols}\n\1', 
                                     latex_content)
    else:  # long_format
        # Remove two-column formatting if it exists
        latex_content = re.sub(r'\\usepackage{multicol}', '', latex_content)
        latex_content = re.sub(r'\\begin{multicols}{2}', '', latex_content)
        latex_content = re.sub(r'\\end{multicols}', '', latex_content)
    
    return latex_content

# Convert LaTeX to PDF and get download link
def get_pdf_download_link(latex_content):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save LaTeX content to a temporary file
        tex_path = os.path.join(tmpdir, "cv.tex")
        with open(tex_path, "w") as f:
            f.write(latex_content)
        
        # Compile LaTeX to PDF
        try:
            subprocess.run(["pdflatex", "-output-directory", tmpdir, tex_path], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Read the generated PDF
            pdf_path = os.path.join(tmpdir, "cv.pdf")
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            # Create a download link
            b64 = base64.b64encode(pdf_bytes).decode()
            return f'<a href="data:application/pdf;base64,{b64}" download="optimized_cv.pdf" class="download-button">Download Optimized CV PDF</a>'
        except subprocess.CalledProcessError:
            return "Error compiling LaTeX to PDF. Please check your LaTeX code."

# Main app functionality
tokenizer, model, summarizer, keyword_extractor = load_nlp_models()

# App header
st.title("CV Optimizer for Job Applications")
st.markdown("""
This app helps you tailor your LaTeX CV to specific job descriptions using NLP techniques. 
Upload your CV and job description to get an optimized version with improved ATS score.
""")

# Sidebar options
st.sidebar.header("Options")
cv_format = st.sidebar.radio("CV Format", ["long_format", "side_panel"], 
                            format_func=lambda x: "Single Column" if x == "long_format" else "Two Column")

# File upload area
col1, col2 = st.columns(2)

with col1:
    st.header("Upload LaTeX CV")
    latex_file = st.file_uploader("Choose a LaTeX (.tex) file", type="tex")
    if latex_file is not None:
        latex_content = latex_file.getvalue().decode("utf-8")
        st.session_state.original_latex = latex_content
        st.success("LaTeX CV uploaded successfully!")

with col2:
    st.header("Upload Job Description")
    jd_option = st.radio("Job Description Format", ["Text", "PDF"])
    
    if jd_option == "Text":
        jd_text = st.text_area("Paste job description here", height=300)
    else:
        jd_file = st.file_uploader("Upload job description PDF", type="pdf")
        if jd_file is not None:
            jd_text = extract_text_from_pdf(jd_file)
            st.success("Job description PDF processed successfully!")
            # Show a preview
            with st.expander("Preview extracted job description"):
                st.text(jd_text[:500] + "..." if len(jd_text) > 500 else jd_text)

# Process button
if st.button("Optimize CV for Job Description"):
    if 'original_latex' in st.session_state and st.session_state.original_latex and jd_text:
        # Show processing indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Process job description
        status_text.text("Analyzing job description...")
        progress_bar.progress(20)
        jd_keywords = extract_keywords(jd_text, keyword_extractor, top_n=15)
        st.session_state.keywords = jd_keywords
        
        # Step 2: Optimize CV
        status_text.text("Optimizing CV content...")
        progress_bar.progress(50)
        optimized_latex, modified_sections = optimize_cv(
            st.session_state.original_latex, jd_text, 
            tokenizer, model, summarizer, keyword_extractor
        )
        st.session_state.modified_sections = modified_sections
        
        # Step 3: Apply format
        status_text.text("Applying selected format...")
        progress_bar.progress(70)
        formatted_latex = generate_formatted_latex(optimized_latex, cv_format)
        st.session_state.optimized_latex = formatted_latex
        
        # Step 4: Calculate ATS score
        status_text.text("Calculating ATS score...")
        progress_bar.progress(90)
        ats_score = calculate_ats_score(formatted_latex, jd_text, tokenizer, model)
        st.session_state.ats_score = ats_score
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Optimization complete!")
        st.session_state.processed = True
        
    else:
        st.error("Please upload both a LaTeX CV and a job description.")

# Display results if processed
if st.session_state.processed:
    st.markdown("---")
    st.header("Optimization Results")
    
    # Display ATS score with gauge
    ats_col1, ats_col2 = st.columns([1, 3])
    with ats_col1:
        st.subheader("ATS Score")
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 3rem; font-weight: bold; color: {'green' if st.session_state.ats_score >= 80 else 'orange' if st.session_state.ats_score >= 60 else 'red'};">
                {st.session_state.ats_score:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with ats_col2:
        # Show keywords from JD
        st.subheader("Key Job Requirements")
        keyword_df = pd.DataFrame(
            [(kw, score) for kw, score in st.session_state.keywords[:10]], 
            columns=["Keyword/Phrase", "Relevance"]
        )
        keyword_df["Relevance"] = keyword_df["Relevance"].apply(lambda x: f"{x:.2f}")
        st.dataframe(keyword_df, use_container_width=True)
    
    # Show modified sections
    st.subheader("Modified Sections")
    if st.session_state.modified_sections:
        for section, modified in st.session_state.modified_sections.items():
            if modified:
                st.markdown(f"✓ **{section}** - Updated to better match job requirements")
    else:
        st.markdown("No sections required significant modification - your CV already matches the job well!")
    
    # Show comparison and LaTeX code tabs
    st.markdown("---")
    tab1, tab2 = st.tabs(["CV Content Comparison", "LaTeX Code"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original CV")
            # Display a simplified version of the original CV
            clean_original = re.sub(r'\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?', ' ', st.session_state.original_latex)
            clean_original = re.sub(r'[{}\\]', ' ', clean_original)
            st.text_area("", clean_original, height=400, disabled=True)
            
        with col2:
            st.subheader("Optimized CV")
            # Display a simplified version of the optimized CV
            clean_optimized = re.sub(r'\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?', ' ', st.session_state.optimized_latex)
            clean_optimized = re.sub(r'[{}\\]', ' ', clean_optimized)
            st.text_area("", clean_optimized, height=400, disabled=True)
    
    with tab2:
        st.subheader("Optimized LaTeX Code")
        st.text_area("Copy this code to use in your LaTeX editor", st.session_state.optimized_latex, height=400)
        
        # LaTeX download button
        latex_download = st.download_button(
            label="Download LaTeX File",
            data=st.session_state.optimized_latex,
            file_name="optimized_cv.tex",
            mime="text/plain"
        )
        
        # PDF download (requires LaTeX installation)
        st.markdown("### PDF Download")
        st.markdown("""
        To generate a PDF, the app needs LaTeX installed on the server. If running locally with LaTeX installed, 
        use the button below. On cloud deployments, please use the LaTeX code in your preferred LaTeX editor.
        """)
        
        if st.button("Generate PDF (requires LaTeX installation)"):
            try:
                pdf_link = get_pdf_download_link(st.session_state.optimized_latex)
                st.markdown(pdf_link, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                st.info("Please download the LaTeX file and compile it locally instead.")

# Instructions at the bottom
with st.expander("How to use this app"):
    st.markdown("""
    ### Instructions
    
    1. **Upload your LaTeX CV** - Select your existing CV in .tex format
    2. **Add job description** - Either paste the text or upload a PDF
    3. **Choose CV format** - Select between single-column or two-column layout
    4. **Click "Optimize CV"** - The app will process your CV and job description
    5. **Review results** - Check the modified sections and ATS score
    6. **Download the optimized CV** - Get the LaTeX code or PDF for your application
    
    ### Tips for best results
    
    * Ensure your original LaTeX CV is properly formatted
    * Include detailed job descriptions for better keyword matching
    * Review and edit the optimized CV before submitting applications
    * Consider making additional manual adjustments based on the identified keywords
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
CV Optimizer App | Built with Streamlit and Hugging Face Transformers
</div>
""", unsafe_allow_html=True)
