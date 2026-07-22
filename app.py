import streamlit as st
import pdfplumber
import tempfile
import os
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from markdown_pdf import MarkdownPdf, Section

# --- Initialize Gemini Client ---
# Streamlit Cloud reads this automatically from your Advanced Settings Secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except KeyError:
    st.error("Missing GEMINI_API_KEY. Please add it to your Streamlit Community Cloud Secrets.")
    st.stop()

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    """Extracts raw text from the uploaded PDF CV."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def calculate_ats_score(resume_text, job_description):
    """Calculates a Cosine Similarity percentage between the CV and Job Description."""
    if not resume_text.strip() or not job_description.strip():
        return 0.0
        
    vectorizer = TfidfVectorizer(stop_words='english')
    # Convert text to numerical vectors
    vectors = vectorizer.fit_transform([resume_text, job_description])
    # Calculate similarity math
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

def optimize_resume(resume_text, job_description):
    """Sends the data to the free Gemini API for intelligent rewriting."""
    prompt = f"""
    You are an expert ATS resume writer. 
    I will provide my current resume and a job description.
    
    STRICT RULES:
    1. Do not invent any new skills, experiences, or degrees.
    2. Rewrite my experience bullet points to highlight relevance to the job description using the Action + Context + Result format.
    3. Weave in matching keywords from the job description naturally.
    4. Output ONLY the optimized resume in Markdown format. Do not include introductory text like "Here is your resume".
    
    Job Description:
    {job_description}
    
    My Resume:
    {resume_text}
    """
    
    # Using gemini-1.5-flash which is fast, capable, and strictly available on the free tier
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    return response.text

def create_pdf_from_markdown(md_text):
    """Converts Markdown text to PDF using a pure Python library and temp files for safety."""
    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(md_text, toc=False))
    
    # Create a temporary file to save the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        
    # Save the PDF to the temporary physical path
    pdf.save(tmp_path)
    
    # Read the bytes back into memory for Streamlit's download button
    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()
        
    # Clean up the temporary file so we don't clutter the server
    os.remove(tmp_path)
    
    return pdf_bytes


# --- Streamlit Frontend ---
st.set_page_config(page_title="AI CV Optimizer", page_icon="📄")

st.title("CV Optimizer (ATS Matcher)")
st.markdown("Powered by the free **Gemini API**")

job_desc = st.text_area("Paste the Target Job Description here:", height=200)
uploaded_cv = st.file_uploader("Upload your current CV (PDF format)", type="pdf")

if st.button("Optimize My CV", type="primary"):
    if not uploaded_cv or not job_desc:
        st.warning("Please upload a CV and paste a Job Description first.")
    else:
        with st.spinner("Extracting text from your PDF..."):
            raw_resume_text = extract_text_from_pdf(uploaded_cv)
            
        if not raw_resume_text.strip():
            st.error("Could not extract text from this PDF. It might be an image rather than text.")
        else:
            # 1. Baseline Score
            original_score = calculate_ats_score(raw_resume_text, job_desc)
            st.info(f"Baseline ATS Match Score: **{original_score}%**")
            
            with st.spinner("Analyzing keywords and rewriting CV via AI..."):
                try:
                    # 2. AI Optimization
                    optimized_markdown = optimize_resume(raw_resume_text, job_desc)
                    
                    # 3. New Score
                    new_score = calculate_ats_score(optimized_markdown, job_desc)
                    st.success(f"New ATS Match Score: **{new_score}%** (An improvement of {round(new_score - original_score, 2)}%)")
                    
                    st.markdown("---")
                    st.markdown("### Your Tailored CV")
                    st.markdown(optimized_markdown)
                    st.markdown("---")
                    
                    # 4. PDF Generation
                    with st.spinner("Generating downloadable PDF..."):
                        pdf_file = create_pdf_from_markdown(optimized_markdown)
                    
                    # 5. Download Button
                    st.download_button(
                        label="⬇️ Download Optimized CV as PDF",
                        data=pdf_file,
                        file_name="Optimized_ATS_Resume.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"An error occurred during AI processing: {e}")
