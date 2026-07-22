import streamlit as st
import pdfplumber
import docx
import requests
from bs4 import BeautifulSoup
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import markdown
from fpdf import FPDF
import time

# --- Initialize Gemini Client ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except KeyError:
    st.error("Missing GEMINI_API_KEY. Please add it to your Streamlit Community Cloud Secrets.")
    st.stop()

# --- Helper Functions ---
def generate_with_retry(prompt, model_name="gemini-2.0-flash", retries=3):
    """Wraps the API call with an automatic retry mechanism for 429 limits."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                st.warning(f"⚠️ Google API Free Tier limit hit. Cooling down for 35 seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(35)
            else:
                raise e

def extract_text_from_file(uploaded_file):
    """Extracts text from PDF, DOCX, or TXT files."""
    text = ""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        elif file_extension == 'docx':
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_extension == 'txt':
            text = uploaded_file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        
    return text

def scrape_job_description(url):
    """Fetches and cleans visible text from a job posting URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "header", "footer", "nav", "noscript", "svg"]):
            element.decompose()
            
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        clean_text = "\n".join(chunk for chunk in lines if chunk)
        
        return clean_text
    except Exception as e:
        st.error(f"Failed to extract job description from URL: {e}")
        return None

def calculate_ats_score(resume_text, job_description):
    """Calculates a Cosine Similarity percentage between the CV and Job Description."""
    if not resume_text.strip() or not job_description.strip():
        return 0.0
        
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

def optimize_resume(resume_text, job_description):
    """Sends the data to the API using the retry logic."""
    prompt = f"""
    You are an expert ATS resume writer. 
    I will provide my current resume and a job description.
    
    STRICT RULES:
    1. Do not invent any new skills, experiences, or degrees.
    2. Rewrite my experience bullet points to highlight relevance to the job description using the Action + Context + Result format.
    3. Weave in matching keywords from the job description naturally.
    4. Output ONLY the optimized resume in Markdown format.
    5. KEEP FORMATTING SIMPLE: Only use headings, bullet points, and bold text.
    
    Job Description:
    {job_description}
    
    My Resume:
    {resume_text}
    """
    return generate_with_retry(prompt)

def generate_cover_letter(resume_text, job_description):
    """Generates the cover letter using the retry logic."""
    prompt = f"""
    Write a professional, authentic, and highly human-sounding cover letter based on the applicant's resume and the job description.
    
    CRITICAL INSTRUCTIONS TO AVOID SOUNDING LIKE AI:
    1. DO NOT use generic, robotic openings like "I am writing to express my interest in..." or "I am thrilled to apply for..."
    2. START with a strong, direct hook that immediately states the value the candidate brings to a specific problem or need mentioned in the job description.
    3. BANNED WORDS: Do not use words like "delve", "testament", "orchestrated", "synergy", "pivotal", "embark", or "furthermore". Keep the vocabulary natural and conversational but professional.
    4. Show, don't just tell. Connect 1 or 2 specific achievements from the resume directly to the needs of the job.
    5. Keep it concise (max 3-4 short paragraphs).
    6. Output ONLY the cover letter text. Include standard [Insert Name/Date/Company] brackets if specific info is missing.
    
    Job Description:
    {job_description}
    
    My Resume:
    {resume_text}
    """
    return generate_with_retry(prompt)

def create_pdf_from_markdown(md_text):
    """Converts simple Markdown to PDF using fpdf2."""
    html_content = markdown.markdown(md_text)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    
    pdf.write_html(html_content)
    return bytes(pdf.output())

def create_pdf_from_text(plain_text):
    """Creates a basic PDF for the cover letter."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    
    clean_text = plain_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, clean_text)
    
    return bytes(pdf.output())

# --- Streamlit Frontend ---
st.set_page_config(page_title="AI CV Optimizer", page_icon="📄", layout="wide")

st.title("CV Optimizer & Cover Letter Generator")
st.markdown("Powered by the free **Gemini API**")

option = st.radio("Choose Job Description Input Method:", ["Paste Job Description Text", "Paste Job Posting Link (URL)"])

job_desc = ""

if option == "Paste Job Description Text":
    job_desc = st.text_area("Paste Target Job Description:", height=150)
else:
    job_url = st.text_input("Paste Job Posting URL (e.g., https://...):")
    if job_url:
        with st.spinner("Scraping job description from URL..."):
            scraped_text = scrape_job_description(job_url)
            if scraped_text:
                job_desc = scraped_text
                st.success("Successfully retrieved job description from URL!")
                with st.expander("Preview Extracted Job Text"):
                    st.text(job_desc[:800] + "..." if len(job_desc) > 800 else job_desc)

uploaded_cv = st.file_uploader("Upload your current CV (.pdf, .docx, or .txt)", type=["pdf", "docx", "txt"])

if st.button("Optimize My CV & Generate Cover Letter", type="primary"):
    if not uploaded_cv or not job_desc:
        st.warning("Please upload a CV and provide a Job Description/URL first.")
    else:
        with st.spinner(f"Extracting text from {uploaded_cv.name}..."):
            raw_resume_text = extract_text_from_file(uploaded_cv)
            
        if not raw_resume_text.strip():
            st.error("Could not extract text from this file. It might be an empty file or a scanned image.")
        else:
            # 1. Baseline Score
            original_score = calculate_ats_score(raw_resume_text, job_desc)
            st.info(f"Baseline ATS Match Score: **{original_score}%**")
            
            try:
                # 2. AI Generation - Resume
                with st.spinner("AI is analyzing keywords and rewriting your CV..."):
                    optimized_markdown = optimize_resume(raw_resume_text, job_desc)
                    new_score = calculate_ats_score(optimized_markdown, job_desc)
                    st.success(f"New ATS Match Score: **{new_score}%** (An improvement of {round(new_score - original_score, 2)}%)")
                
                # Intentional Pause to prevent 429 Rate Limit
                st.info("⏳ Pausing for 15 seconds to respect Google's free tier limits before drafting the cover letter...")
                time.sleep(15)
                
                # 3. AI Generation - Cover Letter
                with st.spinner("Drafting your Cover Letter..."):
                    cover_letter_text = generate_cover_letter(raw_resume_text, job_desc)
                
                st.markdown("---")
                
                # 4. Display in Tabs
                tab1, tab2 = st.tabs(["📝 Tailored CV", "✉️ Professional Cover Letter"])
                
                with tab1:
                    st.markdown(optimized_markdown)
                    pdf_cv = create_pdf_from_markdown(optimized_markdown)
                    st.download_button(
                        label="⬇️ Download Optimized CV as PDF",
                        data=pdf_cv,
                        file_name="Optimized_ATS_Resume.pdf",
                        mime="application/pdf"
                    )
                    
                with tab2:
                    st.markdown(cover_letter_text)
                    pdf_cl = create_pdf_from_text(cover_letter_text)
                    st.download_button(
                        label="⬇️ Download Cover Letter as PDF",
                        data=pdf_cl,
                        file_name="Cover_Letter.pdf",
                        mime="application/pdf"
                    )
                    
            except Exception as e:
                st.error(f"An error occurred during AI processing: {e}")
