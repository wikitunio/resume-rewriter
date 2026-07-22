import streamlit as st
import pdfplumber
import docx
import requests
from bs4 import BeautifulSoup
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import markdown
from fpdf import FPDF
import io
import re

# --- Initialize Groq Client ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except KeyError:
    st.error("Missing GROQ_API_KEY. Please add it to your Streamlit Community Cloud Secrets.")
    st.stop()

# --- Helper Functions ---
def sanitize_text_for_fpdf(text):
    """Replaces Unicode characters with FPDF-safe Latin-1 equivalents."""
    replacements = {
        '•': '-', '–': '-', '—': '-',
        '‘': "'", '’': "'",
        '“': '"', '”': '"',
        '…': '...', '✓': 'v'
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    # Drop any remaining unsupported characters safely
    return text.encode('latin-1', 'ignore').decode('latin-1')

def extract_text_from_file(uploaded_file):
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
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "header", "footer", "nav", "noscript", "svg"]):
            element.decompose()
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        return "\n".join(chunk for chunk in lines if chunk)
    except Exception as e:
        st.error(f"Failed to extract job description from URL: {e}")
        return None

def calculate_ats_score(resume_text, job_description):
    if not resume_text.strip() or not job_description.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

def optimize_resume(resume_text, job_description):
    prompt = f"""
    You are an expert ATS resume writer. 
    I will provide my current resume and a job description.
    
    STRICT RULES:
    1. Do not invent any new skills, experiences, or degrees.
    2. Rewrite my experience bullet points to highlight relevance to the job description using the Action + Context + Result format.
    3. Weave in matching keywords from the job description naturally.
    4. Output ONLY the optimized resume in Markdown format.
    5. KEEP FORMATTING SIMPLE: Only use headings, bullet points, and bold text. Do not include any introductory chat text.
    
    Job Description:
    {job_description}
    
    My Resume:
    {resume_text}
    """
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return completion.choices[0].message.content

def generate_cover_letter(resume_text, job_description):
    prompt = f"""
    Write a professional, authentic, and highly human-sounding cover letter based on the applicant's resume and the job description.
    
    CRITICAL INSTRUCTIONS TO AVOID SOUNDING LIKE AI:
    1. DO NOT use generic, robotic openings like "I am writing to express my interest in..." or "I am thrilled to apply for..."
    2. START with a strong, direct hook that immediately states the value the candidate brings to a specific problem or need mentioned in the job description.
    3. BANNED WORDS: Do not use words like "delve", "testament", "orchestrated", "synergy", "pivotal", "embark", or "furthermore". Keep the vocabulary natural and conversational but professional.
    4. Show, don't just tell. Connect 1 or 2 specific achievements from the resume directly to the needs of the job.
    5. Keep it concise (max 3-4 short paragraphs).
    6. Output ONLY the cover letter text. Include standard [Insert Name/Date/Company] brackets if specific info is missing. Do not include any introductory chat text.
    
    Job Description:
    {job_description}
    
    My Resume:
    {resume_text}
    """
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return completion.choices[0].message.content

# --- PDF Generators ---
def create_pdf_from_markdown(md_text):
    clean_md = sanitize_text_for_fpdf(md_text)
    html_content = markdown.markdown(clean_md)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    pdf.write_html(html_content)
    return bytes(pdf.output())

def create_pdf_from_text(plain_text):
    clean_text = sanitize_text_for_fpdf(plain_text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(0, 6, clean_text)
    return bytes(pdf.output())

# --- Word (.docx) Generators ---
def _add_formatted_runs(paragraph, text):
    """Helper to apply bold formatting inside Word paragraphs."""
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            paragraph.add_run(part[2:-2]).bold = True
        else:
            paragraph.add_run(part)

def create_docx_from_markdown(md_text):
    doc = docx.Document()
    for line in md_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('# '):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith('## '):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith('### '):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
            p = doc.add_paragraph(style='List Bullet')
            _add_formatted_runs(p, line[2:].strip())
        else:
            p = doc.add_paragraph()
            _add_formatted_runs(p, line)
            
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def create_docx_from_text(plain_text):
    doc = docx.Document()
    for paragraph in plain_text.split('\n\n'):
        if paragraph.strip():
            doc.add_paragraph(paragraph.strip())
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


# --- Streamlit Frontend ---
st.set_page_config(page_title="AI CV Optimizer", page_icon="📄", layout="wide")

st.title("CV Optimizer & Cover Letter Generator")
st.markdown("Powered by the ultra-fast **Groq API**")

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
            original_score = calculate_ats_score(raw_resume_text, job_desc)
            st.info(f"Baseline ATS Match Score: **{original_score}%**")
            
            try:
                # AI Generation
                with st.spinner("AI is analyzing keywords and rewriting your CV..."):
                    optimized_markdown = optimize_resume(raw_resume_text, job_desc)
                    new_score = calculate_ats_score(optimized_markdown, job_desc)
                    st.success(f"New ATS Match Score: **{new_score}%** (An improvement of {round(new_score - original_score, 2)}%)")
                
                with st.spinner("Drafting your Cover Letter..."):
                    cover_letter_text = generate_cover_letter(raw_resume_text, job_desc)
                
                st.markdown("---")
                
                # UI Tabs
                tab1, tab2 = st.tabs(["📝 Tailored CV", "✉️ Professional Cover Letter"])
                
                with tab1:
                    st.markdown(optimized_markdown)
                    
                    # Layout download buttons side-by-side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="⬇️ Download CV as PDF",
                            data=create_pdf_from_markdown(optimized_markdown),
                            file_name="Optimized_ATS_Resume.pdf",
                            mime="application/pdf"
                        )
                    with col2:
                        st.download_button(
                            label="⬇️ Download CV as Word (.docx)",
                            data=create_docx_from_markdown(optimized_markdown),
                            file_name="Optimized_ATS_Resume.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        
                with tab2:
                    st.markdown(cover_letter_text)
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.download_button(
                            label="⬇️ Download Cover Letter as PDF",
                            data=create_pdf_from_text(cover_letter_text),
                            file_name="Cover_Letter.pdf",
                            mime="application/pdf"
                        )
                    with col4:
                        st.download_button(
                            label="⬇️ Download Cover Letter as Word (.docx)",
                            data=create_docx_from_text(cover_letter_text),
                            file_name="Cover_Letter.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        
            except Exception as e:
                st.error(f"An error occurred during AI processing: {e}")
