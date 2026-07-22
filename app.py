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
    """Drafts the initial ATS optimized resume."""
    prompt = f"""
    You are an expert ATS resume writer. 
    I will provide my current resume and a job description.
    STRICT RULES:
    1. Do not invent any new skills, experiences, or degrees.
    2. Rewrite my experience bullet points to highlight relevance to the job description.
    3. Output ONLY the optimized resume in Markdown format. Keep formatting simple.
    
    Job Description:\n{job_description}\n\nMy Resume:\n{resume_text}
    """
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return completion.choices[0].message.content

def generate_cover_letter(resume_text, job_description):
    """Drafts the initial human-sounding cover letter."""
    prompt = f"""
    Write a professional, authentic cover letter based on the applicant's resume and the job description.
    1. START with a strong, direct hook stating the value the candidate brings.
    2. BANNED WORDS: "delve", "testament", "orchestrated", "synergy".
    3. Output ONLY the cover letter text.
    
    Job Description:\n{job_description}\n\nMy Resume:\n{resume_text}
    """
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return completion.choices[0].message.content

def finalize_documents(draft_resume, draft_cl, job_description):
    """Acts as an Expert Recruiter to review, apply XYZ formula, and rewrite the final versions."""
    prompt = f"""
    You are an elite Executive Technical Recruiter. I am providing a Draft Resume and Draft Cover Letter. 

    Your task is to:
    1. Critique the drafts using Google's XYZ formula ("Accomplished [X] as measured by [Y], by doing [Z]").
    2. REWRITE BOTH documents to incorporate your critique, creating the final, polished, highly logical versions.
    3. CRITICAL: Do NOT invent fake metrics. If a metric is needed to satisfy the XYZ formula but wasn't in the original text, insert a clear placeholder like [Insert %, e.g., 15%] so the user knows exactly what to manually add.

    You MUST output your response exactly in this format using these strict delimiters:

    ===REVIEW===
    (Your concise explanation of the improvements made and which placeholders the user needs to fill in)
    ===FINAL_RESUME===
    (The finalized markdown resume)
    ===FINAL_COVER_LETTER===
    (The finalized cover letter)
    
    Job Description:
    {job_description}
    
    Draft Resume:
    {draft_resume}
    
    Draft Cover Letter:
    {draft_cl}
    """
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    
    response_text = completion.choices[0].message.content
    
    # Parse the response using regex to separate the sections based on our delimiters
    review_match = re.search(r'===REVIEW===(.*?)===FINAL_RESUME===', response_text, re.DOTALL)
    resume_match = re.search(r'===FINAL_RESUME===(.*?)===FINAL_COVER_LETTER===', response_text, re.DOTALL)
    cl_match = re.search(r'===FINAL_COVER_LETTER===(.*)', response_text, re.DOTALL)
    
    review = review_match.group(1).strip() if review_match else "Review parsing failed, but final documents were generated."
    final_resume = resume_match.group(1).strip() if resume_match else draft_resume
    final_cl = cl_match.group(1).strip() if cl_match else draft_cl
    
    return review, final_resume, final_cl

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
                # 1. AI Generation - Drafts
                with st.spinner("Agent 1: Drafting initial ATS resume..."):
                    draft_markdown = optimize_resume(raw_resume_text, job_desc)
                
                with st.spinner("Agent 2: Drafting human-sounding Cover Letter..."):
                    draft_cl = generate_cover_letter(raw_resume_text, job_desc)
                
                # 2. AI Expert Review & Finalization
                with st.spinner("Agent 3: Expert AI is applying the XYZ formula and finalizing documents..."):
                    review_feedback, final_resume, final_cover_letter = finalize_documents(draft_markdown, draft_cl, job_desc)
                
                # 3. Calculate Final Score
                new_score = calculate_ats_score(final_resume, job_desc)
                st.success(f"Final ATS Match Score: **{new_score}%** (An improvement of {round(new_score - original_score, 2)}%)")
                
                st.markdown("---")
                
                # UI Tabs
                tab1, tab2, tab3 = st.tabs(["📝 Final Tailored CV", "✉️ Final Cover Letter", "🔎 What The AI Improved"])
                
                with tab1:
                    st.info("💡 **Tip:** Look for bracketed placeholders like `[Insert %]` and manually replace them with real numbers in your downloaded file!")
                    st.markdown(final_resume)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="⬇️ Download CV as PDF",
                            data=create_pdf_from_markdown(final_resume),
                            file_name="Optimized_ATS_Resume.pdf",
                            mime="application/pdf"
                        )
                    with col2:
                        st.download_button(
                            label="⬇️ Download CV as Word (.docx)",
                            data=create_docx_from_markdown(final_resume),
                            file_name="Optimized_ATS_Resume.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        
                with tab2:
                    st.markdown(final_cover_letter)
                    col3, col4 = st.columns(2)
                    with col3:
                        st.download_button(
                            label="⬇️ Download Cover Letter as PDF",
                            data=create_pdf_from_text(final_cover_letter),
                            file_name="Cover_Letter.pdf",
                            mime="application/pdf"
                        )
                    with col4:
                        st.download_button(
                            label="⬇️ Download Cover Letter as Word (.docx)",
                            data=create_docx_from_text(final_cover_letter),
                            file_name="Cover_Letter.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                
                with tab3:
                    st.markdown("### The Expert Recruiter's Notes")
                    st.markdown(review_feedback)
                        
            except Exception as e:
                st.error(f"An error occurred during AI processing: {e}")
