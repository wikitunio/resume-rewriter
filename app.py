import markdown
import pdfkit
# Initialize the client using Streamlit Secrets
api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)


def create_pdf_from_markdown(md_text):
    """Converts Markdown text to HTML, then to PDF bytes."""
    # Convert Markdown to HTML
    html_content = markdown.markdown(md_text)
    
    # Add basic styling to make it look like a resume (optional but recommended)
    styled_html = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                h1, h2, h3 {{ color: #333333; }}
                h1 {{ border-bottom: 2px solid #333333; padding-bottom: 5px; }}
                h2 {{ border-bottom: 1px solid #cccccc; padding-bottom: 3px; margin-top: 20px; }}
                ul {{ margin-bottom: 15px; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
    </html>
    """
    
    # Generate the PDF in memory (False returns the raw bytes instead of saving to disk)
    pdf_bytes = pdfkit.from_string(styled_html, False)
    return pdf_bytes




st.markdown("### Your Tailored CV")
    st.markdown(optimized_markdown)
    
    # Generate PDF bytes
    pdf_file = create_pdf_from_markdown(optimized_markdown)
    
    # Show the download button
    st.download_button(
        label="Download Optimized CV as PDF",
        data=pdf_file,
        file_name="Optimized_ATS_Resume.pdf",
        mime="application/pdf"
    )
