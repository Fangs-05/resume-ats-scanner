import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def extract_text_from_pdf(file):
    """
    Extracts text from an uploaded PDF file using PyPDF2.
    """
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def calculate_match(resume_text, jd_text):
    """
    Calculates the match percentage using TF-IDF and Cosine Similarity.
    """
    # 1. Create the TF-IDF Vectorizer
    # This converts text into numerical vectors based on word importance
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # 2. Fit and transform the texts into TF-IDF matrices
    # We are processing two documents: the Resume and the Job Description
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    
    # 3. Calculate Cosine Similarity
    # Since we have 2 vectors (resume and JD), we calculate similarity between index 0 and 1
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # The result is a 2D array, we need the scalar value [0][0]
    match_score = cosine_sim[0][0]
    
    return match_score

# -----------------------------------------------------------------------------
# Streamlit Application Layout
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Resume ATS Scanner", page_icon="📄")
    
    st.title("📄 Resume ATS (Applicant Tracking System)")
    st.markdown("""
    Upload your resume (PDF) and paste the Job Description to see how well you match the role.
    This tool uses **TF-IDF** and **Cosine Similarity** to analyze text relevance.
    """)

    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])

    # 2. Job Description Input
    jd_text = st.text_area("Paste Job Description Here", height=200)

    # 3. Analyze Button
    if st.button("Analyze Resume"):
        if uploaded_file is not None and jd_text.strip():
            try:
                # Extract Text
                with st.spinner("Extracting text from PDF..."):
                    resume_text = extract_text_from_pdf(uploaded_file)
                
                if not resume_text.strip():
                    st.error("Could not extract text from the PDF. It might be scanned image or empty.")
                else:
                    # Calculate Score
                    score = calculate_match(resume_text, jd_text)
                    match_percentage = round(score * 100, 2)

                    # Display Results
                    st.success("Analysis Complete!")
                    
                    st.metric(label="Match Percentage", value=f"{match_percentage}%")
                    
                    # Visual Progress Bar
                    st.progress(int(match_percentage))
                    
                    # Feedback Logic
                    if match_percentage > 75:
                        st.balloons()
                        st.write("🎉 Great match! You are a strong candidate.")
                    elif match_percentage > 50:
                        st.write("👍 Good match. Consider tailoring your resume to include more keywords from the JD.")
                    else:
                        st.write("⚠️ Low match. Please update your resume with more keywords from the job description.")

                    # Expander to see extracted text (for debugging/verification)
                    with st.expander("See extracted Resume Text"):
                        st.text(resume_text)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a PDF and enter a Job Description.")

if __name__ == "__main__":
    main()
