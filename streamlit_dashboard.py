#!/usr/bin/env python3
"""
Streamlit Dashboard for Text Cleaner + Summarizer
"""

import streamlit as st
import re
import nltk
import os
import json
import requests
from urllib.parse import urlparse
from typing import Optional, Tuple
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup

# Initialize session state
if 'cleaned_text' not in st.session_state:
    st.session_state.cleaned_text = ""
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

def is_url(string: str) -> bool:
    """Check if the string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def fetch_url_content(url: str) -> Tuple[str, Optional[str]]:
    """
    Fetch content from a URL.
    
    Args:
        url (str): The URL to fetch
        
    Returns:
        Tuple[str, Optional[str]]: Content and error message (if any)
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        
        if 'text/html' in content_type:
            # Extract text from HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            text = soup.get_text()
            return text, None
        else:
            # Assume plain text
            return response.text, None
            
    except requests.RequestException as e:
        return "", f"Error fetching URL: {str(e)}"
    except Exception as e:
        return "", f"Error processing URL content: {str(e)}"

def extract_text_from_file(file) -> Tuple[str, Optional[str]]:
    """
    Extract text from uploaded file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Tuple[str, Optional[str]]: Extracted text and error message (if any)
    """
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension in ['.txt', '.md']:
            # Plain text and markdown files
            content = file.read().decode('utf-8')
            return content, None
                
        elif file_extension == '.json':
            # JSON files
            content = file.read().decode('utf-8')
            data = json.loads(content)
            # Convert JSON to string representation
            return json.dumps(data, indent=2), None
                
        elif file_extension == '.html':
            # HTML files
            content = file.read().decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            text = soup.get_text()
            return text, None
            
        elif file_extension == '.pdf':
            # PDF files
            # Save file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(file.getvalue())
            
            reader = PdfReader("temp.pdf")
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Remove temporary file
            os.remove("temp.pdf")
            return text, None
            
        elif file_extension == '.docx':
            # Word documents
            # Save file temporarily
            with open("temp.docx", "wb") as f:
                f.write(file.getvalue())
            
            doc = Document("temp.docx")
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Remove temporary file
            os.remove("temp.docx")
            return text, None
            
        else:
            # Unsupported file type, try reading as plain text
            content = file.read().decode('utf-8')
            return content, None
                
    except Exception as e:
        return "", f"Error reading file: {str(e)}"

def clean_text(text: str) -> str:
    """
    Clean the input text by removing ads, special characters, and stopwords.
    
    Args:
        text (str): The input text to clean
        
    Returns:
        str: The cleaned text
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters and digits (optional, can be adjusted)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    return ' '.join(filtered_text)

def get_gemini_response(text: str, api_key: str, prompt: str, max_length: int, language: str) -> Optional[str]:
    """
    Get response from Google Gemini API.
    
    Args:
        text (str): The text to process
        api_key (str): Gemini API key
        prompt (str): The prompt to send to Gemini
        max_length (int): Maximum length of the response
        language (str): Language for the response
        
    Returns:
        Optional[str]: The response or None if failed
    """
    try:
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Use gemini-1.5-flash as it's widely available and fast
        model_name = "gemini-1.5-flash"
        
        # Set up the model
        model = genai.GenerativeModel(model_name)
        
        # Create the full prompt with language and length instructions
        language_instruction = f"Please respond in {language} language. "
        length_instruction = f"Please limit your response to approximately {max_length} words. "
        
        full_prompt = f"{language_instruction}{length_instruction}{prompt}\n\n{text}"
        
        # Generate content
        response = model.generate_content(full_prompt)
        
        return response.text
    except Exception as e:
        st.error(f"Error getting response from Gemini: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Text Cleaner + Summarizer",
        page_icon="🧹",
        layout="wide"
    )
    
    # Welcome message
    st.title("🧹 Text Cleaner + Summarizer")
    st.markdown("""
    Welcome! This friendly tool cleans noisy text and processes it using Google Gemini.
    Simply upload a file, enter text, or provide a URL to get started.
    """)
    
    # API Key input with environment variable support
    default_api_key = os.getenv("GEMINI_API_KEY", "")
    st.session_state.api_key = st.text_input(
        "Google Gemini API Key", 
        value=st.session_state.api_key or default_api_key, 
        type="password",
        help="Enter your Google Gemini API key to enable text processing. You can also set GEMINI_API_KEY environment variable."
    )
    
    if not st.session_state.api_key:
        st.warning("Please enter your Google Gemini API key to use all features. You can set it as an environment variable named GEMINI_API_KEY.")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📁 File Upload", "🌐 URL Input"])
    
    with tab1:
        st.header("Text Input")
        st.markdown("---")
        user_text = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Paste your text here..."
        )
        
        if st.button("Clean Text", key="clean_text_input"):
            if user_text:
                with st.spinner("Cleaning text..."):
                    cleaned = clean_text(user_text)
                    st.session_state.cleaned_text = cleaned
                    st.success("Text cleaned successfully!")
            else:
                st.warning("Please enter some text first.")
    
    with tab2:
        st.header("File Upload")
        st.markdown("---")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["txt", "md", "json", "html", "pdf", "docx"],
            help="Supported formats: TXT, MD, JSON, HTML, PDF, DOCX"
        )
        
        if uploaded_file is not None:
            st.info(f"File uploaded: {uploaded_file.name}")
            
            if st.button("Extract and Clean Text", key="clean_file"):
                with st.spinner("Extracting and cleaning text..."):
                    text, error = extract_text_from_file(uploaded_file)
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        cleaned = clean_text(text)
                        st.session_state.cleaned_text = cleaned
                        st.success("Text extracted and cleaned successfully!")
    
    with tab3:
        st.header("URL Input")
        st.markdown("---")
        url = st.text_input(
            "Enter URL:",
            placeholder="https://example.com/article"
        )
        
        if st.button("Fetch and Clean Text", key="clean_url"):
            if url:
                if is_url(url):
                    with st.spinner("Fetching and cleaning text..."):
                        text, error = fetch_url_content(url)
                        if error:
                            st.error(f"Error: {error}")
                        else:
                            cleaned = clean_text(text)
                            st.session_state.cleaned_text = cleaned
                            st.success("Text fetched and cleaned successfully!")
                else:
                    st.error("Please enter a valid URL.")
            else:
                st.warning("Please enter a URL first.")
    
    # Display results in the requested layout
    if st.session_state.cleaned_text:
        # Display cleaned text at the top
        st.header("Cleaned Text")
        st.markdown("---")
        st.text_area(
            "Cleaned text:",
            value=st.session_state.cleaned_text,
            height=200,
            key="cleaned_text_area"
        )
        
        # Download options for cleaned text
        st.subheader("Save Cleaned Text")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="Download as TXT",
                data=st.session_state.cleaned_text,
                file_name="cleaned_text.txt",
                mime="text/plain"
            )
        with col2:
            st.download_button(
                label="Download as MD",
                data=st.session_state.cleaned_text,
                file_name="cleaned_text.md",
                mime="text/markdown"
            )
        with col3:
                # Create a simple PDF version
                pdf_content = f"Cleaned Text\n\n{st.session_state.cleaned_text}"
                st.download_button(
                    label="Download as PDF",
                    data=pdf_content,
                    file_name="cleaned_text.pdf",
                    mime="application/pdf"
                )
        
        # Clear button for cleaned text only (placed near the cleaned text section)
        if st.button("Clear Cleaned Text"):
            st.session_state.cleaned_text = ""
            st.session_state.processed_text = ""
            st.experimental_rerun()
        
        st.markdown("---")
        
        # Processing options below cleaned text
        st.header("Process Cleaned Text")
        st.markdown("---")
        
        # Additional options for response customization
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider("Maximum response length (words)", min_value=50, max_value=500, value=150, step=50)
        with col2:
            language = st.selectbox(
                "Response language",
                ["English", "Hindi", "Spanish", "French", "German", "Chinese", "Japanese", "Korean"],
                index=0
            )
        
        processing_option = st.selectbox(
            "Choose processing option:",
            [
                "Summarize text",
                "Extract key points",
                "Rewrite in formal tone",
                "Translate to another language",
                "Custom prompt"
            ]
        )
        
        custom_prompt = ""
        if processing_option == "Summarize text":
            prompt = "Please summarize the following text:"
        elif processing_option == "Extract key points":
            prompt = "Please extract the key points from the following text as a bullet list:"
        elif processing_option == "Rewrite in formal tone":
            prompt = "Please rewrite the following text in a more formal tone:"
        elif processing_option == "Translate to another language":
            prompt = "Please translate the following text:"
        elif processing_option == "Custom prompt":
            custom_prompt = st.text_area(
                "Enter your custom prompt:",
                placeholder="Enter your custom instruction for the text..."
            )
            prompt = custom_prompt
        
        if st.button("Process with Gemini", key="process_gemini"):
            if not st.session_state.api_key:
                st.error("Please enter your Google Gemini API key first.")
            elif processing_option == "Custom prompt" and not custom_prompt:
                st.error("Please enter a custom prompt.")
            else:
                with st.spinner("Processing with Google Gemini..."):
                    response = get_gemini_response(
                        st.session_state.cleaned_text, 
                        st.session_state.api_key, 
                        prompt,
                        max_length,
                        language
                    )
                    if response:
                        st.session_state.processed_text = response
                        st.success("Text processed successfully!")
        
        # Display processed text below the processing options
        if st.session_state.processed_text:
            st.header("Processed Text")
            st.markdown("---")
            st.text_area(
                "Result:",
                value=st.session_state.processed_text,
                height=300,
                key="processed_text_area"
            )
            
            # Download options for processed text
            st.subheader("Save Processed Text")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="Download as TXT",
                    data=st.session_state.processed_text,
                    file_name="processed_text.txt",
                    mime="text/plain"
                )
            with col2:
                st.download_button(
                    label="Download as MD",
                    data=st.session_state.processed_text,
                    file_name="processed_text.md",
                    mime="text/markdown"
                )
            with col3:
                # Create a simple PDF version
                pdf_content = f"Processed Text\n\n{st.session_state.processed_text}"
                st.download_button(
                    label="Download as PDF",
                    data=pdf_content,
                    file_name="processed_text.pdf",
                    mime="application/pdf"
                )
            
            # Clear button for all results (placed near the processed text section)
            if st.button("Clear All Results"):
                st.session_state.cleaned_text = ""
                st.session_state.processed_text = ""
                st.experimental_rerun()

if __name__ == "__main__":
    main()