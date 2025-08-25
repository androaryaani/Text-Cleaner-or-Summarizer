#!/usr/bin/env python3
"""
Text Cleaner + Summarizer

This script cleans noisy text (ads, special characters, stopwords) and generates
a summary using OpenAI API. It supports multiple input formats including:
- Plain text files (.txt)
- Markdown files (.md)
- JSON files (.json)
- HTML files (.html)
- PDF files (.pdf)
- Word documents (.docx)
- URLs
- Direct text input
"""

import re
import nltk
import argparse
import os
import json
import requests
from typing import Optional, Tuple
from urllib.parse import urlparse

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
from nltk.tokenize import word_tokenize, sent_tokenize


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
            from bs4 import BeautifulSoup
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


def extract_text_from_file(file_path: str) -> Tuple[str, Optional[str]]:
    """
    Extract text from various file formats.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        Tuple[str, Optional[str]]: Extracted text and error message (if any)
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in ['.txt', '.md']:
            # Plain text and markdown files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), None
                
        elif file_extension == '.json':
            # JSON files
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert JSON to string representation
                return json.dumps(data, indent=2), None
                
        elif file_extension == '.html':
            # HTML files
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            text = soup.get_text()
            return text, None
            
        elif file_extension == '.pdf':
            # PDF files
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text, None
            
        elif file_extension == '.docx':
            # Word documents
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text, None
            
        else:
            # Unsupported file type, try reading as plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), None
                
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


def get_summary(text: str, api_key: str, max_length: int = 150) -> Optional[str]:
    """
    Get summary from OpenAI API.
    
    Args:
        text (str): The text to summarize
        api_key (str): OpenAI API key
        max_length (int): Maximum length of the summary
        
    Returns:
        Optional[str]: The summary or None if failed
    """
    try:
        import openai
    except ImportError:
        print("Error: openai package not found. Install it with 'pip install openai'")
        return None
    
    # Set up OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Please summarize the following text in {max_length} words or less:\n\n{text}"}
            ],
            max_tokens=max_length * 2,  # Approximation
            temperature=0.5
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting summary from OpenAI: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Text Cleaner + Summarizer - Cleans noisy text and generates summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python text_cleaner_summarizer.py article.txt -k YOUR_API_KEY
  python text_cleaner_summarizer.py https://example.com/article -k YOUR_API_KEY
  python text_cleaner_summarizer.py document.pdf -o summary.txt -k YOUR_API_KEY
  echo "Noisy text here" | python text_cleaner_summarizer.py - -k YOUR_API_KEY
        """
    )
    parser.add_argument("input", help="Input file path, URL, or '-' for stdin")
    parser.add_argument("-o", "--output", help="Output file for summary (default: stdout)")
    parser.add_argument("-k", "--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("-l", "--length", type=int, default=150, help="Maximum summary length (words)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Provide it with -k or set OPENAI_API_KEY environment variable.")
        return 1
    
    # Read input text
    if args.input == "-":
        text = input("Enter text to clean and summarize:\n")
        input_source = "stdin"
    elif is_url(args.input):
        print(f"Fetching content from URL: {args.input}")
        text, error = fetch_url_content(args.input)
        input_source = f"URL ({args.input})"
        if error:
            print(f"Error: {error}")
            return 1
    else:
        if not os.path.exists(args.input):
            print(f"Error: File '{args.input}' not found.")
            return 1
        
        print(f"Reading content from file: {args.input}")
        text, error = extract_text_from_file(args.input)
        input_source = f"file ({args.input})"
        if error:
            print(f"Error: {error}")
            return 1
    
    if not text:
        print("Error: No content found in input.")
        return 1
    
    print(f"Processing content from {input_source}...")
    print(f"Original text length: {len(text)} characters")
    
    # Clean text
    print("Cleaning text...")
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        print("Warning: No text remaining after cleaning.")
        return 1
    
    print(f"Cleaned text length: {len(cleaned_text)} characters")
    
    # Get summary
    print("Generating summary...")
    summary = get_summary(cleaned_text, api_key, args.length)
    
    if not summary:
        print("Error: Failed to generate summary.")
        return 1
    
    # Output result
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Summary saved to '{args.output}'")
        except Exception as e:
            print(f"Error writing to file: {e}")
            return 1
    else:
        print(f"\nSummary of content from {input_source}:")
        print("=" * 50)
        print(summary)
        print("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())