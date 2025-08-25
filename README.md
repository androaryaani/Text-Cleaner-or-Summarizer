# Text Cleaner + Summarizer

A Python CLI tool and Streamlit dashboard that cleans noisy text (ads, special characters, stopwords) and generates concise summaries using Google Gemini API. Supports multiple input formats including text files, PDFs, Word documents, HTML files, JSON files, and URLs.

## Features

- **Multi-format Input Support**:
  - Plain text files (.txt)
  - Markdown files (.md)
  - JSON files (.json)
  - HTML files (.html)
  - PDF files (.pdf)
  - Word documents (.docx)
  - URLs
  - Direct text input via stdin or dashboard

- **Advanced Text Cleaning**:
  - Removes URLs, email addresses, HTML tags
  - Eliminates special characters and extra whitespace
  - Converts text to lowercase
  - Filters out common English stopwords

- **Google Gemini Integration**:
  - Uses Google's Gemini-1.5-Flash model for text processing
  - Multiple processing options:
    * Summarize text
    * Extract key points
    * Rewrite in formal tone
    * Translate to multiple languages
    * Custom prompt support
  - Configurable response length and language

- **Multiple Output Formats**:
  - Save results as TXT, MD, or PDF files
  - Download both cleaned and processed text

- **Environment Variable Support**:
  - Set API key as environment variable for easier management

## Requirements

- Python 3.7+
- NLTK
- Google Generative AI
- Requests
- BeautifulSoup4
- PyPDF2
- python-docx
- Streamlit

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLTK data (happens automatically on first run):
   ```bash
   python text_cleaner_summarizer.py --help
   ```

## Environment Setup

To use the Google Gemini API, you need to set up an API key:

1. Get your Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
2. Set it as an environment variable:
   ```bash
   # On Windows (Command Prompt)
   set GEMINI_API_KEY=your_api_key_here
   
   # On Windows (PowerShell)
   $env:GEMINI_API_KEY="your_api_key_here"
   
   # On macOS/Linux
   export GEMINI_API_KEY=your_api_key_here
   ```
3. Alternatively, you can create a `.env` file by copying `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file to add your API key.

## Command-Line Usage

### Basic Usage

```bash
# Summarize a text file
python text_cleaner_summarizer.py input.txt -k YOUR_API_KEY

# Summarize content from a URL
python text_cleaner_summarizer.py https://example.com/article -k YOUR_API_KEY

# Summarize text from stdin
echo "Your noisy text here" | python text_cleaner_summarizer.py - -k YOUR_API_KEY
```

### Options

- `-o, --output FILE`: Save summary to a file instead of printing to stdout
- `-k, --api-key KEY`: Google Gemini API key (can also use GEMINI_API_KEY environment variable)
- `-l, --length NUM`: Maximum summary length in words (default: 150)

### Supported Input Formats

1. **Text Files** (`.txt`, `.md`):
   ```bash
   python text_cleaner_summarizer.py article.txt -k YOUR_API_KEY
   ```

2. **JSON Files** (`.json`):
   ```bash
   python text_cleaner_summarizer.py data.json -k YOUR_API_KEY
   ```

3. **HTML Files** (`.html`):
   ```bash
   python text_cleaner_summarizer.py webpage.html -k YOUR_API_KEY
   ```

4. **PDF Files** (`.pdf`):
   ```bash
   python text_cleaner_summarizer.py document.pdf -k YOUR_API_KEY
   ```

5. **Word Documents** (`.docx`):
   ```bash
   python text_cleaner_summarizer.py document.docx -k YOUR_API_KEY
   ```

6. **URLs**:
   ```bash
   python text_cleaner_summarizer.py https://example.com/article -k YOUR_API_KEY
   ```

7. **Direct Text Input**:
   ```bash
   echo "Your text here" | python text_cleaner_summarizer.py - -k YOUR_API_KEY
   ```

### Examples

```bash
# Using environment variable for API key
export GEMINI_API_KEY=your_api_key_here
python text_cleaner_summarizer.py article.txt

# Save summary to a file
python text_cleaner_summarizer.py article.txt -o summary.txt

# Limit summary to 100 words
python text_cleaner_summarizer.py article.txt -l 100

# Process a PDF document
python text_cleaner_summarizer.py document.pdf -k YOUR_API_KEY

# Summarize content from a webpage
python text_cleaner_summarizer.py https://example.com/blog-post -k YOUR_API_KEY

# Read from stdin
cat article.txt | python text_cleaner_summarizer.py - -k YOUR_API_KEY
```

## Streamlit Dashboard Usage

### Running the Dashboard

```bash
streamlit run streamlit_dashboard.py
```

### Dashboard Features

1. **Multiple Input Methods**:
   - Text input: Paste text directly into the dashboard
   - File upload: Upload TXT, MD, JSON, HTML, PDF, or DOCX files
   - URL input: Enter a web URL to fetch and process content

2. **Text Processing Options**:
   - Summarize text
   - Extract key points
   - Rewrite in formal tone
   - Translate to multiple languages
   - Custom prompt support

3. **Customization Options**:
   - Adjust response length (50-500 words)
   - Select response language (English, Hindi, Spanish, French, German, Chinese, Japanese, Korean)
   - Environment variable support for API key

4. **Output Options**:
   - Download cleaned text as TXT, MD, or PDF
   - Download processed text as TXT, MD, or PDF
   - Clear results with one click

## How It Works

1. **Input Processing**:
   - Automatically detects input type (file, URL, text input)
   - Extracts text content from various formats
   - Handles web content fetching with proper headers

2. **Text Cleaning**:
   - Removes URLs, email addresses, and HTML tags
   - Eliminates special characters and extra whitespace
   - Converts text to lowercase
   - Filters out common English stopwords

3. **Text Processing**:
   - Sends cleaned text to Google Gemini API
   - Supports customizable response length and language
   - Multiple processing options for different use cases

## License

MIT License