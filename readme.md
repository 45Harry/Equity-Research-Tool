# News Research Tool 🗞️🔍

A Streamlit application that analyzes news articles using Groq's Llama 3 model and FAISS vector search.

## Features
- Process up to 3 news article URLs simultaneously
- Extract and chunk text content
- Generate embeddings using HuggingFace models
- Query articles with natural language questions
- Source attribution for answers

## Prerequisites
- Python 3.8+
- Groq API key (get it from [Groq Cloud](https://console.groq.com/keys))

## Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Create a .env file or set environment variable:
    text

    GROQ_API_KEY=your_api_key_here

## Usage
1. Run the application:
    ```bash 
   streamlit run app.py

2. In the sidebar:

   Enter 1-3 news article URLs
    
   Click "Process URLs"

3. Wait for processing to complete (embeddings generation may take 1-2 minutes)

4. Ask questions in the main input field