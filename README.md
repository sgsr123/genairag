# PDF RAG Flask Application

This is a Flask application that allows users to upload PDF files, extract text from them, and ask questions about the content using Google's Generative AI with Retrieval-Augmented Generation (RAG).

## Features

- PDF upload and text extraction
- Text chunking and embedding using Google's embedding model
- Vector storage using FAISS
- Question answering using Google's Gemini Pro model with RAG
- Conversation history preservation

## Prerequisites

- Python 3.9+
- Google AI API key

## Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd pdf-rag-flask-app
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Google API key:
   ```
   export GOOGLE_API_KEY="your-google-api-key"  # On Windows: set GOOGLE_API_KEY=your-google-api-key
   ```

5. Create the templates directory and add the index.html file:
   ```
   mkdir templates
   # Copy the index.html file to templates/
   ```

## Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. Upload a PDF file using the "Upload PDF" button.
2. Wait for the processing to complete.
3. Once processed, the "Ask Questions" section will appear.
4. Type your question in the input field and press "Ask".
5. The application will retrieve relevant information from the PDF and generate a response.

## How It Works

1. **PDF Processing**:
   - The uploaded PDF is processed using PyPDF2 to extract text.
   - The text is split into chunks using LangChain's RecursiveCharacterTextSplitter.

2. **Embedding and Indexing**:
   - Text chunks are embedded using Google's embedding model.
   - Embeddings are stored in a FAISS vector database for efficient similarity search.

3. **Question Answering**:
   - When a question is asked, the system retrieves the most relevant text chunks.
   - These chunks are sent to Google's Gemini Pro model along with the question.
   - The model generates a response based on the retrieved context.
   - Conversation history is maintained for follow-up questions.

## Project Structure

- `app.py`: Main Flask application
- `templates/index.html`: Frontend HTML template
- `uploads/`: Directory for temporarily storing uploaded files
- `requirements.txt`: List of Python dependencies

## Dependencies

- Flask: Web framework
- PyPDF2: PDF text extraction
- LangChain: Framework for building applications with LLMs
- Google Generative AI: Google's API for generative AI models
- FAISS: Vector database for similarity search
-demo test