import os
import uuid
from flask import Flask, request, render_template, jsonify
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai

app = Flask(__name__)

# Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store the conversation chain
conversation_chain = None
# Global variable to track uploaded files
uploaded_files = []


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def create_embeddings_and_db(texts):
    """Create embeddings and vector database from multiple texts."""
    # Combine all texts
    combined_text = " ".join(texts)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(combined_text)
    
    # Create embeddings and store in vector database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(chunks, embeddings)
    
    return vector_db


def create_conversation_chain(vector_db):
    """Create a conversational retrieval chain."""
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-",  # Updated model name
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # Create a memory buffer for the conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the conversation chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        verbose=True
    )
    
    return chain


def save_uploaded_file(file):
    """Save an uploaded file to the uploads folder with a unique filename."""
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)
    return file_path


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle multiple file uploads and process them."""
    global conversation_chain, uploaded_files
    
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    # Reset uploaded files list if this is a new upload session
    if request.form.get('reset', 'false') == 'true':
        uploaded_files = []
    
    all_texts = []
    processed_files = []
    
    for file in files:
        if file and file.filename.endswith('.pdf'):
            # Save the file to the uploads folder
            file_path = save_uploaded_file(file)
            
            # Extract text from PDF
            text = extract_text_from_pdf(file_path)
            
            # Add to the texts list
            all_texts.append(text)
            
            # Keep track of uploaded files
            uploaded_files.append({
                'name': file.filename,
                'path': file_path,
                'size': os.path.getsize(file_path)
            })
            
            processed_files.append(file.filename)
    
    if not all_texts:
        return jsonify({'error': 'No valid PDF files found'}), 400
    
    # Create vector database from all texts
    vector_db = create_embeddings_and_db(all_texts)
    
    # Create conversation chain
    conversation_chain = create_conversation_chain(vector_db)
    
    return jsonify({
        'success': 'Files processed successfully',
        'processed_files': processed_files,
        'total_files': len(uploaded_files)
    }), 200


@app.route('/files', methods=['GET'])
def list_files():
    """Return a list of all uploaded files."""
    global uploaded_files
    return jsonify({'files': uploaded_files}), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle questions about the uploaded PDFs."""
    global conversation_chain
    
    try:
        # Print request data for debugging
        print(f"Request Content-Type: {request.content_type}")
        print(f"Request data: {request.data}")
        
        # Parse JSON data carefully
        if request.is_json:
            data = request.json
        else:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        # Log received data
        print(f"Parsed data: {data}")
        
        # Extract question with more robust error handling
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON format'}), 400
            
        question = data.get('question', '')
        print(f"Extracted question: '{question}'")
        
        if not question or not isinstance(question, str):
            return jsonify({'error': 'No valid question provided'}), 400
        
        if not conversation_chain:
            return jsonify({'error': 'Please upload PDF files first'}), 400
        
        # Get response from the conversation chain
        print(f"Calling conversation chain with question: '{question}'")
        response = conversation_chain({'question': question})
        answer = response.get('answer', 'I could not find an answer to your question.')
        print(f"Got answer: '{answer[:50]}...'")
        
        return jsonify({'answer': answer}), 200
        
    except Exception as e:
        print(f"Error in /ask route: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/reset', methods=['POST'])
def reset_session():
    """Reset the current session, optionally deleting uploaded files."""
    global conversation_chain, uploaded_files
    
    delete_files = request.json.get('delete_files', False)
    
    # Reset conversation chain
    conversation_chain = None
    
    # Optionally delete files
    if delete_files:
        for file_info in uploaded_files:
            if os.path.exists(file_info['path']):
                os.remove(file_info['path'])
    
    # Reset uploaded files list
    uploaded_files = []
    
    return jsonify({'success': 'Session reset successfully'}), 200


if __name__ == '__main__':
    app.run(debug=True)