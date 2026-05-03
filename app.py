from flask import Flask, request, jsonify
import google.generativeai as genai
from pinecone import Pinecone
import os
from langdetect import detect
import requests
import time

# Configure the Gemini API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("farmer-chatbot")

# Hugging Face API Configuration
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL_ID = "intfloat/multilingual-e5-base"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Gemini Model Configuration (configurable via env var)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Request timeout configuration
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Grape Master chatbot is running!"})

def get_language_from_query(query):
    """Detect the language of the user's query"""
    try:
        lang = detect(query)
        return lang
    except:
        return 'en'  # Default to English if detection fails

def get_embeddings_from_hf(text):
    """Get embeddings from Hugging Face API using intfloat/multilingual-e5-base with retry logic"""
    try:
        print(f"[DEBUG] Calling Hugging Face API for embeddings...")
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": f"query: {text}"}
        
        # Retry logic for robustness
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    HF_API_URL, 
                    headers=headers, 
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result[0] if isinstance(result, list) else result
                    print(f"[DEBUG] Embedding size: {len(embedding)}")
                    return embedding
                
                elif response.status_code == 503:
                    # Model is loading, retry
                    if attempt < MAX_RETRIES - 1:
                        print(f"[DEBUG] Model loading, retrying in {RETRY_DELAY}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"[ERROR] Hugging Face model still loading after {MAX_RETRIES} attempts")
                        return None
                else:
                    print(f"[ERROR] Hugging Face API error: {response.status_code} - {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"[ERROR] Request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
            except requests.exceptions.ConnectionError as e:
                print(f"[ERROR] Connection error: {e} (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
        
        return None
        
    except Exception as e:
        print(f"[ERROR] Unexpected error getting embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_context_from_pinecone(query, k=4):
    """Query Pinecone to get top k relevant results"""
    try:
        # Get embedding from Hugging Face API
        query_embedding = get_embeddings_from_hf(query)
        
        if not query_embedding:
            print("[ERROR] Failed to get embedding from HF API")
            return ""
        
        # Validate embedding dimension
        if len(query_embedding) != 768:
            print(f"[WARNING] Unexpected embedding dimension: {len(query_embedding)}, expected 768")
        
        # Query Pinecone for top k results
        print(f"[DEBUG] Querying Pinecone for top {k} results...")
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        if not results or 'matches' not in results:
            print("[WARNING] No results returned from Pinecone")
            return ""
        
        print(f"[DEBUG] Pinecone returned {len(results['matches'])} matches")
        
        # Extract context from results
        context = ""
        for i, match in enumerate(results['matches']):
            metadata = match.get('metadata', {})
            question = metadata.get('question', '')
            answer = metadata.get('answer', '')
            score = match.get('score', 0)
            print(f"[DEBUG] Match {i+1}: Score={score:.4f}")
            print(f"[DEBUG]   Q: {question[:100]}..." if len(question) > 100 else f"[DEBUG]   Q: {question}")
            print(f"[DEBUG]   A: {answer[:150]}..." if len(answer) > 150 else f"[DEBUG]   A: {answer}")
            context += f"Q: {question}\nA: {answer}\n\n"
        
        print(f"[DEBUG] Final context length: {len(context)} characters")
        print(f"[DEBUG] Context preview (first 300 chars):\n{context[:300]}")
        
        return context
    except Exception as e:
        print(f"[ERROR] Error querying Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return ""

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for processing user queries"""
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400
        
        user_query = data.get('query', '').strip()
        user_lang = data.get('language', '').strip()

        if not user_query:
            return jsonify({"error": "Query is required and cannot be empty"}), 400

        print(f"[DEBUG] Received query: {user_query}")
        
        # Detect the language of the query if not provided
        if not user_lang:
            try:
                user_lang = get_language_from_query(user_query)
                print(f"[DEBUG] Detected language: {user_lang}")
            except Exception as e:
                print(f"[WARNING] Language detection failed: {e}, defaulting to English")
                user_lang = 'en'
        
        # Get relevant context from Pinecone
        print("[DEBUG] Getting context from Pinecone...")
        context = get_context_from_pinecone(user_query, k=4)
        
        if not context:
            print("[WARNING] No relevant context found in Pinecone, proceeding with Gemini")
            context = "No specific knowledge base context available."
        
        print(f"[DEBUG] Context retrieved: {len(context)} characters")
        
        # Generate response using Gemini
        print(f"[DEBUG] Generating response with Gemini ({GEMINI_MODEL})...")
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""You are a helpful assistant for Grape Master (a farming and agriculture chatbot).
Answer in the SAME language as the user's question — detect it from the query text itself.

Knowledge Base Context:
{context}

User Question: {user_query}

Provide a helpful and accurate answer based on the context above. If the context doesn't contain relevant information, provide your best general knowledge answer."""
        
        print(f"[DEBUG] Generating Gemini response...")
        response = gemini_model.generate_content(prompt)
        
        if not response or not response.text:
            return jsonify({"error": "Failed to generate response from Gemini"}), 500
        
        print(f"[DEBUG] Response generated successfully")
        
        return jsonify({
            "response": response.text,
            "detected_language": user_lang,
            "context_found": len(context) > 0
        })

    except ValueError as ve:
        error_msg = f"Invalid request: {str(ve)}"
        print(f"[ERROR] ValueError: {error_msg}")
        return jsonify({"error": error_msg}), 400
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Exception occurred: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {error_msg}"}), 500

if __name__ == '__main__':
    # Validate required environment variables
    required_env_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY", "HF_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing_vars)}")
        print("[ERROR] Please set the following environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        exit(1)
    
    print("[INFO] All required environment variables are set")
    print("[INFO] Starting Grape Master chatbot server...")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
