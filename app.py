from flask import Flask, request, jsonify
import google.generativeai as genai
from pinecone import Pinecone
import requests
import os
from langdetect import detect

# Configure the Gemini API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("farmer-chatbot")

# Hugging Face API for embeddings
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
HF_INFERENCE_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction"
HF_MODEL_ID = "intfloat/multilingual-e5-base"

app = Flask(__name__)

def get_language_from_query(query):
    """Detect the language of the user's query"""
    try:
        lang = detect(query)
        return lang
    except:
        return 'en'  # Default to English if detection fails

def get_embeddings_from_hf(text):
    """Get embeddings from Hugging Face Inference API"""
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {
            "inputs": text,
            "options": {"use_cache": False}
        }
        
        response = requests.post(
            HF_INFERENCE_API_URL,
            headers=headers,
            json=payload,
            params={"model": HF_MODEL_ID}
        )
        
        if response.status_code == 200:
            embedding = response.json()
            return embedding
        else:
            print(f"HF API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None

def get_context_from_pinecone(query, k=4):
    """Query Pinecone to get top 4 relevant results"""
    try:
        # Get embedding from Hugging Face API
        query_embedding = get_embeddings_from_hf(query)
        
        if not query_embedding:
            return ""
        
        # Query Pinecone for top k results
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        # Extract context from results
        context = ""
        for match in results['matches']:
            metadata = match.get('metadata', {})
            question = metadata.get('question', '')
            answer = metadata.get('answer', '')
            context += f"Q: {question}\nA: {answer}\n\n"
        
        return context
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return ""

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    user_lang = data.get('language', '')  # User's preferred language

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Detect the language of the query if not provided
        if not user_lang:
            user_lang = get_language_from_query(user_query)
        
        # Get relevant context from Pinecone
        context = get_context_from_pinecone(user_query, k=4)
        
        # Generate response using Gemini 2.5 Flash
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""You are a helpful assistant for Grape Master (a farming and agriculture chatbot).
        Answer the user's question based on the following context from the knowledge base.
        The user is asking in the language code '{user_lang}'. Your response MUST be in the same language as the user's query.
        
        Knowledge Base Context:
        {context}
        
        User Question: {user_query}
        
        Provide a helpful and accurate answer based on the context above. If the context doesn't have relevant information, provide the best answer you can based on your knowledge."""
        
        response = gemini_model.generate_content(prompt)
        
        return jsonify({
            "response": response.text,
            "detected_language": user_lang
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
