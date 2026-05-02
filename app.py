from flask import Flask, request, jsonify
import google.generativeai as genai
from pinecone import Pinecone
from huggingface_hub import InferenceClient
import os
from langdetect import detect

# Configure the Gemini API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("farmer-chatbot")

# Initialize Hugging Face Inference Client
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ.get("HUGGINGFACE_API_KEY")
)

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
    """Get embeddings from Hugging Face Inference Client"""
    try:
        print(f"[DEBUG] Calling HF InferenceClient for embeddings...")
        embedding = hf_client.feature_extraction(
            f"query: {text}",
            model="intfloat/multilingual-e5-base"
        )
        # Returns a numpy array — convert to list for Pinecone
        embedding_list = embedding.tolist()
        
        # If shape is [tokens, dims], mean pool to [dims]
        if isinstance(embedding_list[0], list):
            embedding_list = [
                sum(token[i] for token in embedding_list) / len(embedding_list)
                for i in range(len(embedding_list[0]))
            ]
        
        print(f"[DEBUG] Embedding size: {len(embedding_list)}")
        return embedding_list
    except Exception as e:
        print(f"[ERROR] Error getting embeddings: {e}")
        return None

def get_context_from_pinecone(query, k=4):
    """Query Pinecone to get top 4 relevant results"""
    try:
        # Get embedding from Hugging Face API
        query_embedding = get_embeddings_from_hf(query)
        
        if not query_embedding:
            print("[ERROR] Failed to get embedding from HF API")
            return ""
        
        # Query Pinecone for top k results
        print(f"[DEBUG] Querying Pinecone for top {k} results...")
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
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
    data = request.json
    user_query = data.get('query')
    user_lang = data.get('language', '')  # User's preferred language

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        print(f"[DEBUG] Received query: {user_query}")
        
        # Detect the language of the query if not provided
        if not user_lang:
            user_lang = get_language_from_query(user_query)
            print(f"[DEBUG] Detected language: {user_lang}")
        
        # Get relevant context from Pinecone
        print("[DEBUG] Getting context from Pinecone...")
        context = get_context_from_pinecone(user_query, k=4)
        print(f"[DEBUG] Context retrieved: {len(context)} characters")
        
        # Generate response using Gemini 2.5 Flash
        print("[DEBUG] Generating response with Gemini...")
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""You are a helpful assistant for Grape Master (a farming and agriculture chatbot).
Answer in the SAME language as the user's question — detect it from the query text itself.

Knowledge Base Context:
{context}

User Question: {user_query}

Provide a helpful and accurate answer based on the context above."""
        
        print(f"[DEBUG] Prompt being sent to Gemini:")
        print(f"[DEBUG] ==========================================")
        print(f"[DEBUG] {prompt[:500]}...")
        print(f"[DEBUG] ==========================================")
        
        response = gemini_model.generate_content(prompt)
        print(f"[DEBUG] Response generated successfully: {response.text[:200]}...")
        
        return jsonify({
            "response": response.text,
            "detected_language": user_lang
        })

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Exception occurred: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {error_msg}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
