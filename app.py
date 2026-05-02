from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import google.generativeai as genai
import os

# Configure the Gemini API key
# Make sure to set the GOOGLE_API_KEY environment variable
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

app = Flask(__name__)

# Load the data and create embeddings
@app.before_first_request
def load_model_and_data():
    global model, df, embeddings
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    df = pd.read_csv('Cleaned_CropMaster_QA.csv')
    # Ensure the 'Answer' column is of type string
    df['Answer'] = df['Answer'].astype(str)
    embeddings = model.encode(df['Question'].tolist(), convert_to_tensor=True)
    print("Model and data loaded.")

def find_top_k_matches(query, k=4):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_k_indices = np.argpartition(-cos_scores, range(k))[:k]
    return df.iloc[top_k_indices]

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    user_lang = data.get('language', 'en') # Default to English

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # Find relevant context
    top_matches = find_top_k_matches(user_query)
    context = "\n".join([f"Q: {row['Question']}\nA: {row['Answer']}" for index, row in top_matches.iterrows()])

    # Generate response using Gemini
    try:
        gemini_model = genai.GenerativeModel('gemini-pro')
        prompt = f"""You are a helpful assistant for Grape Master. 
        Answer the user's question based on the following context.
        The user is asking in {user_lang}. Your response should be in {user_lang}.

        Context:
        {context}

        User Question: {user_query}

        Answer:"""
        
        response = gemini_model.generate_content(prompt)
        
        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
