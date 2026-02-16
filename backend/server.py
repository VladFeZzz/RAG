import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
import ollama
from groq import Groq
from dotenv import load_dotenv
import pypdf

app = Flask(__name__)
CORS(app)

load_dotenv(dotenv_path="../.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in .env file")
    exit()
else:
    print("Groq API key loaded successfully")

groq_client = Groq(api_key=GROQ_API_KEY)
chroma_client = chromadb.PersistentClient(path="./my_base")
collection = chroma_client.get_or_create_collection(name="Curse_docs")


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get('message')
    print(f"\nUser question: {user_question}")

    try:
        query_embed = ollama.embeddings(model="nomic-embed-text", prompt=user_question)['embedding']
        results = collection.query(query_embeddings=[query_embed], n_results=1)
        
        context = ""
        if results['documents'] and results['documents'][0]:
            context = results['documents'][0][0]
            print(f"Found context: {context[:100]}...")
        else:
            print("No context found in database")
            
    except Exception as e:
        print(f"Search error: {e}")
        context = ""

    system_prompt = f"""
    You are a helpful music expert assistant. 
    Answer the user's question based ONLY on the provided CONTEXT.
    If the answer is not in the context, politely say that you don't have this information in your database.
    Respond in Ukrainian.

    CONTEXT: 
    {context}
    """
    
    print("Generating response...")
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
        )
        bot_response = completion.choices[0].message.content
        print("Response generated successfully")
        
        return jsonify({"response": bot_response})
        
    except Exception as e:
        print(f"Groq generation error: {e}")
        return jsonify({"response": "Вибачте, сталася помилка при генерації відповіді."}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Файл не знайдено у запиті"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Файл не вибрано"}), 400

    print(f"\nProcessing file: {file.filename}")
    
    try:
        pdf_reader = pypdf.PdfReader(file)
        text_content = ""
        
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n\n"
        
        chunks = text_content.split('\n\n')
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 20]
        
        print(f"Text split into {len(chunks)} chunks. Starting indexing...")
        
        for i, chunk in enumerate(chunks):
            embed = ollama.embeddings(model="nomic-embed-text", prompt=chunk)['embedding']
            chunk_id = f"{file.filename}_chunk_{i}"
            
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embed],
                documents=[chunk]
            )
            
        print(f"File {file.filename} successfully added to knowledge base")
        return jsonify({"status": "success", "message": "Файл успішно оброблено та збережено!"})
        
    except Exception as e:
        print(f"PDF processing error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\nServer started at http://127.0.0.1:5000")
    print("Press CTRL+C to stop\n")
    app.run(debug=True, port=5000)