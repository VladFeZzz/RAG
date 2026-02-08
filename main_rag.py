import os
import chromadb
import ollama
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Помилка: Не знайдено GROQ_API_KEY у файлі .env")
    exit()


DB_PATH = "./my_base"
COLLECTION_NAME = "Curse_docs"

# --- STEP 1: SETUP ---
print("⚙️  Initializing RAG system with .env config...")

groq_client = Groq(api_key=GROQ_API_KEY)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# --- STEP 2: USER INPUT ---
user_question = "Для чого використовують Python?"
print(f"\n User Question: {user_question}")

# --- STEP 3: RETRIEVAL ---
print("Searching for context...")

query_embed = ollama.embeddings(model="nomic-embed-text", prompt=user_question)['embedding']

results = collection.query(
    query_embeddings=[query_embed],
    n_results=1 
)

if results['documents']:
    retrieved_context = results['documents'][0][0]
    print(f"Found Context: {retrieved_context}")
else:
    print("No context found!")
    retrieved_context = ""

# --- STEP 4: Augmentation ---
system_prompt = f"""
You are a helpful assistant. 
Use the provided CONTEXT to answer the user's QUESTION.
If the answer is not in the context, say "I don't know based on the context".

CONTEXT:
{retrieved_context}
"""

print("Generating answer...")

chat_completion = groq_client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ],
    model="llama-3.3-70b-versatile",
    temperature=0.5,
)

print("\n" + "="*40)
print("✅ FINAL ANSWER:")
print("="*40)
print(chat_completion.choices[0].message.content)
print("="*40)