import chromadb
import ollama

# Підключаємося до бази
chroma_client = chromadb.PersistentClient(path="./my_base")
collection = chroma_client.get_or_create_collection(name="Curse_docs")

# Тестуємо різні питання
questions = [
    "для чого python",
    "що таке RAG",
    "що таке Groq"
]

for question in questions:
    print(f"\n{'='*50}")
    print(f"Питання: {question}")
    print('='*50)
    
    # Генеруємо embedding
    query_embed = ollama.embeddings(model="nomic-embed-text", prompt=question)['embedding']
    
    # Шукаємо найближчий результат
    results = collection.query(query_embeddings=[query_embed], n_results=3)
    
    print(f"\nЗнайдено {len(results['documents'][0])} результатів:")
    
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"\n{i+1}. Схожість: {1 - distance:.4f}")
        print(f"   Текст: {doc}")
