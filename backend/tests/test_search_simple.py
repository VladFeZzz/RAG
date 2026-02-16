import chromadb
import ollama

chroma_client = chromadb.PersistentClient(path="./my_base")
collection = chroma_client.get_or_create_collection(name="Curse_docs")

print(f"📊 База містить {collection.count()} документів\n")


test_questions = [
    "що таке Python",
    "що таке RAG", 
    "що таке Spotify",
    "як аналізувати музику"
]

for question in test_questions:
    print("=" * 70)
    print(f"❓ Питання: {question}")
    print("=" * 70)

    query_embed = ollama.embeddings(model="nomic-embed-text", prompt=question)['embedding']
    

    results = collection.query(
        query_embeddings=[query_embed],
        n_results=3
    )
 
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        distance = results['distances'][0][i]
        
        print(f"\n{i+1}. Distance: {distance:.2f}")
        print(f"   📄 {doc}")
    
    print()
