import chromadb
import ollama


client = chromadb.PersistentClient(path="./my_base")
collection = client.get_collection(name="Curse_docs")

user_query = "Що таке Ollama?" 
print(f"Питання: {user_query}")


print("... перетворюю питання на цифри ...")
response = ollama.embeddings(model="nomic-embed-text", prompt=user_query)
query_vector = response["embedding"]


results = collection.query(
    query_embeddings=[query_vector],
    n_results=1
)

best_match = results['documents'][0][0]

print("\n✅ Знайдено у базі:")
print("-" * 30)
print(best_match)
print("-" * 30)