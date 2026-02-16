import chromadb
import ollama

client = chromadb.PersistentClient(path="./my_base")
collection = client.get_or_create_collection(name="Curse_docs")

documents = [
    "RAG (Retrieval-Augmented Generation) — це технологія, що дозволяє ШІ користуватися твоїми власними даними.",
    "Groq — це компанія, яка створила LPU-процесори для миттєвої генерації тексту.",
    "Векторна база даних зберігає не слова, а їх математичний зміст (ембеддінги).",
    "Ollama — це інструмент для локального запуску моделей, таких як Llama 3.",
    "Python — це мова програмування, яку найчастіше використовують в Machine Learning."
]

print("Starting indexing docs...")

for i, doc in enumerate(documents):
    response = ollama.embeddings(model="nomic-embed-text", prompt=doc)
    embedding = response["embedding"]
    
    collection.upsert(
        ids=[str(i)],           # Унікальний номер документа
        embeddings=[embedding], # Вектор (координати змісту)
        documents=[doc]         # Оригінальний текст (щоб ми могли його прочитати)
    )
    print(f"Doc #{i + 1} added.")

print("\n🎉 Success! Folder 'my_base' created and filled.")

