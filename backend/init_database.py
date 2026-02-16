import chromadb
import ollama

print("Initializing ChromaDB vector database for Music AI Assistant...")
print("-" * 60)

client = chromadb.PersistentClient(path="./my_base")
collection = client.get_or_create_collection(name="Curse_docs")

music_knowledge = [
    "Python — це мова програмування, яку найчастіше використовують в Machine Learning та аналізі музичних даних.",
    "RAG (Retrieval-Augmented Generation) — це технологія ШІ, що дозволяє використовувати власну базу знань для генерації точних відповідей.",
    "Ollama — це інструмент для локального запуску великих мовних моделей, таких як Llama 3.",
    "Groq — це компанія, яка створила спеціалізовані LPU-процесори для надшвидкої генерації тексту та обробки AI-моделей.",
    "Векторна база даних зберігає інформацію у вигляді математичних векторів (embeddings), що дозволяє знаходити схожі за змістом документи.",
    "ChromaDB — це векторна база даних з відкритим кодом для зберігання та пошуку ембеддінгів.",
    "Music Information Retrieval (MIR) — це галузь, що займається автоматичним аналізом музики: жанрів, ритму, мелодій.",
    "MIDI (Musical Instrument Digital Interface) — це стандартний протокол для передачі музичних даних між інструментами та комп'ютерами.",
    "Spotify використовує Machine Learning для створення персоналізованих плейлистів та рекомендацій музики.",
    "FFT (Fast Fourier Transform) — це алгоритм, який перетворює звукові хвилі в частотний спектр для аналізу музики.",
    "Нейронні мережі можуть генерувати музику, аналізуючи паттерни в існуючих композиціях.",
    "BPM (Beats Per Minute) — це міра темпу музики, яка визначає кількість ударів на хвилину.",
    "Жанри музики включають рок, джаз, класику, поп, хіп-хоп, електронну музику та багато інших стилів.",
    "Аудіо-фінгерпринтинг — це технологія розпізнавання музики, яку використовують сервіси як Shazam.",
    "Librosa — це Python бібліотека для аналізу музичних та аудіо сигналів.",
]

print(f"Loading {len(music_knowledge)} documents into database...\n")

for i, doc in enumerate(music_knowledge, 1):
    response = ollama.embeddings(model="nomic-embed-text", prompt=doc)
    embedding = response["embedding"]
    
    collection.upsert(
        ids=[f"doc_{i}"],
        embeddings=[embedding],
        documents=[doc]
    )
    
    print(f"Document {i}/{len(music_knowledge)}: {doc[:60]}...")

print("\n" + "=" * 60)
print("Database successfully created and populated")
print(f"Total documents: {collection.count()}")
print("=" * 60)
