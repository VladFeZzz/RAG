import ollama
import numpy as np

# Тестуємо англійською
questions_en = [
    "what is Python",
    "what is RAG",
    "what is Spotify"
]

embeddings_list = []

print("🇬🇧 Тест з АНГЛІЙСЬКОЮ мовою:")
print("="*60)

for q in questions_en:
    emb = ollama.embeddings(model="nomic-embed-text", prompt=q)['embedding']
    embeddings_list.append(emb)
    print(f"'{q}' -> перші 5: {emb[:5]}")

e1 = np.array(embeddings_list[0])
e2 = np.array(embeddings_list[1])
e3 = np.array(embeddings_list[2])

print("\nРізниця між embeddings:")
print(f"Python vs RAG: {np.sum(np.abs(e1 - e2)):.2f}")
print(f"Python vs Spotify: {np.sum(np.abs(e1 - e3)):.2f}")
print(f"RAG vs Spotify: {np.sum(np.abs(e2 - e3)):.2f}")

if np.allclose(e1, e2) and np.allclose(e2, e3):
    print("\n❌ Embeddings ОДНАКОВІ")
else:
    print("\n✅ Embeddings РІЗНІ - модель працює!")
