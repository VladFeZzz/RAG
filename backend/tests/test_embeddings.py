import ollama
import numpy as np

questions = [
    "що таке Python",
    "що таке RAG",
    "що таке Spotify"
]

embeddings_list = []

for q in questions:
    emb = ollama.embeddings(model="nomic-embed-text", prompt=q)['embedding']
    embeddings_list.append(emb)
    print(f"'{q}' -> embedding довжина: {len(emb)}, перші 5 значень: {emb[:5]}")

# Перевіряємо чи embeddings різні
print("\n" + "="*60)
print("Перевірка різниці між embeddings:")
print("="*60)

e1 = np.array(embeddings_list[0])
e2 = np.array(embeddings_list[1])
e3 = np.array(embeddings_list[2])

print(f"Різниця між Python і RAG: {np.sum(np.abs(e1 - e2)):.2f}")
print(f"Різниця між Python і Spotify: {np.sum(np.abs(e1 - e3)):.2f}")
print(f"Різниця між RAG і Spotify: {np.sum(np.abs(e2 - e3)):.2f}")

if np.allclose(e1, e2) and np.allclose(e2, e3):
    print("\n⚠️ ПРОБЛЕМА: Всі embeddings ОДНАКОВІ!")
else:
    print("\n✅ Embeddings різні для різних питань")
