import chromadb

client = chromadb.PersistentClient(path="./my_base")

try:
    collection = client.get_collection(name="Curse_docs")

    count = collection.count()
    print(f"Total documents in DB: {count}")

    print("\n--- Database Contents ---")
    data = collection.get()
    
    for i in range(count):
        print(f"ID: {data['ids'][i]}")
        print(f"Content: {data['documents'][i]}")
        print("-" * 30)

except Exception as e:
    print(f"Error: {e}")
    print("Hint: Check if the folder path './my_base' or collection name 'Curse_docs' is correct.")