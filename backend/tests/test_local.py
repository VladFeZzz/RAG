import ollama

print("Загрузка...")

response = ollama.chat(model='llama3.1', messages=[
  {
    'role': 'user',
    'content': 'Напиши рандомне слово',
  },
])

print("Відповідь:")
print(response['message']['content'])