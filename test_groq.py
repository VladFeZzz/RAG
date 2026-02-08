import os
from groq import Groq


client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
)

print("Request on Groq...")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Скажи 3 слова",
        }
    ],
    model="llama-3.3-70b-versatile", 
)

print("Response:")
print(chat_completion.choices[0].message.content)