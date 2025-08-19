# from google import genai

# api_key = "AIzaSyDhuvyZS5D5jYpoZo2ZoVF41bj9PqCiD6A"

# MODEL = "gemini-2.5-flash"
# client = genai.Client(api_key=api_key)
# response = client.models.generate_content(
#                 model=MODEL,
#                 contents="hello world!",
#             )
# model_response_text = response.text
# print(model_response_text)

from openai import OpenAI

client = OpenAI(
    api_key="AIzaSyDhuvyZS5D5jYpoZo2ZoVF41bj9PqCiD6A",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "who are you?"
        }
    ]
)

print(response.choices[0].message.content)