import google.generativeai as genai
import os

api_key = os.getenv('GENAI_API_KEY')

if not api_key:
    raise ValueError("API key not found. Please set the 'GENAI_API_KEY' environment variable.")

    # 配置Google Generative AI
genai.configure(api_key=api_key, transport='rest')
model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat(history=[])
response = chat.send_message('What is the capital of France?')
print(response.text)
print(chat.history)
