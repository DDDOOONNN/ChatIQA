from openai import OpenAI
 
client = OpenAI(
    api_key = "sk-yx5yC8y0hIrfl19gCcB94cB11fAe4e87A35155C6De78Ce80", 

    base_url = "http://localhost:3001/v1"
)
chat_completion = client.chat.completions.create(
    model="SparkDesk-v3.5", 
    messages=[
        {
            "role": "user",
            "content": "say a joke",
        }
    ]
)

print(chat_completion.choices[0].message.content)