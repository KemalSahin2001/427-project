import openai

client = openai.OpenAI(api_key="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")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "<system prompt here>"},
        {"role": "user", "content": "<user prompt here>"}
    ],
    temperature=0,
    top_p=0.95
)

print(response.choices[0].message.content)
