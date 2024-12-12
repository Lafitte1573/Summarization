from openai import OpenAI
import json
import os
from os import path

# Set OpenAI's API key and API base to use vLLM's API server.
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# chat_response = client.chat.completions.create(
#     model="Qwen2-7B-Instruct",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me something about large language models."},
#     ]
# )
# print("Chat response:", chat_response.choices[0].message.content)


with open('data/lcsts_2000.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# with open('./data/lcsts_2000.jsonl', 'r') as f:
#     data = [json.loads(line) for line in f]

results = []

for i, d in enumerate(data):
    # 就是说 ollama 也是实现了一个类似 openai API/Zhipu API 的接口 - chat()，调用方式也是类似的
    # 所以理论上我如果想用 qwen2 的话，直接用 ollama 把它跑在本地就行了，就不花钱使用官方 API 了

    response = client.chat.completions.create(
        model='Qwen2-7B-Instruct',
        messages=[{
            'role': 'user',
            'content': '将以下文本转述为20个字以内的摘要：\n' + d['content']
        }]
    )
    response = response.choices[0].message.content
    print(i, response)

    res = {
        "doc_id": f"{i}",
        "system_id": "Qwen2-7B-Instruct",
        "source": d['content'],
        "reference": d['summary'],
        "system_output": response,
    }
    results.append(res)


if not path.exists('data/vllm'):
    os.makedirs('data/vllm')

with open('data/vllm/summ_res.jsonl', 'w') as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + '\n')
