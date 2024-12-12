import ollama
import json

with open('LlmSumm/data/lcsts_2000.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# with open('./data/lcsts_2000.jsonl', 'r') as f:
#     data = [json.loads(line) for line in f]

results = []

for i, d in enumerate(data):
    # 就是说 ollama 也是实现了一个类似 openai API/Zhipu API 的接口 - chat()，调用方式也是类似的
    # 所以理论上我如果想用 qwen2 的话，直接用 ollama 把它跑在本地就行了，就不花钱使用官方 API 了

    response = ollama.chat(
        model='qwen2',
        messages=[{
            'role': 'user',
            'content': '将以下文本转述为20个字以内的摘要：\n' + d['content']
        }]
    )
    response = response['message']['content']
    print(i, response)

    res = {
        "doc_id": f"{i}",
        "system_id": "qwen2",
        "source": d['content'],
        "reference": d['summary'],
        "system_output": response,
        "scores": {
            "coherence": None,
            "consistency": None,
            "fluency": None,
            "relevance": None,
            "overall": None
        }
    }
    results.append(res)

with open('LlmSumm/data/summ_res.jsonl', 'w') as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + '\n')
