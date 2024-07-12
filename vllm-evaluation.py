from prompts import metrics
import ollama
import json
# from vllm import LLM, SamplingParams
from openai import OpenAI

with open('data/vllm/summ_res.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# 评价结果
eval_results = []

offset = 10
# llm = LLM(model='Qwen/Qwen2-72B', trust_remote_code=True)
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

for i, dt in enumerate(data[offset:100]):
    res = dt
    res['scores'] = {}
    print(f'第{i+offset}个文档：\n{res["source"]}\n{res["system_output"]}')

    for criterion, definition in metrics.items():
        prompt_text = f"评价标准：{criterion} - {definition}\n" \
                      f"源文档：{data[1]['source']}\n" \
                      f"摘要：{data[1]['system_output']}"
        # f"请参考源文档和评价标准对摘要进行评价。注意，你的输出一个1~5范围内的分数（分数越高表明摘要越符合评价标准的要求）且不能包含任何其他内容！\n" \
        # f"请参考源文档和评价标准对摘要进行评价。注意，你的输出必须按照如下格式：\n" \
        # f"评分：（一个1~5范围内的分数，分数越高表明摘要越符合评价标准的要求）\n" \
        # f"解释：（阐述你给出评分的理由）\n" \

        response = client.chat.completions.create(
            model='Qwen2-72B',
            messages=[{
                    'role': 'system',
                    'content': f'你是一个自动文本摘要评价器，你需要参考「源文档」和「评价标准」对「摘要」进行评价。需要注意的是，你的输出必须按照如下格式：\n'
                               f"评分：（一个1~5范围内的分数，分数越高表明摘要越符合评价标准的要求）\n"
                               f"解释：（阐述你给出评分的理由，不要出现换行符）"
                }, {
                    'role': 'user',
                    'content': prompt_text
                }]
        )
        response = response.choices[0].message.content
        exit()

        response = ollama.chat(
            model='qwen2:72b',
            messages=[
                # {
                #     'role': 'system',
                #     'content': '你是一个自动文本摘要评价器，你需要参考源文档和评价标准对摘要进行评价。注意，你的输出一个1~5范围内的分数（分数越高表明摘要越符合评价标准的要求）且不能包含任何其他内容！'
                # },
                {
                    'role': 'system',
                    'content': f'你是一个自动文本摘要评价器，你需要参考「源文档」和「评价标准」对「摘要」进行评价。需要注意的是，你的输出必须按照如下格式：\n'
                               f"评分：（一个1~5范围内的分数，分数越高表明摘要越符合评价标准的要求）\n"
                               f"解释：（阐述你给出评分的理由，不要出现换行符）"
                }, {
                    'role': 'user',
                    'content': prompt_text
                }]
        )
        response = response['message']['content']
        try:
            response = response.split('评分：')[1].split('解释：')
            score = response[0].strip()
            reason = response[1].strip()
            res['scores'][criterion] = {
                'score': score,
                'reason': reason
            }
            print(criterion, score, reason)
        except IndexError as e:
            print(e)
            # print(criterion, response)
        # print(criterion, response)

    eval_results.append(res)

    if (i + 1) % 2 == 0:
        # 写入结果
        with open('data/eval_res.jsonl', 'a+') as f:
            for res in eval_results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

