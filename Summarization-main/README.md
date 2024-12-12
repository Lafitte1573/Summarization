# LlmSumma
A simple demo of using a small LLM (QWen2-7B) to perform text summarization and evaluate the results with a larger LLM (QWen2-72B).

## 安装依赖
1. 安装ollama
```
curl -fsSL https://ollama.com/install.sh | sh
```

2. 安装ollama-python
```
pip install ollama
```

3. 安装vllm
```
pip install vllm
```

# ollama方式实现摘要生成与评价
## 调用Qwen2-7B做摘要生成
首先在命令行输入如下命令运行Qwen2-7B模型
```
ollama run qwen2
```
然后运行代码
```
python ollama-summa.py
```

## 调用Qwen2-72B评价生成摘要的质量
首先在命令行输入如下命令运行Qwen2-72B模型
```
ollama run qwen2:72b
```
然后运行代码
```
python ollama-evaluation.py
```

# vllm方式实现摘要生成与评价
## 调用Qwen2-7B-Instruct做摘要生成
首先允许vllm从ModelScope下载模型：
```
export VLLM_USE_MODELSCOPE=True
```
然后输入如下命令运行Qwen2-7B-Instruct模型
```
nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-7B-Instruct --model Qwen/Qwen2-7B-Instruct &
```
如果是第一次运行该模型，推荐使用下述方式，以方便查看模型下载是否成功
```
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-7B-Instruct --model Qwen/Qwen2-7B-Instruct
```
最后，运行代码
```
python vllm-summa.py
```

## 调用Qwen2-72B评价生成摘要的质量
首先运行Qwen2-72B模型
```
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-72B --model Qwen/Qwen2-72B
```
然后执行代码
```
python vllm-evaluation.py
```
