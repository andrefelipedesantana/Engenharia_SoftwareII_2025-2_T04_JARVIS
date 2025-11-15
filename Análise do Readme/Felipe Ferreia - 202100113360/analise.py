from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device="cpu")

texto_para_analisar = """
Aqui esta o redme

JARVIS

arXiv Open in Spaces

The mission of JARVIS is to explore artificial general intelligence (AGI) and deliver cutting-edge research to the whole community.

What's New

We release Easytool for easier tool usage.
The code and datasets are available at EasyTool.
The paper is available at EasyTool: Enhancing LLM-based Agents with Concise Tool Instruction.

We release TaskBench for evaluating task automation capability of LLMs.
The code and datasets are available at TaskBench.
The paper is available at TaskBench: Benchmarking Large Language Models for Task Automation.

We are now in the process of planning evaluation and project rebuilding. We will release a new version of Jarvis in the near future.

We released a light langchain version of Jarvis. See here.

Jarvis now supports the OpenAI service on the Azure platform and the GPT-4 model.

We added the Gradio demo and built the web API for /tasks and /results in server mode.
The Gradio demo is now hosted on Hugging Face Space. (Build with inference_mode=hybrid and local_deployment=standard)
The Web API /tasks and /results access intermediate results for Stage #1: task planning and Stage #1-3: model selection with execution results. See here.

We added the CLI mode and provided parameters for configuring the scale of local endpoints.
You can enjoy a lightweight experience with Jarvis without deploying the models locally. See here.
Just run python awesome_chat.py --config configs/config.lite.yaml to experience it.

We updated a version of code for building.

Overview

Language serves as an interface for LLMs to connect numerous AI models for solving complicated AI tasks!

See our paper: HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace, Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu and Yueting Zhuang (the first two authors contribute equally)

We introduce a collaborative system that consists of an LLM as the controller and numerous expert models as collaborative executors (from HuggingFace Hub). The workflow of our system consists of four stages:

Task Planning: Using ChatGPT to analyze the requests of users to understand their intention, and disassemble them into possible solvable tasks.
Model Selection: To solve the planned tasks, ChatGPT selects expert models hosted on Hugging Face based on their descriptions.
Task Execution: Invokes and executes each selected model, and return the results to ChatGPT.
Response Generation: Finally, using ChatGPT to integrate the prediction of all models, and generate responses.

System Requirements

Default (Recommended)
For configs/config.default.yaml:

Ubuntu 16.04 LTS
VRAM >= 24GB
RAM > 12GB (minimal), 16GB (standard), 80GB (full)
Disk > 284GB
42GB for damo-vilab/text-to-video-ms-1.7b
126GB for ControlNet
66GB for stable-diffusion-v1-5
50GB for others
"""

meus_labels = [
    'Modelo Cliente-Servidor (separação clara entre o cliente que pede e o servidor que responde)',
    'Arquitetura Orientada a Serviços (SOA) (serviços maiores e reutilizáveis que se comunicam)',
    'Arquitetura Orientada a Eventos (comunicação baseada na produção e consumo de eventos assíncronos)',
    'Arquitetura Monolítica (aplicação construída como uma única unidade coesa)',
    'Arquitetura em Camadas (componentes organizados por função ex: apresentação lógica dados)',
    'Arquitetura de Microsserviços (sistema dividido em pequenos serviços independentes)',
    'Arquitetura Serverless (execução de código gerenciada pela nuvem sem gerenciamento de servidor)'
]



resultado = classifier(texto_para_analisar,
                       meus_labels,
                       multi_label=True)
print(f"--- Resultados da Análise para o README 'HuggingGPT' ---")
resultados_ordenados = sorted(zip(resultado['labels'], resultado['scores']), key=lambda x: x[1], reverse=True)
for label, score in resultados_ordenados:
    print(f"{label}: {score*100:.1f}%")