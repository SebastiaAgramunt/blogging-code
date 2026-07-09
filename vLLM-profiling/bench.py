# bench.py
from vllm import LLM, SamplingParams
import json

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", gpu_memory_utilization=0.85)

prompts = [f"Explain the concept of {topic} in three sentences." 
           for topic in ["entropy", "recursion", "inflation", "photosynthesis"] * 8]  # 32 prompts

sampling = SamplingParams(temperature=0.7, max_tokens=128)

outputs = llm.generate(prompts, sampling)
print(f"Generated {len(outputs)} completions")