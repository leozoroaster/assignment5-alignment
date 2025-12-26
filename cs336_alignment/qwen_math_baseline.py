from vllm import LLM, SamplingParams
from datasets import load_dataset
import re
import json
from pathlib import Path

#math dataset
dataset = load_dataset("gsm8k", "main")

# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
)

# Create an LLM.
llm = LLM(model="Qwen/Qwen2.5-Math-1.5B", gpu_memory_utilization=0.8)

r1_prompt_path="prompts/r1_zero.prompt"

with open(r1_prompt_path, "r", encoding="utf-8") as f:
    r1_zero_prompt = f.read()

from drgrpo_grader import r1_zero_reward_fn

def extract_answer(text: str) -> str | None:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn,
    prompts,
    references,
    eval_sampling_params:SamplingParams
)-> None:

    outputs=vllm_model.generate(prompts, eval_sampling_params)

    results=[]

    for i, out in enumerate(outputs):
        generated_answer=out.outputs[0].text
        answer=extract_answer(generated_answer)
        metrics = reward_fn(generated_answer, references[i])

        results.append({
            "prompt": prompts[i],
            "generation": generated_answer,
            "answer": answer,
            "reference": references[i],
            "metrics": metrics,
        })

    write_jsonl("results.jsonl", results)

def parse_math(example):
  final_answer=example["answer"].split("####")[-1].strip()
  return final_answer

prompts=[]
references=[]

for sample in dataset["train"]:
  prompt=r1_zero_prompt.format(question=sample["question"])
  prompts.append(prompt)
  reference=parse_math(sample)
  references.append(reference)

evaluate_vllm(llm, r1_zero_reward_fn, prompts, references, sampling_params)


