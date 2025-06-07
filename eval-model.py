import argparse
from datasets import load_dataset
import random
import hashlib
import json
import asyncio
from openai import AsyncOpenAI, OpenAI
from math_verify import parse, verify
from tqdm import tqdm

# from huggingface_hub import login
# login("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")

# Initialize OpenAI client for GPT-4-mini
gpt4_mini_client = AsyncOpenAI(
    api_key="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"
)

async def get_gpt4o_mini_response(prompt):
    """Get a single response from GPT-4-mini"""
    try:
        response = await gpt4_mini_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

async def is_correct_gpt4o_mini(gen_answer, gold_answer):
    prompt = f'Are these parsed math expressions equivalent? Reference: "{gold_answer}" Generated: "{gen_answer}". Answer with \\boxed{{Yes}} or \\boxed{{No}}.'
    response = await get_gpt4o_mini_response(prompt)    
    return response, 'yes' in response.lower()


class ModelClient:
    """
    Usage is unchanged:

        client = create_client(cfg)          # wraps this class
        msgs = client.get_prompt_messages(question)
        resp  = await client.get_response(msgs)
        resps = await client.get_responses(list_of_msg_lists)
    """

    def __init__(self, config):
        self.cfg = config
        self.client = AsyncOpenAI(
            base_url=f"http://localhost:{config['port']}/v1",
            api_key="EMPTY"          # any non-empty string works for vLLM
        )

    # -----------------------------------------------------------
    # Prompt helpers (same names / semantics as before)
    # -----------------------------------------------------------
    def get_prompt_messages(self, question):
        """Return a list[message] that the caller can send."""
        if "gemma" in self.cfg["model"].lower():   # Gemma ignores system role
            return [
                {
                    "role": "user",
                    "content": f"Instruction: {self.cfg['system_prompt']}\n\n{question}"
                }
            ]
        return [
            {"role": "system", "content": self.cfg["system_prompt"]},
            {"role": "user",   "content": question},
        ]

    # -----------------------------------------------------------
    # Single async request – returns the full response obj
    # -----------------------------------------------------------
    async def get_response(self, prompt_messages):
        return await self.client.chat.completions.create(
            model=self.cfg["model"],                   # ⚠️ must match /v1/models
            messages=prompt_messages,
            temperature=self.cfg.get("temperature", 0.2),
            max_tokens=self.cfg.get("max_tokens", 512),
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )

    # -----------------------------------------------------------
    # Batched helper – unchanged call signature
    # -----------------------------------------------------------
    async def get_responses(self, prompt_messages):
        """
        Accepts a list of *message-arrays*, sends them concurrently in
        batches, and returns a list of completion objects.
        """
        batch_sz   = self.cfg.get("batch_size", 32)
        responses  = []
        total      = len(prompt_messages)
        n_batches  = (total + batch_sz - 1) // batch_sz

        with tqdm(total=total, desc="Getting responses", unit="prompt") as pbar:
            for i in range(0, total, batch_sz):
                batch = prompt_messages[i : i + batch_sz]
                tasks = [self.get_response(msgs) for msgs in batch]
                batch_res = await asyncio.gather(*tasks)
                responses.extend(batch_res)

                pbar.update(len(batch))
                pbar.set_postfix(batch=f"{i//batch_sz + 1}/{n_batches}")

        return responses

def format_dataset(dataset, dataset_name, seed, sample_size):
    formatted_dataset = []
    for item in dataset:
        if dataset_name == "agentica-org/DeepScaleR-Preview-Dataset":
            formatted_dataset.append({
                "question": item["problem"],
                "solution": item["solution"],
                "answer": item["answer"]
            })
    formatted_dataset.sort(key=lambda x: x['question'])
    random.seed(seed)
    formatted_dataset = random.sample(formatted_dataset, sample_size)
    
    # Print deterministic hash of formatted dataset. Just to ensure that the sampled dataset is same across runs.
    dataset_str = json.dumps(formatted_dataset, sort_keys=True)
    hash_obj = hashlib.sha256(dataset_str.encode())
    print(f"Dataset hash: {hash_obj.hexdigest()}")
    
    return formatted_dataset, hash_obj

def load_eval_dataset(config):
    dataset_name, seed, sample_size = config['dataset'], config['seed'], config['sample_size']
    raw_dataset = load_dataset(dataset_name)['train']
    return format_dataset(raw_dataset, dataset_name, seed, sample_size)

def create_client(config):
    return ModelClient(config)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model on a dataset')
    parser.add_argument('--config', type=str, default="configs/gsm8k.json",
                      help='Config file to use (default: configs/gsm8k.json)')
    args = parser.parse_args()
    return args

async def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    client = create_client(config)
    print(f"Loading dataset...")
    dataset, dataset_hash = load_eval_dataset(config)
    
    print(f"Preparing {len(dataset)} prompts...")
    all_prompt_messages = []
    for item in dataset:
        prompt_messages = client.get_prompt_messages(item['question'])
        all_prompt_messages.append(prompt_messages)

    print(f"Getting model responses in batches...")
    responses = await client.get_responses(all_prompt_messages)
    print(f"Got {len(responses)} responses...")
    
    print(f"Processing results...")
    eval_results = []
    for i, response in tqdm(enumerate(responses), total=len(responses), desc="Evaluating"):
        gen_solution = response.choices[0].message.content
        gen_answer = parse(gen_solution)
        is_correct = verify(gen_answer, dataset[i]['answer'])
        if len(gen_answer) > 1:
            gen_answer = str(gen_answer[-1])
        else:
            gen_answer = ""
        eval_response = None
        if not is_correct:
            eval_response, is_correct = await is_correct_gpt4o_mini(gen_answer, dataset[i]['answer'])
        
        eval_result = {
            "question": dataset[i]['question'],
            "gen_solution": gen_solution,
            "gen_answer": gen_answer,
            "gold_solution": dataset[i]['solution'],
            "gold_answer": dataset[i]['answer'],
            "is_correct": is_correct,
            "4o-mini_eval_response": eval_response
        }
        eval_results.append(eval_result)

    final_results = {
        "config": config,
        "dataset_hash": dataset_hash.hexdigest(),
        "accuracy": sum(result['is_correct'] for result in eval_results) / len(eval_results),
        "eval_results": eval_results,
    }
    with open(config['output_path'], 'w') as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())