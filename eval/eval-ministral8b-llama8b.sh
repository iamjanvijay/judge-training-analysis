python eval/run_batched_eval.py \
  --configs ./configs/evaluations/set_1/zero/ministral8b.jsonl ./configs/evaluations/set_1/sft/ministral8b.jsonl ./configs/evaluations/set_1/dpo/ministral8b.jsonl ./configs/evaluations/set_1/sft_dpo/ministral8b.jsonl ./configs/evaluations/set_2/zero/llama8b.jsonl ./configs/evaluations/set_2/sft/llama8b.jsonl ./configs/evaluations/set_2/dpo/llama8b.jsonl ./configs/evaluations/set_2/sft_dpo/llama8b.jsonl \
  --gpu_ids 0 1 2 3 4 5 6 7 \
  --max_tokens 4096 \
  --temperature 0.0 \
  --top_p 1.0 \
  --gpu_mem_util 0.9