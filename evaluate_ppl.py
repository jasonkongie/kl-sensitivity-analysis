 cat evaluate_ppl.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate

repo_name = "nvidia/Hymba-1.5B-Base"
local_model_dir = "./my_hymba_local"

# 1) Load your model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo_name, trust_remote_code=True)
model.cuda().half()

# 2) [Optional] Modify the model
# ...

# 3) Save to a local directory (contains config.json, pytorch_model.bin, tokenizer files, etc.)
if not os.path.exists(local_model_dir):
    os.makedirs(local_model_dir)
model.save_pretrained(local_model_dir)
tokenizer.save_pretrained(local_model_dir)

# 4) Wrap using your local directory
wrapped_model = HFLM(
    pretrained=local_model_dir,    # Must be a valid folder with config.json
    tokenizer=tokenizer,
    parallelize=False,
    device_map=None,
    trust_remote_code=True
)

# 5) Evaluate
evaluation_results = simple_evaluate(
    model=wrapped_model,
    tasks=["wikitext"],
    num_fewshot=0,
    batch_size=8,
)


ppl = evaluation_results['results']            # get perplexity or None if not found
print(f"WikiText Perplexity: {ppl}")%      
