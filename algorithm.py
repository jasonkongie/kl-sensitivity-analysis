"""
Post-Training Mixed-Precision Quantization
(using KL-divergence for layer sensitivity)

Two JSON files are expected:
    ├─ kl_4bit_sensitivity.json   # KL list for 4-bit trials
    └─ kl_8bit_sensitivity.json   # KL list for 8-bit trials

Each file must be a dict  {layer_name: {"kl_student_to_teacher": float, …}}
"""

import os, json, copy, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate
from quant_utils import quantize_weight_per_channel_absmax


# ---------------------------------------------------------------------------
#  runtime device (avoid hard-coded CUDA on macOS; prefer MPS if available)
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hide CUDA slots on non-CUDA builds


# ----------  low-level helpers ------------------------------------------------
def quantize_single_layer(model, layer_name, n_bits):
    """In-place per-channel abs-max quantization of the layer weights."""
    for name, mod in model.named_modules():
        if name == layer_name and hasattr(mod, "weight"):
            mod.weight.data = quantize_weight_per_channel_absmax(
                mod.weight.data, n_bits=n_bits
            )
            return
    raise KeyError(f"{layer_name} not found")


def evaluate_perplexity(model, tokenizer, workdir, task="wikitext", batch_size=8):
    """Returns word-level perplexity on `task`."""
    model.save_pretrained(workdir)
    tokenizer.save_pretrained(workdir)
    wrapped = HFLM(
        pretrained=workdir,
        tokenizer=tokenizer,
        parallelize=False,
        device=DEVICE,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )
    res = simple_evaluate(
        model=wrapped, tasks=[task], num_fewshot=0, batch_size=batch_size
    )
    return res["results"][task]["word_perplexity,none"]


# ----------  phase-1: build sensitivity list ----------------------------------
def build_sensitivity_list(path4, path8):
    """
    Merge 4-bit and 8-bit KL files into
        [(layer, bit, kl_value), …]  sorted ASC by kl_value (smaller = better)
    """
    sens4 = json.load(open(path4))
    sens8 = json.load(open(path8))

    merged = []
    for layer, stats in sens4.items():
        merged.append((layer, 4, stats["kl_student_to_teacher"]))
    for layer, stats in sens8.items():
        merged.append((layer, 8, stats["kl_student_to_teacher"]))

    return sorted(merged, key=lambda t: t[2])  # ↑ ASC


# ----------  phase-2: greedy allocation --------------------------------------
def main():
    pretrained = "state-spaces/mamba-790m-hf"
    workdir = "tmp_quant_dir"
    os.makedirs(workdir, exist_ok=True)
    KL4_PATH = "./results/mamba130m_sensitivity_results.json"
    KL8_PATH = "./results/mamba_130m_results_8bits.json"
    GAMMA = 1.10  # max allowed 10 % perplexity inflation

    # phase-1  (sensitivity list S)
    S = build_sensitivity_list(KL4_PATH, KL8_PATH)

    # load baseline (full/16-bit) model
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained, trust_remote_code=True
    ).half()

    base_ppl = evaluate_perplexity(model, tokenizer, workdir)
    print(f"baseline perplexity: {base_ppl:.2f}")

    # phase-2  (greedy layer allocation)
    for layer, bit, _ in S:
        print(f"\n▶ quantize {layer} → {bit}-bit")
        # save current weights for potential rollback
        cache = copy.deepcopy(model.state_dict()[f"{layer}.weight"])

        quantize_single_layer(model, layer, n_bits=bit)
        ppl = evaluate_perplexity(model, tokenizer, workdir)
        print(f"   perplexity = {ppl:.2f}")

        if ppl > base_ppl * GAMMA:  # budget exceeded → revert
            print("exceeds; revert layer")
            model.state_dict()[f"{layer}.weight"].copy_(cache)
        else:  # accept layer; update budget
            print("   ✅ kept")
            base_ppl = ppl

    # optional: final save
    model.save_pretrained("mixed_precision_final")
    tokenizer.save_pretrained("mixed_precision_final")


if __name__ == "__main__":
    main()%  
