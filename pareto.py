"""
Post-Training Mixed-Precision Quantization
(Sweep through sorted KL-divergence list, logging perplexity + memory at each step)

Two JSON files are expected:
    ├─ kl_4bit_sensitivity.json   # KL list for 4-bit trials
    └─ kl_8bit_sensitivity.json   # KL list for 8-bit trials

Each file must be a dict  {layer_name: {"kl_student_to_teacher": float, …}}

Output: results/quantization_sweep.json
    A list of dicts, one per quantization step, each containing:
        - step
        - layer_name
        - bit  (4 or 8)
        - kl_value
        - perplexity
        - memory_gb
"""

import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate
from quant_utils import quantize_weight_per_channel_absmax


# ---------------------------------------------------------------------------
#  runtime device
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ----------  helpers ----------------------------------------------------------

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


def get_model_memory_gb(model):
    """Compute total parameter memory in GB based on actual dtype of each param."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    return total_bytes / (1024 ** 3)


# ----------  phase-1: build sensitivity list ----------------------------------

def build_sensitivity_list(path4, path8):
    """
    Merge 4-bit and 8-bit KL files into
        [(layer, bit, kl_value), …]  sorted ASC by kl_value (smaller KL = safer to quantize)
    """
    sens4 = json.load(open(path4))
    sens8 = json.load(open(path8))

    merged = []
    for layer, stats in sens4.items():
        merged.append((layer, 4, stats["kl_student_to_teacher"]))
    for layer, stats in sens8.items():
        merged.append((layer, 8, stats["kl_student_to_teacher"]))

    return sorted(merged, key=lambda t: t[2])  # ASC by KL


# ----------  main: greedy sweep -----------------------------------------------

def main():
    pretrained = "state-spaces/mamba-130m-hf"
    workdir = "tmp_quant_dir"
    os.makedirs(workdir, exist_ok=True)
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    KL4_PATH = "./results/mamba130m_sensitivity_results_4bits.json"
    KL8_PATH = "./results/mamba130m_sensitivity_results_8bits.json"

    # phase-1: build sorted sensitivity list
    S = build_sensitivity_list(KL4_PATH, KL8_PATH)

    # load baseline (full-precision / fp16) model
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained, trust_remote_code=True
    ).half()

    # --- baseline measurement (step 0) ---
    base_ppl = evaluate_perplexity(model, tokenizer, workdir)
    base_mem = get_model_memory_gb(model)
    print(f"baseline  |  perplexity: {base_ppl:.2f}  |  memory: {base_mem:.4f} GB")

    sweep_log = [
        {
            "step": 0,
            "layer_name": "baseline",
            "bit": 16,
            "kl_value": 0.0,
            "perplexity": base_ppl,
            "memory_gb": base_mem,
        }
    ]

    # --- sweep through sorted list, quantize each layer cumulatively ---
    for idx, (layer, bit, kl_val) in enumerate(S, start=1):
        print(f"\n▶ step {idx}: quantize {layer} → {bit}-bit  (kl={kl_val:.6f})")

        quantize_single_layer(model, layer, n_bits=bit)

        ppl = evaluate_perplexity(model, tokenizer, workdir)
        mem = get_model_memory_gb(model)
        print(f"   perplexity = {ppl:.2f}  |  memory = {mem:.4f} GB")

        sweep_log.append(
            {
                "step": idx,
                "layer_name": layer,
                "bit": bit,
                "kl_value": kl_val,
                "perplexity": ppl,
                "memory_gb": mem,
            }
        )

        # checkpoint after every step so nothing is lost on crash
        out_path = os.path.join(output_dir, "quantization_sweep.json")
        with open(out_path, "w") as f:
            json.dump(sweep_log, f, indent=2)

    # final save of quantized model
    model.save_pretrained("mixed_precision_final")
    tokenizer.save_pretrained("mixed_precision_final")

    print(f"\nsweep complete — {len(sweep_log)} entries written to {out_path}")


if __name__ == "__main__":
    main()
