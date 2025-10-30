# KL Divergence based Post-Training Mixed-Precision Quantization (KL-MPQ)

Minimal setup for **KL Divergence based Post-Training Mixed-Precision Quantization (KL-MPQ)** 

---

## Docker Image
docker pull ghcr.io/tilmto/hymba:v1

## Repository Structure
algorithm.py               # Mixed-precision allocator (KL-guided)
evaluate_ppl.py            # Perplexity evaluation via lm-eval
rank.py                    # Rank surrogate metrics vs perplexity
sensitivity_analysis.py    # Per-layer sensitivity + stability metrics
results/                   # Store KL JSON files here
quant_utils.py             # Quantization utilities (user-provided)

🚀 Run the Container
GPU
docker run --rm -it --gpus all \
  -v $PWD:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  ghcr.io/tilmto/hymba:v1 bash

🧩 Quickstart
# 1. Evaluate baseline perplexity
python evaluate_ppl.py

# 2. Run per-layer sensitivity analysis
python sensitivity_analysis.py

# 3. Rank metrics vs perplexity
python rank.py

# 4. Run greedy mixed-precision allocation
python algorithm.py
