# KL Divergence based Post-Training Mixed-Precision Quantization (KL-MPQ)

Minimal setup for **KL Divergence based Post-Training Mixed-Precision Quantization (KL-MPQ)** 

---

## Docker Image
docker pull ghcr.io/tilmto/hymba:v1

## Repository Structure
1. algorithm.py               # Mixed-precision allocator (KL-guided)  <br />
2. evaluate_ppl.py            # Perplexity evaluation via lm-eval  <br />
3. rank.py                    # Rank surrogate metrics vs perplexity  <br />
4. sensitivity_analysis.py    # Per-layer sensitivity + stability metrics  <br />
5. results/                   # Store KL JSON files here <br />
6. quant_utils.py             # Quantization utilities (user-provided)  <br />

## Run the Container
GPU
docker run --rm -it --gpus all \
  -v $PWD:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  ghcr.io/tilmto/hymba:v1 bash

## Quickstart
```bash
### 1️⃣ Evaluate Baseline Perplexity
python evaluate_ppl.py \
  --repo_name nvidia/Hymba-1.5B-Base \
  --local_model_dir ./my_hymba_local \
  --task wikitext \
  --batch_size 8

2️⃣ Run Per-Layer Sensitivity Analysis
python sensitivity_analysis.py \
  --model_id state-spaces/mamba-790m-hf \
  --bits 4 \
  --output results/sensitivity_results.json

3️⃣ Rank Metrics vs Perplexity
python rank.py \
  --metric_path results/metrics.json \
  --ppl_path results/sensitivity_results.json

4️⃣ Run Greedy Mixed-Precision Allocation
python algorithm.py \
  --pretrained state-spaces/mamba-790m-hf \
  --kl4 ./results/mamba130m_sensitivity_results.json \
  --kl8 ./results/mamba_130m_results_8bits.json \
  --gamma 1.10
````
