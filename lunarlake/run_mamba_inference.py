from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

# Load from the exported OpenVINO IR produced by pareto.py
OV_MODEL_DIR = "../openvino_exports/mixed_precision_final"

model = OVModelForCausalLM.from_pretrained(OV_MODEL_DIR, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(OV_MODEL_DIR, trust_remote_code=True)

# compile for Lunarlake NPU
model.compile(device="NPU")
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
