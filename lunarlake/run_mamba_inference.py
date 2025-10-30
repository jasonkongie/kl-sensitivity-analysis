from optimum.intel import OVModelForCausalLM

# new imports for inference
from transformers import AutoTokenizer

# load the model
model_id = "state-spaces/mamba-130m-hf" #replace this with mamba model
OVModelForCausalLM.from_pretrained("Zyphra/Zamba-7B-v1", export=True, trust_remote_code=True)
model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)

# inference
prompt = "How's the weather in San Diego right now?"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.compile(device="GPU")
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
