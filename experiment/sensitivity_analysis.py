from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate
from quant_utils import quantize_weight_per_channel_absmax, evaluate_layer_lyapunov_metric 
# evaluate_layer_stability_metrics_mean, 
import json
import os


# Define quantization for a single layer
def quantize_single_layer(model, layer_name, n_bits=4):
    """
    Quantizes the weights of a single layer in the model.
    """
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, "weight"):
            print(f"Quantizing layer: {name}")
            module.weight.data = quantize_weight_per_channel_absmax(module.weight.data, n_bits=n_bits)
            break
    return model

# Define evaluation function
def evaluate_model(model, tokenizer, local_model_dir, task="wikitext", batch_size=8):
    """
    Evaluates the model's perplexity on the specified task.
    """
    model.save_pretrained(local_model_dir)
    tokenizer.save_pretrained(local_model_dir)
    
    wrapped_model = HFLM(
    pretrained=local_model_dir,    # Must be a valid folder with config.json
    tokenizer=tokenizer,
    parallelize=False,
    device_map=None,
    trust_remote_code=True
)

    # Use your evaluation method or adjust as per `lm_eval`
    from lm_eval.evaluator import simple_evaluate
    result = simple_evaluate(
        model=wrapped_model,
        tasks=[task],
        num_fewshot=0,
        batch_size=batch_size
    )

    ppl = result['results'][task]['word_perplexity,none']
    return ppl

# Main function for sensitivity analysis
def main():
    # Model and tokenizer initialization
    print("Current working directory:", os.getcwd())
    # pretrained_model = "nvidia/Hymba-1.5B-Base"  # Replace with your model path or ID
    # pretrained_model = "/Users/jasonkong/Documents/Hymba-1.5B-Base" #Nautilus usage
    pretrained_model = "state-spaces/mamba-790m-hf"
    model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model.half()

    # Set directory for saving intermediate weights and results
    local_model_dir = "quantized_model_dir"
    results_file = "sensitivity_results.json"
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)

    # List quantizable layers
    quantizable_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    print(f'Quantiable Modules: {quantizable_layers}')

    # Results dictionary
    sensitivity_results = {}

    # Layer-by-layer quantization
    for layer_name in quantizable_layers:
        print(f"Processing layer: {layer_name}")

        # Reload the model for a clean start
        model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True)
        model.half()
        
        #pre-quantization stability metrics 
        # pre_mean = evaluate_layer_stability_metrics_mean(model, layer_name)
        # print(f"Pre-quantization stability metrics for {layer_name}: Mean: {pre_mean}")

        pre_lyap = evaluate_layer_lyapunov_metric(model, layer_name)
        print(f"Pre-quantization Lyapunov metric for {layer_name}: {pre_lyap}")


        # Quantize the current layer
        model = quantize_single_layer(model, layer_name, n_bits=4)

        #post-quantization stability metrics
        # post_mean = evaluate_layer_stability_metrics_mean(model, layer_name)
        # print(f"Post-quantization stability metrics for {layer_name}: Mean: {post_mean}")

        post_lyap = evaluate_layer_lyapunov_metric(model, layer_name)
        print(f"Post-quantization Lyapunov metric for {layer_name}: {post_lyap}")

        # Evaluate perplexity
        perplexity = evaluate_model(model, tokenizer, local_model_dir)
        print(f"Layer: {layer_name}, Perplexity: {perplexity}")

        # Save result
        # sensitivity_results[layer_name] = perplexity
        sensitivity_results[layer_name] = {
            "pre_quantization": {"lyapunov": pre_lyap},
            "post_quantization": {"lyapunov": post_lyap},
            "perplexity": perplexity
        }

    # Save results to a file
    with open(results_file, "w") as f:
        json.dump(sensitivity_results, f, indent=4)

    print(f"Sensitivity analysis complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()
