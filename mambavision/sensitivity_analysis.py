from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from quant_utils import quantize_weight_per_channel_absmax
import json
import os
from transformers import AutoModelForImageClassification
from datasets import load_dataset
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
import torch.nn.functional as F
import random


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def build_calib_batch(model, dataset, n=64, device="cuda"):
    transform = create_transform(
        input_size=(3, 224, 224),
        is_training=False,
        mean=model.config.mean,
        std=model.config.std,
        crop_mode=model.config.crop_mode,
        crop_pct=model.config.crop_pct
    )
    xs = []
    ys = []
    for i in range(n):
        s = dataset[i]
        xs.append(transform(s["image"].convert("RGB")))
        ys.append(s["label"])
    x = torch.stack(xs, dim=0).to(device)
    y = torch.tensor(ys, device=device)
    return x, y


# def quantize_layer(model, layer_name, n_bits=4):
#     for name, module in model.named_modules():
#         if name == layer_name and hasattr(module, "weight"):
#             print(f"Quantizing layer: {name}")
#             module.weight.data = quantize_weight_per_channel_absmax(module.weight.data, n_bits=n_bits)
#             break
#     return model

def quantize_layer(model, layer_name, n_bits=4):
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, "weight") and module.weight is not None:
            print(f"Quantizing layer: {name}")
            w = module.weight.data
            if w.dim() == 4:
                O = w.size(0)
                w2 = w.reshape(O, -1)
                wq2 = quantize_weight_per_channel_absmax(w2, n_bits=n_bits)
                module.weight.data = wq2.reshape_as(w)
            else:
                module.weight.data = quantize_weight_per_channel_absmax(w, n_bits=n_bits)
            break
    return model


def evaluate_model(model, dataset, max_samples=1000, batch_size=32, device="cuda"):
    """
    Evaluates the model's accuracy on the specified task.
    """
    transform = create_transform(
        input_size=(3, 224, 224),
        is_training=False,
        mean=model.config.mean,
        std=model.config.std,
        crop_mode=model.config.crop_mode,
        crop_pct=model.config.crop_pct
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(min(len(dataset), max_samples))):
            sample = dataset[i]
            img = transform(sample["image"].convert("RGB")).unsqueeze(0).to(device).to(dtype=next(model.parameters()).dtype)
            logits = model(img)["logits"]
            pred = logits.argmax(-1).item()
            if pred == sample["label"]:
                correct += 1
            total += 1

    acc = 100 * correct / total
    print(f"Top-1 Accuracy: {100 * correct / total:.2f}%")
    return acc


def compute_sqnr(orig, quant):
    sig_p = orig.pow(2).mean()
    noise = (orig - quant).pow(2).mean()
    return 10 * torch.log10(sig_p / (noise + 1e-8)).item()


def compute_kl(orig, quant):
    p_log = F.log_softmax(orig, dim=-1)
    q_log = F.log_softmax(quant, dim=-1)
    kl_pq = F.kl_div(q_log, p_log, reduction="batchmean", log_target=True).item()
    kl_qp = F.kl_div(p_log, q_log, reduction="batchmean", log_target=True).item()
    return kl_pq, kl_qp


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Current working directory:", os.getcwd())
    pretrained = "nvidia/MambaVision-L-21K"

    set_seed(42)

    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")

    base = AutoModelForImageClassification.from_pretrained(
        pretrained, trust_remote_code=True
    ).half().to(device).eval()

    calib_x, _ = build_calib_batch(base, dataset, n=64, device=device)
    calib_x = calib_x.to(dtype=next(base.parameters()).dtype)

    local_model_dir = "quantized_model_dir"
    results_file = "mamba_vision_L_sensitivity_results_8bits.json"
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)

    quantizable_layers = [name for name, module in base.named_modules() if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))]
    print(f"Quantiable Modules: {quantizable_layers}")

    sensitivity_results = {}

    for layer_name in quantizable_layers:
        print(f"Processing layer: {layer_name}")

        teacher = AutoModelForImageClassification.from_pretrained(
            pretrained, trust_remote_code=True
        ).half().to(device).eval()

        student = AutoModelForImageClassification.from_pretrained(
            pretrained, trust_remote_code=True
        ).half().to(device).eval()

        student = quantize_layer(student, layer_name, n_bits=8) #change to 8 bits

        with torch.no_grad():
            orig_logits = teacher(calib_x)["logits"]
            quant_logits = student(calib_x)["logits"]

        sqnr = compute_sqnr(orig_logits, quant_logits)
        kl_pq, kl_qp = compute_kl(orig_logits, quant_logits)
        acc = evaluate_model(student, dataset, max_samples=1000, batch_size=32, device=device)

        print(f"Layer: {layer_name}, Accuracy: {acc}")

        sensitivity_results[layer_name] = {
            "accuracy_top1_subset": acc,
            "sqnr_db": sqnr,
            "kl_teacher_to_student": kl_pq,
            "kl_student_to_teacher": kl_qp
        }

    with open(results_file, "w") as f:
        json.dump(sensitivity_results, f, indent=4)

    print(f"Sensitivity analysis complete. Results saved to {results_file}")


if __name__ == "__main__":
    main()
