import os
import json
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate
from quant_utils import quantize_weight_per_channel_absmax


def _load_mamba2(pretrained):
    """
    Load state-spaces/mamba2-130m (non-hf) with AutoModelForCausalLM.
    The non-hf config.json uses the mamba_ssm schema (d_model) instead of
    the transformers Mamba2Config schema (hidden_size).  When hidden_size is
    absent, AutoConfig silently defaults it to 4096, building the wrong
    architecture.  We patch d_model -> hidden_size before constructing the
    model to avoid the size-mismatch error.
    """
    cfg = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    if hasattr(cfg, "d_model") and getattr(cfg, "hidden_size", None) != cfg.d_model:
        cfg.hidden_size = cfg.d_model
    return AutoModelForCausalLM.from_pretrained(pretrained, config=cfg, trust_remote_code=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_pile(nsamples, seed, seqlen, model, tokenizer_name=""):
    traindata = load_dataset(
        "json",
        data_files="/cpfs01/user/chenmengzhao/prompt_quantization/val.jsonl.zst",
        split="train"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model, use_fast=True)
    enc = tokenizer("\n\n".join(traindata["text"][:1000]), return_tensors="pt")

    set_seed(seed)
    loader = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader, None


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer_name=""):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model, use_fast=True)
    enc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    set_seed(seed)
    loader = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader, enc


def get_ptb(nsamples, seed, seqlen, model, tokenizer_name=""):
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model, use_fast=True)
    enc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")

    set_seed(seed)
    loader = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader, enc


def get_c4(nsamples, seed, seqlen, model, tokenizer_name=""):
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model, use_fast=True)

    set_seed(seed)
    loader = []
    for _ in range(nsamples):
        while True:
            idx = random.randint(0, len(traindata) - 1)
            enc = tokenizer(traindata[idx]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader, None


def get_ptb_new(nsamples, seed, seqlen, model, tokenizer_name=""):
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata  = load_dataset("ptb_text_only", "penn_treebank", split="test")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model, use_fast=True)

    train_enc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    test_enc  = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    set_seed(seed)
    loader = []
    for _ in range(nsamples):
        i = random.randint(0, train_enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = train_enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))
    return loader, test_enc


def get_c4_new(nsamples, seed, seqlen, model, tokenizer_name=""):
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train"
    )
    valdata   = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model, use_fast=True)

    set_seed(seed)
    loader = []
    for _ in range(nsamples):
        while True:
            idx = random.randint(0, len(traindata) - 1)
            enc = tokenizer(traindata[idx]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        loader.append((inp, tar))

    val_enc = tokenizer(" ".join(valdata["text"][:1100]), return_tensors="pt")
    val_enc = val_enc.input_ids[:, : (256 * seqlen)]
    return loader, val_enc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model="", tokenizer_name=""):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer_name)
    if "pile" in name:
        return get_pile(nsamples, seed, seqlen, model, tokenizer_name)
    if "ptb" in name and "new" not in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer_name)
    if "ptb_new" in name:
        return get_ptb_new(nsamples, seed, seqlen, model, tokenizer_name)
    if "c4_new" in name:
        return get_c4_new(nsamples, seed, seqlen, model, tokenizer_name)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer_name)
    raise ValueError(f"Unknown loader name: {name}")


def compute_sqnr(orig, quant):
    sig_p = orig.pow(2).mean()
    noise = (orig - quant).pow(2).mean()
    return 10 * torch.log10(sig_p / (noise + 1e-8)).item()


def compute_kl(orig, quant):
    p_log = F.log_softmax(orig, dim=-1)
    q_log = F.log_softmax(quant, dim=-1)
    p      = p_log.exp()
    kl_pq  = F.kl_div(q_log, p, reduction="batchmean", log_target=True).item()
    kl_qp  = F.kl_div(p_log, F.log_softmax(quant, dim=-1), reduction="batchmean", log_target=True).item()
    return kl_pq, kl_qp


def compute_delta_ce(orig, quant, targets):
    V       = orig.size(-1)
    loss_o  = F.cross_entropy(orig.view(-1, V), targets.view(-1), reduction="mean")
    loss_q  = F.cross_entropy(quant.view(-1, V), targets.view(-1), reduction="mean")
    return (loss_q - loss_o).item()


def eval_ppl(model, tokenizer, tmp_dir="tmp_lmeval", task="wikitext", batch_size=8):
    # always wipe the dir so stale configs from previous/different models don't bleed in
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    wrapped = HFLM(
        pretrained=tmp_dir,
        tokenizer=tokenizer,
        parallelize=False,
        device_map=None,
        trust_remote_code=True
    )
    out = simple_evaluate(
        model=wrapped,
        tasks=[task],
        num_fewshot=0,
        batch_size=batch_size
    )
    return out["results"][task]["word_perplexity,none"]


def quantize_layer(model, layer_name, n_bits=4):
    for nm, mod in model.named_modules():
        if nm == layer_name and hasattr(mod, "weight"):
            mod.weight.data = quantize_weight_per_channel_absmax(mod.weight.data, n_bits=n_bits)
            break
    return model


def main():
    device         = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained     = "state-spaces/mamba2-130m"
    # mamba2-130m has no bundled tokenizer; use the GPT-NeoX one it was trained with
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    set_seed(42)

    # get 64 samples of length 512 from Wikitext-2 train
    trainloader, _ = get_loaders(
        name           = "wikitext2",
        nsamples       = 64,
        seed           = 42,
        seqlen         = 512,
        model          = pretrained,
        tokenizer_name = tokenizer_name,
    )

    inputs  = torch.cat([inp for inp, _ in trainloader], dim=0).to(device)
    targets = torch.cat([tar for _, tar in trainloader], dim=0).to(device)

    # find all linear layers
    base   = _load_mamba2(pretrained).half()
    layers = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)]
    print(f"Found {len(layers)} linear layers.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    results   = {}
    tmp_dir   = "quant_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    for layer in layers:
        print(f"→ Layer: {layer}")

        teacher = _load_mamba2(pretrained).half().to(device).eval()
        student = _load_mamba2(pretrained).half().to(device).eval()

        student = quantize_layer(student, layer, n_bits=4)

        with torch.no_grad():
            orig_logits  = teacher(input_ids=inputs).logits
            quant_logits = student(input_ids=inputs).logits

        sqnr          = compute_sqnr(orig_logits, quant_logits)
        kl_pq, kl_qp = compute_kl(orig_logits, quant_logits)
        delta_ce      = compute_delta_ce(orig_logits, quant_logits, targets)
        ppl           = eval_ppl(student, tokenizer, tmp_dir)

        results[layer] = {
            "perplexity":            ppl,
            "sqnr_db":               sqnr,
            "kl_teacher_to_student": kl_pq,
            "kl_student_to_teacher": kl_qp,
            "delta_cross_entropy":   delta_ce,
        }

    with open("sensitivity_results_mamba2-130m_4bits.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
