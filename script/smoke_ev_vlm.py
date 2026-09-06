"""Smoke test for the new EV-VLM LLM-backbone model: shape + gradient flow.

Runs on random inputs (no dataset), checks:
  1. output shape [B, N, pred_len]
  2. non-None grads on trainable front-ends (PatchEmbed, QFormer, LoRA A/B,
     forecast_token, ForecastHead, RevIN affine)
  3. GPT-2 base params have requires_grad=False and grad is None
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from api.model.ev_vlm import EV_VLM
from transformers import GPT2Tokenizer

GPT2_PATH = 'api/model/weights/gpt2_model'

B, N, T, pre_len = 2, 5, 72, 6
seq_l = T

tokenizer = GPT2Tokenizer.from_pretrained(GPT2_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Build token ids mimicking the collate output
shared_text = (
    "• You are good at charging demand forecasting.\n"
    f"• Your task is forecasting the next {pre_len} steps using past {seq_l} steps.\n"
)
shared_ids = tokenizer(shared_text, return_tensors='pt', add_special_tokens=False)['input_ids']  # [1, S_shared]

time_texts = [f"• The forecast start time is 2024-01-0{i % 9 + 1} 08:00:00, which is in the morning on weekdays.\n" for i in range(B)]
time_enc = tokenizer(time_texts, padding=True, return_tensors='pt', add_special_tokens=False)
time_ids, time_mask = time_enc['input_ids'], time_enc['attention_mask']  # [B, S_time]

static_texts = [f"Nearby: restaurant: {i}, cafe: {i % 3}." for i in range(N)]
static_enc = tokenizer(static_texts, padding=True, return_tensors='pt', add_special_tokens=False)
static_ids, static_mask = static_enc['input_ids'], static_enc['attention_mask']  # [N, S_static]

stats_texts = [f"• The historical EV charging sequence ranges from {i * 0.1:.2f} to {i * 0.1 + 1.0:.2f}, with a mean value of {i * 0.1 + 0.5:.2f}.\n" for i in range(B * N)]
stats_enc = tokenizer(stats_texts, padding=True, return_tensors='pt', add_special_tokens=False)
stats_ids = stats_enc['input_ids'].reshape(B, N, -1)
stats_mask = stats_enc['attention_mask'].reshape(B, N, -1)

vision_emb = torch.randn(N, 196, 768)

model = EV_VLM(
    seq_len=seq_l,
    pred_len=pre_len,
    num_nodes=N,
    gpt2_model_path=GPT2_PATH,
)
model.train()

feat = torch.randn(B, N, T)
preds = model(feat, shared_ids, time_ids, time_mask, static_ids, static_mask, stats_ids, stats_mask, vision_emb)
print(f"preds shape = {tuple(preds.shape)}  (expected ({B}, {N}, {pre_len}))")
assert preds.shape == (B, N, pre_len), preds.shape

label = torch.randn(B, N, pre_len)
loss = torch.nn.HuberLoss(delta=1.0)(preds, label)
loss.backward()

# Collect gradient status
def has_grad(module):
    return any(p.grad is not None for p in module.parameters() if p.requires_grad)

checks = {
    'patch_embed': has_grad(model.patch_embed),
    'qformer': has_grad(model.qformer),
    'forecast_head': has_grad(model.forecast_head),
    'revin': has_grad(model.revin),
    'forecast_token': model.forecast_token.grad is not None,
}

# LoRA A/B grad on the first block
first_lora = model.gpt2.h[0].attn.c_attn
checks['lora_A'] = first_lora.A.grad is not None
checks['lora_B'] = first_lora.B.grad is not None

print("gradient check:")
for k, v in checks.items():
    print(f"  {k}: {'OK' if v else 'MISSING'}")
    assert v, f"{k} has no gradient"

# GPT-2 base params frozen, grad None
frozen_ok = True
for n, p in model.gpt2.named_parameters():
    if n.endswith('.A') or n.endswith('.B'):
        continue  # LoRA params are trainable
    if p.requires_grad:
        frozen_ok = False
        print(f"  NOT FROZEN: {n}")
        break
    if p.grad is not None:
        frozen_ok = False
        print(f"  HAS GRAD (should be None): {n}")
        break
print(f"  gpt2 base frozen & grad None: {'OK' if frozen_ok else 'FAIL'}")
assert frozen_ok

# Count trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params = {trainable:,} / total {total:,}")
print("SMOKE TEST PASSED")
