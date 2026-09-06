"""Compute PCA principal embeddings from GPT-2 wte, save to weights/.
Run once offline. Uses scikit-learn PCA.
"""
import os, torch
import numpy as np
from sklearn.decomposition import PCA

GPT2_PATH = "api/model/weights/gpt2_model"
SAVE_PATH = "api/model/weights/pca_principal_emb.pt"
N_COMPONENTS = 500


def main():
    print(f"Loading GPT-2 wte from {GPT2_PATH} ...")
    from transformers import GPT2Model
    gpt2 = GPT2Model.from_pretrained(GPT2_PATH)
    wte = gpt2.get_input_embeddings().weight.detach().float().cpu().numpy()
    # wte: [50257, 768]
    print(f"wte shape: {wte.shape}")

    print(f"Running PCA (n_components={N_COMPONENTS}) ...")
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(wte)

    explained = pca.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained*100:.1f}%")

    # Save pca.components_ (the principal directions), shape [500, 768]
    principal_emb = torch.tensor(pca.components_, dtype=torch.float32)
    print(f"principal_emb shape: {principal_emb.shape}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(principal_emb, SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
