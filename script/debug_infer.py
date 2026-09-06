"""Single-batch debug: print predictions vs labels for the LLM-backbone EV-VLM."""
import torch, sys, os
sys.path.insert(0, '/home/data/yzzpz/Charged')

from api.dataset.common import EVDataset
from api.model.config import PredictionModel

device = torch.device('cuda:0')

# Load dataset
ev_dataset = EVDataset('volume', 'None', data_path='data/MEL_remove_zero/')
ev_dataset.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
ev_dataset.create_loaders(
    seq_l=72, pre_len=6, batch_size=32, device=device, model_name='ev_vlm',
    tiles_path='tiles_dataset/', city='MEL', image_size=224, skip_images=False,
    gpt2_model_path='api/model/weights/gpt2_model/',
    siglip_model_path='api/model/weights/siglip2_model/',
)

# Build model (same as training)
num_node = ev_dataset.feat.shape[1]
ev_model = PredictionModel(
    num_node=num_node, n_fea=1, model_name='ev_vlm',
    seq_l=72, pre_len=6,
    gpt2_model_path='api/model/weights/gpt2_model/',
    siglip_model_path='api/model/weights/siglip2_model/',
)
ev_model.model = ev_model.model.to(device)
ev_model.model.eval()

# Get one training batch and one validation batch
train_iter = iter(ev_dataset.train_loader)
valid_iter = iter(ev_dataset.valid_loader)

for name, loader_iter in [('TRAIN', train_iter), ('VALID', valid_iter)]:
    batch = next(loader_iter)
    feat, label, shared_ids, time_ids, time_mask, static_ids, static_mask, stats_ids, stats_mask, vision_emb = batch
    feat = feat.to(device)
    label = label.to(device)
    shared_ids = shared_ids.to(device)
    time_ids = time_ids.to(device)
    time_mask = time_mask.to(device)
    static_ids = static_ids.to(device)
    static_mask = static_mask.to(device)
    stats_ids = stats_ids.to(device)
    stats_mask = stats_mask.to(device)
    vision_emb = vision_emb.to(device)

    with torch.no_grad():
        preds = ev_model.model(
            feat, shared_ids=shared_ids, time_ids=time_ids, time_mask=time_mask,
            static_ids=static_ids, static_mask=static_mask,
            stats_ids=stats_ids, stats_mask=stats_mask, vision_emb=vision_emb,
        )

    # Print first 3 stations, first 2 samples
    B, N, H = preds.shape
    print(f"\n=== {name} BATCH (shape: preds={preds.shape}, label={label.shape}) ===")
    for b in range(min(2, B)):
        for n in range(min(3, N)):
            p = preds[b, n].cpu().tolist()
            l = label[b, n].cpu().tolist()
            lst = feat[b, n, -1].item()
            print(f"  sample={b} site={n}: last={lst:.4f} | pred={[f'{v:.4f}' for v in p]} | label={[f'{v:.4f}' for v in l]}")

    print(f"  pred mean={preds.mean():.4f} std={preds.std():.4f}")
    print(f"  label mean={label.mean():.4f} std={label.std():.4f}")
    loss = torch.nn.HuberLoss(delta=1.0)(preds, label)
    print(f"  HuberLoss={loss:.4f}")
    last_val = feat[:, :, -1].unsqueeze(-1).expand(-1, -1, H)
    base_loss = torch.nn.HuberLoss(delta=1.0)(last_val, label)
    print(f"  Last-value baseline loss={base_loss:.4f}")
