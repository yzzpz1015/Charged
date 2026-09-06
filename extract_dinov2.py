"""Extract DINOv2 embeddings from satellite mosaic images."""
import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', required=True)
    ap.add_argument('--tiles_root', default='tiles_dataset')
    ap.add_argument('--zoom', default='z16')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    mosaic_dir = os.path.join(args.tiles_root, args.city, 'mosaic', args.zoom)
    out_dir = os.path.join(args.tiles_root, args.city, 'embedding', args.zoom)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'sat_embed_dinov2_768.npy')

    files = sorted(
        [f for f in os.listdir(mosaic_dir) if f.endswith('.png')],
        key=lambda x: int(x.split('.')[0])
    )
    print(f'{args.city}: {len(files)} mosaic images')

    # Load DINOv2 ViT-B/14
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(args.device).eval()

    # 1536 -> resize to multiple of 14 close to 518
    tfm = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeds = []
    with torch.no_grad():
        for f in files:
            img = Image.open(os.path.join(mosaic_dir, f)).convert('RGB')
            x = tfm(img).unsqueeze(0).to(args.device)
            # CLS token
            feat = model(x).cpu().numpy()[0]  # [768]
            embeds.append(feat)
            print(f'  {f}: shape={feat.shape}')

    arr = np.stack(embeds, axis=0)  # [N, 768]
    np.save(out_path, arr)
    print(f'\nSaved: {out_path}, shape={arr.shape}')


if __name__ == '__main__':
    main()
