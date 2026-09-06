# -*- coding: utf-8 -*-
"""Generate natural-language descriptions of each site's satellite mosaic image.

Third external path of the corrected L2 design (POI / satellite image / satellite
image description). A VLM (Qwen2-VL-2B-Instruct) reads the 1536x1536 site mosaic
and produces a short text description; cached per city for reuse.

Usage:
    python3 script/gen_sat_captions.py --city MEL --device 0 [--limit 62]
"""
import argparse
import glob
import json
import os

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated PNGs (e.g. site 3.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='MEL')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0, help='0 = all sites')
    parser.add_argument('--model_path', default='/home/guest3/modelscope/Qwen2-VL-2B-Instruct')
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    mosaic_dir = f'tiles_dataset/{args.city}/mosaic/z16'
    imgs = sorted(glob.glob(f'{mosaic_dir}/*.png'))
    if args.limit > 0:
        imgs = imgs[:args.limit]
    print(f'[caption] {len(imgs)} images from {mosaic_dir}')

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    model_name = args.model_path
    print(f'[caption] loading {model_name} ...')
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device).eval()

    PROMPT = ("This is a satellite view of an EV charging station location. "
              "Reply with ONLY 4 short comma-separated items, no sentences: "
              "dominant_land_use, building_density, transport_access, notable_features. "
              "Max 5 words per item. Be specific to THIS site, not generic.")

    captions = {}
    for i, p in enumerate(imgs):
        site_id = os.path.basename(p).split('.')[0]  # mosaic named by site id
        try:
            im = Image.open(p).convert('RGB')
        except Exception as e:
            print(f'  !! skip corrupt image {os.path.basename(p)}: {e}')
            captions[site_id] = 'unreadable satellite image'
            continue
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': im},
            {'type': 'text', 'text': PROMPT},
        ]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[im], return_tensors='pt').to(device).to(torch.bfloat16)
        with torch.no_grad():
            torch.manual_seed(42)
            gen = model.generate(
                **inputs, max_new_tokens=48, do_sample=True, temperature=0.8, top_p=0.9)
        cap = processor.batch_decode(gen[:, inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True)[0].strip()
        captions[site_id] = cap
        if (i + 1) % 10 == 0 or i == len(imgs) - 1:
            print(f'  [{i+1}/{len(imgs)}] site {site_id}: {cap[:90]}...')
            # incremental save
            os.makedirs(f'tiles_dataset/{args.city}/embedding/z16', exist_ok=True)
            with open(f'tiles_dataset/{args.city}/embedding/z16/sat_captions.json', 'w') as f:
                json.dump(captions, f, ensure_ascii=False, indent=2)

    print(f'[caption] done: {len(captions)} captions -> '
          f'tiles_dataset/{args.city}/embedding/z16/sat_captions.json')

if __name__ == '__main__':
    main()
