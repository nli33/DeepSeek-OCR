"""Shared DeepSeek-OCR vision-encoder tap (see RETRIEVAL_PLAN.md).

Loads the model once and turns a page image into compressed vision-token
embeddings using the DeepEncoder (SAM + CLIP + projector), WITHOUT running the
MoE decoder. Used by:
  - scripts/encode_page.py       (M0 probe)
  - scripts/build_corpus_index.py (M1 corpus pre-encoding)

Preprocessing mirrors DeepseekOCR.infer(); feature assembly mirrors
DeepseekOCRModel.forward(). Reproduced here (not imported) so we can stop before
the decoder. If upstream remote code changes, re-check modeling_deepseekocr.py.
GPU required.
"""
import sys

import torch
from transformers import AutoModel, AutoTokenizer

# Resolution tiers, from DeepSeek-OCR-vllm/config.py. `expected` = nominal
# vision-token budget for a square page (before aspect-ratio scaling).
TIERS = {
    "tiny":   dict(base_size=512,  image_size=512,  crop_mode=False, expected=64),
    "small":  dict(base_size=640,  image_size=640,  crop_mode=False, expected=100),
    "base":   dict(base_size=1024, image_size=1024, crop_mode=False, expected=256),
    "large":  dict(base_size=1280, image_size=1280, crop_mode=False, expected=400),
    "gundam": dict(base_size=1024, image_size=640,  crop_mode=True,  expected=None),
}

MODEL_ID = "deepseek-ai/DeepSeek-OCR"


def load_model(model_id=MODEL_ID, attn_implementation="eager"):
    """Load DeepSeek-OCR on GPU in bf16. Returns (model, tokenizer, remote_module)."""
    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-OCR encoding needs a CUDA GPU (use Slurm; see HANDOFF.md).")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, _attn_implementation=attn_implementation,
        trust_remote_code=True, use_safetensors=True,
    ).eval().cuda().to(torch.bfloat16)
    mod = sys.modules[type(model).__module__]  # trust_remote_code module
    return model, tokenizer, mod


def preprocess(image, mod, base_size, image_size, crop_mode):
    """Reproduce DeepseekOCR.infer() image preprocessing -> encoder input tensors.

    Returns (images_crop, images_ori, images_spatial_crop, meta).
    """
    tf = mod.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    ImageOps = mod.ImageOps
    pad_color = tuple(int(x * 255) for x in tf.mean)

    w, h = image.size
    ratio = 1 - ((max(w, h) - min(w, h)) / max(w, h))  # aspect-ratio token scaling

    images_list, crop_list = [], []
    if crop_mode:
        if image.size[0] <= 640 and image.size[1] <= 640:
            crop_ratio = [1, 1]
        else:
            crop_raw, crop_ratio = mod.dynamic_preprocess(image)  # (crops, (cols, rows))
        global_view = ImageOps.pad(image, (base_size, base_size), color=pad_color)
        images_list.append(tf(global_view).to(torch.bfloat16))
        width_crop_num, height_crop_num = crop_ratio
        if width_crop_num > 1 or height_crop_num > 1:
            for c in crop_raw:
                crop_list.append(tf(c).to(torch.bfloat16))
        spatial = [width_crop_num, height_crop_num]
    else:
        img = image.resize((image_size, image_size)) if image_size <= 640 else image
        global_view = ImageOps.pad(img, (image_size, image_size), color=pad_color)
        images_list.append(tf(global_view).to(torch.bfloat16))
        spatial = [1, 1]

    images_ori = torch.stack(images_list, dim=0)
    images_crop = torch.stack(crop_list, dim=0) if crop_list else torch.zeros((1, 3, base_size, base_size))
    images_spatial_crop = torch.tensor([spatial], dtype=torch.long)

    n_crops = len(crop_list)
    if base_size == 1024:
        valid = int(256 * ratio)
    elif base_size == 1280:
        valid = int(400 * ratio)
    elif base_size == 640:
        valid = 100
    elif base_size == 512:
        valid = 64
    else:
        valid = 0
    if crop_mode and image_size == 640:
        valid += n_crops * 100
    meta = dict(ratio=ratio, n_crops=n_crops, valid_img_tokens=valid)
    return images_crop, images_ori, images_spatial_crop, meta


@torch.no_grad()
def encode(model, images_crop, images_ori, images_spatial_crop):
    """Reproduce DeepseekOCRModel.forward() vision path -> vision-token embeddings.

    Returns (full_tokens, content_tokens) as CPU float tensors:
      full_tokens    = exactly what the LLM sees (includes newline/separator rows)
      content_tokens = page tokens only (separators excluded), used for pooling
    """
    m = model.get_model()
    sam, vit, projector = m.sam_model, m.vision_model, m.projector
    dev, dt = next(m.parameters()).device, next(m.parameters()).dtype
    patches = images_crop.to(dev, dt)
    image_ori = images_ori.to(dev, dt)
    cols, rows = int(images_spatial_crop[0][0]), int(images_spatial_crop[0][1])

    def vis(x):
        f1 = sam(x)
        f2 = vit(x, f1)
        return projector(torch.cat((f2[:, 1:], f1.flatten(2).permute(0, 2, 1)), dim=-1))

    glob = vis(image_ori)
    _, hw, n_dim = glob.shape
    h = w = int(hw ** 0.5)
    glob_content = glob.reshape(-1, n_dim)
    glob_full = torch.cat([glob.view(h, w, n_dim),
                           m.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1).view(-1, n_dim)

    if torch.sum(patches).item() != 0:
        local = vis(patches)
        _2, hw2, n_dim2 = local.shape
        h2 = w2 = int(hw2 ** 0.5)
        local_content = local.reshape(-1, n_dim2)
        local_grid = local.view(rows, cols, h2, w2, n_dim2).permute(0, 2, 1, 3, 4).reshape(rows * h2, cols * w2, n_dim2)
        local_full = torch.cat([local_grid,
                                m.image_newline[None, None, :].expand(rows * h2, 1, n_dim2)], dim=1).view(-1, n_dim2)
        full = torch.cat([local_full, glob_full, m.view_seperator[None, :]], dim=0)
        content = torch.cat([local_content, glob_content], dim=0)
    else:
        full = torch.cat([glob_full, m.view_seperator[None, :]], dim=0)
        content = glob_content
    return full.float().cpu(), content.float().cpu()


def pooled_vector(content_tokens):
    """Mean-pool content tokens -> one L2-normalized page vector (numpy)."""
    v = content_tokens.mean(dim=0)
    return torch.nn.functional.normalize(v, dim=0).numpy()


def encode_page(model, mod, image, tier_cfg):
    """Convenience: image + tier config -> (pooled_vector, content_tokens, meta)."""
    ic, io, isc, meta = preprocess(image, mod, tier_cfg["base_size"], tier_cfg["image_size"], tier_cfg["crop_mode"])
    full, content = encode(model, ic, io, isc)
    meta = dict(meta, content_tokens=int(content.shape[0]), full_tokens=int(full.shape[0]), dim=int(content.shape[1]))
    return pooled_vector(content), content, meta
