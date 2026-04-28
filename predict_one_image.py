import torch
import cv2
import numpy as np
import argparse
from models.maniqa_new import MANIQA_NEW

# KADID-10k 的25种失真类型名称（用于显示诊断结果）
DISTORTION_CLASSES = [
    "Gaussian Blur", "Lens Blur", "Motion Blur", "Color Diffusion", "Color Shift",
    "Color Quantization", "Color Saturation 1", "Color Saturation 2", "JPEG2000", "JPEG",
    "White Noise", "White Noise Color Component", "Impulse Noise", "Multiplicative Noise", "Denoise",
    "Brighten", "Darken", "Mean Shift", "Jitter", "Non-eccentricity Patch",
    "Pixelate", "Quantization", "Color Block", "High Sharpen", "Contrast Change"
]


def get_args():
    parser = argparse.ArgumentParser(description="Inference for Semantic-Aware IQA")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/epoch1.pt",
                        help="Path to model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def preprocess_image(img_path, device, crop_size=224):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    h, w, _ = img.shape
    if h < crop_size or w < crop_size:
        img = cv2.resize(img, (crop_size, crop_size))
    else:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        img = img[top:top + crop_size, left:left + crop_size]

    img = np.array(img).astype('float32') / 255.0  # [0, 1]
    img = (img - 0.5) / 0.5  # 归一化到 [-1, 1]

    img = torch.from_numpy(img).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    img = img.unsqueeze(0).to(device)  # [1, C, H, W]
    return img


def main():
    args = get_args()
    print(f"Using device: {args.device}")


    model = MANIQA_NEW(
        embed_dim=768,
        dim_mlp=768,
        patch_size=8,
        img_size=224,
        window_size=4,
        depths=[2, 2],
        num_heads=[4, 4],
        num_tab=2,
        scale=0.8,
        num_outputs=1,
        num_classes=25
    )
    model = model.to(args.device)

    # 2. 加载权重
    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        # 兼容不同的保存格式
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded checkpoint: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    # 3. 预处理图片
    img_tensor = preprocess_image(args.img_path, args.device)

    # 4. 推理
    with torch.no_grad():
        with torch.amp.autocast(device_type=args.device if args.device == 'cuda' else 'cpu', dtype=torch.bfloat16):
            score, dist_logits = model(img_tensor)

        # 处理分数
        pred_score = score.item()

        probs = torch.softmax(dist_logits, dim=1)
        top_prob, top_class_idx = torch.max(probs, dim=1)
        distortion_name = DISTORTION_CLASSES[top_class_idx.item()]

    # 5. 打印结果
    print("\n" + "=" * 40)
    print(f"✅ Predicted Quality Score:  {pred_score:.4f}")
    print(f"🔍 Diagnosed Distortion:     {distortion_name}")
    print(f"📊 Diagnosis Confidence:     {top_prob.item() * 100:.2f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()