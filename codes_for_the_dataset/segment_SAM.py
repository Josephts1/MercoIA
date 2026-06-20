import cv2
import numpy as np
from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor

IMAGE_PATH  = "/home/joseph/Documents/MERCOIA/dataset/Pinas/WhatsApp Image 2026-05-19 at 3.45.45 PM.jpeg"
TEXT_PROMPT = "Pineapple"
OUTPUT_DIR  = "/home/joseph/Documents/MERCOIA/dataset/Pinas"

SAM_CHECKPOINT = "/home/joseph/Documents/MERCOIA/Algorithm/codes/sam3.pt"

predictor = SAM3SemanticPredictor(overrides=dict(
    conf=0.5,
    task="segment",
    mode="predict",
    model=SAM_CHECKPOINT,
    save=False,
    iou=0.95,
))

predictor.set_image(IMAGE_PATH)
results = predictor(text=[TEXT_PROMPT])

# Guardar cada máscara detectada como imagen separada
image    = cv2.imread(IMAGE_PATH)
white_bg = np.full_like(image, 255)
masks    = results[0].masks.data.cpu().numpy()
stem     = Path(IMAGE_PATH).stem

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"Máscaras detectadas: {len(masks)}")
for i, mask in enumerate(masks):
    mask     = mask.astype(bool)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    result   = np.where(mask_3ch, image, white_bg).astype(np.uint8)

    ys, xs  = np.where(mask)
    result  = result[ys.min():ys.max()+1, xs.min():xs.max()+1]

    out_path = str(Path(OUTPUT_DIR) / f"{stem}_mask{i+1}.png")
    cv2.imwrite(out_path, result)
    print(f"  [{i+1}] Guardado en: {out_path}")
