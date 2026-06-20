from pathlib import Path;import cv2
import numpy as np
from tqdm import tqdm
from ultralytics.models.sam import SAM3SemanticPredictor


REPO_ROOT = Path(__file__).resolve().parents[2]
SAM3_WEIGHTS = str(REPO_ROOT / "sam3.pt")

# Varios prompts para capturar piñas en distintos estados de madurez.
PINEAPPLE_PROMPTS = [
    "Banana bunch",
    "green banana bunch",
    "unripe banana bunch",
]

# % máximo de píxeles blancos (fondo) permitido en el crop final.
# Si una piña está muy ocluida, su crop tiene mucho blanco -> se descarta.
# Más permisivo que con moras (0.4) porque las piñas tienen corona y formas
# irregulares que dejan más fondo blanco dentro del bbox aunque estén completas.
MAX_WHITE_PCT = 0.58

# Fracción mínima del área total de la máscara que debe ocupar el componente
# conectado MÁS GRANDE. Si está por debajo, la máscara está fragmentada
# (piña + pedazos sueltos, dos piñas fusionadas, etc.) -> se descarta.
# 0.85 permite la corona algo separada del cuerpo, pero pesca fragmentos espurios.
MIN_MAIN_BLOB_RATIO = 0.85


def build_predictor(conf=0.75, iou=0.75):
    """Carga el modelo SAM 3. Llamar una sola vez por sesión (pesa 3.4 GB)."""
    overrides = dict(
        conf=conf, task="segment", mode="predict",
        model=SAM3_WEIGHTS, half=True,
        save=False, save_txt=False, verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.args.iou = iou
    return predictor


def remove_duplicates(boxes, scores, iou_threshold=0.5):
    """Filtra detecciones repetidas (varios prompts marcan la misma piña).

    Recorre las cajas de mayor a menor confianza y conserva una sólo si NO
    se solapa demasiado (IoU > iou_threshold) con otra ya conservada.
    """
    order = np.argsort(-scores)
    kept = []
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        es_duplicado = False
        for k in kept:
            kx1, ky1, kx2, ky2 = boxes[k]
            inter_w = max(0, min(x2, kx2) - max(x1, kx1))
            inter_h = max(0, min(y2, ky2) - max(y1, ky1))
            inter = inter_w * inter_h
            union = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - inter
            if union > 0 and inter / union > iou_threshold:
                es_duplicado = True
                break
        if not es_duplicado:
            kept.append(int(i))
    return kept


def white_percentage(out_img):
    """Fracción (0 a 1) de píxeles puramente blancos en la imagen final."""
    return (out_img == 255).all(axis=2).mean()


def main_blob_ratio(mask):
    """Fracción del área de la máscara ocupada por el componente conectado más grande.

    1.0 = máscara de una sola pieza (piña limpia).
    <1.0 = hay fragmentos sueltos (piña ocluida, dos piñas fusionadas, etc.).
    """
    num, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return 0.0
    # stats[0] es el fondo; los demás son los componentes de la máscara.
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(areas.max() / areas.sum())


def segment_image(predictor, image_path, out_dir, discarded_dir=None):
    """Segmenta las piñas de una imagen. Guarda una PNG por cada detección.

    Si `discarded_dir` es una ruta, los crops con demasiado fondo blanco
    (piñas ocluidas / partes de piña) se guardan ahí para poder revisarlos.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if discarded_dir is not None:
        discarded_dir = Path(discarded_dir)
        discarded_dir.mkdir(parents=True, exist_ok=True)

    predictor.set_image(str(image_path))
    results = predictor(text=PINEAPPLE_PROMPTS)
    if not results:
        return 0
    result = results[0]
    if result.boxes is None or result.masks is None:
        return 0

    original = cv2.imread(str(image_path))
    boxes = result.boxes.xyxy.float().cpu().numpy().astype(int)
    scores = result.boxes.conf.float().cpu().numpy()
    masks = result.masks.data.cpu().numpy() > 0.5

    keep = remove_duplicates(boxes, scores, iou_threshold=0.5)

    saved = 0
    discarded = 0
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        region = original[y1:y2, x1:x2]
        mask = masks[i, y1:y2, x1:x2]
        if not mask.any():
            continue
        out_img = np.full_like(region, 255)  # fondo blanco
        out_img[mask] = region[mask]

        cv2.imwrite(str(out_dir / f"{image_path.stem}_{saved:02d}.png"), out_img)
        saved += 1
        if discarded_dir is not None:
            cv2.imwrite(
                str(discarded_dir / f"{image_path.stem}_{discarded:02d}.png"),
                out_img,
            )
            discarded += 1
    return saved


def segment_folder(predictor, folder, out_dir, discarded_dir=None):
    """Procesa todas las imágenes (.jpg/.jpeg/.png) de una carpeta."""
    folder = Path(folder)
    images = sorted(
        list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    )
    total = 0
    for img in tqdm(images, desc=folder.name):
        total += segment_image(predictor, img, out_dir, discarded_dir)
    return total


if __name__ == "__main__":
    # Pega aquí la ruta de UNA imagen o de UNA carpeta.
    INPUT = "/home/joseph/Documents/MERCOIA/Algorithm/Our_dataset/bananos/Original"

    OUT_DIR = REPO_ROOT / "Our_dataset" / "bananos" / "Seleccionadas"
    DESCARTE = REPO_ROOT / "Our_dataset" / "bananos" / "Descartadas"

    predictor = build_predictor(conf=0.6, iou=0.7)
    INPUT = Path(INPUT)

    if INPUT.is_file():
        n = segment_image(predictor, INPUT, OUT_DIR, discarded_dir=DESCARTE)
    elif INPUT.is_dir():
        n = segment_folder(predictor, INPUT, OUT_DIR, discarded_dir=DESCARTE)
    else:
        raise FileNotFoundError(INPUT)
