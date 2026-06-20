from pathlib import Path;import cv2; import numpy as np;from tqdm import tqdm
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

REPO_ROOT = Path(__file__).resolve().parents[3]
SAM3_WEIGHTS = str(REPO_ROOT / "codes" / "sam3.pt")

# Prompts para capturar moras de distintos colores/madurez
MORA_PROMPTS = [
    "diseased blackberry fruit",
    "red blackberry",
    "green blackberry",
    "brown blackberry",
]

# ── Filtro 1: tamaño ─────────────────────────────────────────────────────────
MIN_SIDE_PX = 120       # lado más corto del bbox en px

# ── Filtro 2: forma ──────────────────────────────────────────────────────────
MAX_ASPECT_RATIO = 1.6    # relación lado_largo/lado_corto máxima (descarta ramas)

# ── Filtro 3: oclusión ───────────────────────────────────────────────────────
MAX_WHITE_PCT = 0.5     # fracción máxima de fondo blanco en el crop

# ── Filtro 4: enfoque (ratio de energías del gradiente) ─────────────────────
# Basado en laplaciano_histograma.py: Sobel -> magnitud |G| -> histograma.
# T separa |G| bajos (zonas planas/borrosas) de |G| altos (bordes nítidos).
# SAT descarta la columna saturada del histograma (borde fruta/fondo artificial).
# ratio = E_high/E_low:  alto -> enfocada,  bajo -> borrosa.
G_THRESH        = 100   # threshold T que separa G bajos de G altos
G_SAT           = 300   # |G| >= G_SAT se descarta (borde fruta/fondo artificial)
MIN_FOCUS_RATIO = 0.3   # ratio mínimo para aceptar la mora como enfocada (ajustar según dataset)


def build_predictor(conf=0.25, iou=0.75):
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
    """NMS manual: conserva la caja de mayor confianza entre pares con IoU > umbral."""
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
            union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
            if union > 0 and inter / union > iou_threshold:
                es_duplicado = True; break
        if not es_duplicado:
            kept.append(int(i))
    return kept


def white_percentage(out_img):
    """Fracción (0-1) de píxeles puramente blancos en el crop."""
    return (out_img == 255).all(axis=2).mean()


def is_compact_shape(region, max_aspect_ratio=MAX_ASPECT_RATIO):
    """True si el bbox no es demasiado alargado (descarta ramas/palos)."""
    h, w = region.shape[:2]
    if min(h, w) == 0:
        return False
    return max(h, w) / min(h, w) <= max_aspect_ratio


def is_big_enough(region, min_side=MIN_SIDE_PX):
    """True si el bbox supera el tamaño mínimo en px."""
    h, w = region.shape[:2]
    return min(h, w) >= min_side


def is_focused(out_img, thresh=G_THRESH, sat=G_SAT, min_ratio=MIN_FOCUS_RATIO):
    """True si la mora está enfocada según el ratio de energías del gradiente.

    Aplica Sobel sobre la escala de grises del crop, luego:
      - Excluye el fondo blanco (>= 230 en cualquier canal).
      - Excluye píxeles con |G| >= sat (borde fruta/fondo artificial).
      - E_low  = Σ|G|²  para  |G| < thresh  (zonas planas / blur)
      - E_high = Σ|G|²  para  |G| >= thresh (bordes nítidos)
      - ratio  = E_high / E_low
    Si ratio < min_ratio -> borrosa (descartar).  Si ratio >= min_ratio -> nítida.
    """
    gray_f = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Gx = cv2.Sobel(gray_f, cv2.CV_32F, dx=1, dy=0, ksize=3)
    Gy = cv2.Sobel(gray_f, cv2.CV_32F, dx=0, dy=1, ksize=3)
    mag = np.sqrt(Gx**2 + Gy**2)

    fruit_mask = ~np.all(out_img >= 230, axis=2)   # excluir fondo blanco
    g = mag[fruit_mask]
    g = g[g < sat]                                  # excluir columna saturada

    if len(g) < 50:     # muy pocos píxeles de fruto -> descartar por seguridad
        return False

    E_low  = float(np.sum(g[g <  thresh] ** 2))
    E_high = float(np.sum(g[g >= thresh] ** 2))
    if E_low == 0:
        return True     # todo es borde fuerte -> definitivamente enfocada
    return (E_high / E_low) >= min_ratio


def segment_image(predictor, image_path, out_dir, discarded_dir=None):
    """Segmenta las moras de UNA imagen. Guarda una PNG por mora que pase los 4 filtros."""
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if discarded_dir is not None:
        discarded_dir = Path(discarded_dir)
        discarded_dir.mkdir(parents=True, exist_ok=True)

    predictor.set_image(str(image_path))
    results = predictor(text=MORA_PROMPTS)

    # Liberar la caché GPU antes de trabajar con los resultados.
    # SAM3 deja tensores intermedios (encoder features) reservados en la caché
    # de PyTorch; si no se limpian, se acumulan entre imágenes y fragmentan la
    # VRAM hasta que no hay bloque contiguo suficientemente grande.
    torch.cuda.empty_cache()

    if not results:
        return 0
    result = results[0]
    if result.boxes is None or result.masks is None:
        return 0

    original = cv2.imread(str(image_path))
    boxes  = result.boxes.xyxy.float().cpu().numpy().astype(int)
    scores = result.boxes.conf.float().cpu().numpy()
    masks  = result.masks.data.cpu().numpy() > 0.5

    keep = remove_duplicates(boxes, scores, iou_threshold=0.5)

    saved = 0; discarded = 0
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        region = original[y1:y2, x1:x2]
        mask   = masks[i, y1:y2, x1:x2]
        if not mask.any():
            continue
        out_img = np.full_like(region, 255)   # fondo blanco
        out_img[mask] = region[mask]

        # Filtros en cascada: cada uno añade un motivo de descarte
        if not is_compact_shape(region):
            reason = "forma"
        elif not is_big_enough(region):
            reason = "tamano"
        elif white_percentage(out_img) > MAX_WHITE_PCT:
            reason = "ocluida"
        elif not is_focused(out_img):
            reason = "blur"
        else:
            reason = None

        if reason is None:
            cv2.imwrite(str(out_dir / f"{image_path.stem}_{saved:02d}.png"), out_img)
            saved += 1
        elif discarded_dir is not None:
            cv2.imwrite(
                str(discarded_dir / f"{image_path.stem}_{discarded:02d}_{reason}.png"),
                out_img,
            )
            discarded += 1
    return saved


def segment_folder(predictor, folder, out_dir, discarded_dir=None):
    """Segmenta todas las .png de una carpeta. Devuelve total de moras guardadas."""
    folder = Path(folder)
    images = sorted(folder.glob("*.png"))
    total = 0
    for img in tqdm(images, desc=folder.name):
        total += segment_image(predictor, img, out_dir, discarded_dir)
    return total


if __name__ == "__main__":
    import os
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # reduce fragmentación VRAM

    INPUT   = "/home/joseph/Documents/MERCOIA/Algorithm/Our_dataset/Moras/original"
    OUT_DIR = REPO_ROOT / "Our_dataset" / "Moras" / "Seleccionadas"
    DESCARTE = REPO_ROOT / "Our_dataset" / "Moras" / "Descartadas"

    predictor = build_predictor(conf=0.45, iou=0.9)
    INPUT = Path(INPUT)

    if INPUT.is_file():
        n = segment_image(predictor, INPUT, OUT_DIR, discarded_dir=DESCARTE)
    elif INPUT.is_dir():
        n = segment_folder(predictor, INPUT, OUT_DIR, discarded_dir=DESCARTE)
    else:
        raise FileNotFoundError(INPUT)

    print(f"{n} moras segmentadas en {OUT_DIR}")
    print(f"descartadas (forma/tamano/ocluida/blur) en {DESCARTE}")
