"""Clasificador de enfermedades — EfficientNet-B0 entrenado (Proyecto MERCOIA).

Envuelve el checkpoint  models/mejor_modelo.pth  producido por codes/efficenet.py
y expone una API mínima para clasificar los recortes que devuelve la segmentación
de SAM 3, sin tocar el disco.

El preprocesamiento replica EXACTAMENTE el transform de evaluación del
entrenamiento (Resize -> CenterCrop -> ToTensor -> Normalize con ImageNet); si
se cambia allí, hay que cambiarlo aquí o las predicciones se degradan en
silencio.

Uso típico:
    import clasificador
    clasificador.cargar_modelo()                       # una vez, al arrancar
    preds = clasificador.clasificar_lote(crops_bgr,    # lista de numpy BGR
                                         clases_permitidas=
                                         clasificador.clases_de_fruta("banana"))
    # preds[i] -> [("banano_racimo_sano", 0.94), ("banano_fusarium", 0.03), ...]
"""

from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# .../Algorithm/codes/Dataset_codes/Deteccion_y_filtrado/clasificador.py
#      parents[3] = .../Algorithm
REPO_ROOT = Path(__file__).resolve().parents[3]
MODELO_PATH = REPO_ROOT / "models" / "mejor_modelo.pth"

# Prefijo de clase que corresponde a cada tipo de fruta del segmentador.
# Sirve para restringir la predicción a las clases plausibles: si el usuario
# pidió segmentar bananos, no tiene sentido predecir "mora_antracnosis".
PREFIJO_POR_FRUTA = {"banana": "banano", "mora": "mora", "pina": "pina"}

# Nombre legible del fruto para los mensajes del chat.
_FRUTA_BONITA = {"banano": "banano", "mora": "mora", "pina": "piña"}

# Estado del módulo (se carga una sola vez).
_MODELO = None
_TRANSFORM = None
_CLASES = None
_DEVICE = None


def cargar_modelo(ruta=None, device=None):
    """Carga el checkpoint en memoria. Idempotente: la segunda llamada no hace nada.

    Devuelve (clases, device) para que quien llame pueda loguearlo.
    """
    global _MODELO, _TRANSFORM, _CLASES, _DEVICE
    if _MODELO is not None:
        return _CLASES, _DEVICE

    ruta = Path(ruta) if ruta else MODELO_PATH
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe el modelo entrenado en {ruta}\n"
            f"    Entrénalo primero:  python codes/efficenet.py"
        )

    _DEVICE = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = torch.load(ruta, map_location=_DEVICE, weights_only=False)

    _CLASES = ckpt["clases"]
    img_size = ckpt.get("img_size", 224)
    media = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
    desv = ckpt.get("normalize_std", [0.229, 0.224, 0.225])

    modelo = models.efficientnet_b0(weights=None)
    modelo.classifier[1] = nn.Linear(
        modelo.classifier[1].in_features, len(_CLASES)
    )
    modelo.load_state_dict(ckpt["model_state_dict"])
    modelo.eval().to(_DEVICE)
    _MODELO = modelo

    # Mismo transform que tf_eval en el entrenamiento.
    _TRANSFORM = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(media, desv),
    ])

    return _CLASES, _DEVICE


def clases_de_fruta(fruit_type):
    """Clases del modelo que pertenecen a un tipo de fruta del segmentador.

    Devuelve None si la fruta no se reconoce (=> no restringir nada).
    """
    prefijo = PREFIJO_POR_FRUTA.get(fruit_type)
    if prefijo is None or _CLASES is None:
        return None
    permitidas = [c for c in _CLASES if c.startswith(prefijo + "_")]
    return permitidas or None


def nombre_bonito(clase):
    """'banano_Sigatoka negra' -> 'Sigatoka negra (banano)'."""
    fruta, _, resto = clase.partition("_")
    resto = resto.replace("_", " ") or clase
    return f"{resto} ({_FRUTA_BONITA.get(fruta, fruta)})"


def _preparar(crop_bgr):
    """Recorte BGR de OpenCV -> tensor normalizado listo para la red."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return _TRANSFORM(Image.fromarray(rgb))


def clasificar_lote(crops_bgr, topk=3, clases_permitidas=None):
    """Clasifica una lista de recortes BGR en UNA sola pasada por la red.

    - crops_bgr: lista de numpy arrays BGR (los que devuelve segment_array).
    - clases_permitidas: lista de nombres de clase a los que restringir la
      predicción; el resto se anula antes del softmax, así las probabilidades
      se renormalizan sobre las clases plausibles.

    Devuelve una lista paralela a crops_bgr; cada elemento es una lista de
    tuplas (clase, probabilidad) ordenada de mayor a menor, de longitud topk.
    """
    if not crops_bgr:
        return []
    if _MODELO is None:
        cargar_modelo()

    lote = torch.stack([_preparar(c) for c in crops_bgr]).to(_DEVICE)

    with torch.no_grad():
        logits = _MODELO(lote).float()

        if clases_permitidas:
            permitidas = set(clases_permitidas)
            mascara = torch.tensor(
                [c not in permitidas for c in _CLASES],
                device=logits.device,
            )
            # -inf antes del softmax => probabilidad 0 y renormalización
            # automática sobre las clases plausibles.
            logits = logits.masked_fill(mascara, float("-inf"))

        probs = torch.softmax(logits, dim=1).cpu()

    k = min(topk, probs.shape[1])
    salida = []
    for fila in probs:
        valores, indices = fila.topk(k)
        salida.append([
            (_CLASES[i], float(v))
            for v, i in zip(valores.tolist(), indices.tolist())
            if v > 0.0            # descarta las clases anuladas por la máscara
        ])
    return salida


def clasificar(crop_bgr, topk=3, clases_permitidas=None):
    """Versión de un solo recorte. Devuelve [(clase, prob), ...]."""
    res = clasificar_lote([crop_bgr], topk=topk, clases_permitidas=clases_permitidas)
    return res[0] if res else []


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python clasificador.py <imagen.png> [banana|mora|pina]")
        sys.exit(1)

    clases, device = cargar_modelo()
    print(f"Modelo cargado en {device} — {len(clases)} clases")

    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit(f"No pude leer {sys.argv[1]}")

    permitidas = clases_de_fruta(sys.argv[2]) if len(sys.argv) > 2 else None
    for clase, prob in clasificar(img, topk=3, clases_permitidas=permitidas):
        print(f"  {nombre_bonito(clase):35s} {prob:6.1%}")
