import os; import re; import sys; import csv; import json; import time; import random
import argparse; from pathlib import Path; import numpy as np
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: solo guarda las figuras a disco
import matplotlib.pyplot as plt
import torch; import torch.nn as nn; from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------------------------------------------------------------------
# RUTAS (relativas a este archivo)
# ----------------------------------------------------------------------------------------
DIR_SCRIPT = Path(__file__).resolve().parent
DIR_DATA = DIR_SCRIPT / "data"
DIR_ETIQUETADAS = DIR_DATA / "etiquetadas"   # raíz para ImageFolder (la crea etiquetar.py)
DIR_MODELS = DIR_SCRIPT / "models"
DIR_OUTPUTS = DIR_SCRIPT / "outputs"

# Normalización estándar de ImageNet (obligatoria al usar pesos preentrenados)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Para agrupar fotos del mismo fruto a partir del nombre de archivo
RE_FRAME = re.compile(r"_\d+(?:_[a-z]+)?\.(?:png|jpe?g|bmp)$", re.IGNORECASE)

def fijar_semilla(seed: int) -> None:
    """Hace el experimento reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def llave_de_fruto(nombre_archivo: str) -> str:
    """Identificador del fruto: quita el índice de frame y el sufijo de calidad.
    Ej.: cap1_13may_11_02_blur.png -> cap1_13may_11
    """
    return RE_FRAME.sub("", nombre_archivo)


def construir_transforms(img_size: int):
    """Devuelve (transform_entrenamiento, transform_evaluacion)."""
    # Entrenamiento: augmentation moderado pensado para frutos con pocos datos.
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),            # un fruto puede estar en cualquier orientación
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # Validación/prueba: sin augmentation, solo redimensionar y centrar.
    tf_eval = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf_train, tf_eval


def dividir_por_fruto(samples, p_train=0.70, p_val=0.15, seed=42):
    """Split ESTRATIFICADO y POR FRUTO.
    - samples: lista [(ruta, clase_idx), ...] (de ImageFolder).
    - Agrupa los índices por (clase, fruto) y reparte los GRUPOS (no las fotos sueltas),
      por clase, para no mezclar frames del mismo fruto entre conjuntos.
    Devuelve (idx_train, idx_val, idx_test).
    """
    # 1) agrupar índices de imagen por (clase, fruto)
    grupos = defaultdict(list)
    for i, (ruta, clase) in enumerate(samples):
        fruto = llave_de_fruto(Path(ruta).name)
        grupos[(clase, fruto)].append(i)

    # 2) organizar los grupos por clase
    grupos_por_clase = defaultdict(list)
    for (clase, _fruto), indices in grupos.items():
        grupos_por_clase[clase].append(indices)

    rng = random.Random(seed)
    idx_train, idx_val, idx_test = [], [], []

    for clase, lista_grupos in sorted(grupos_por_clase.items()):
        lg = lista_grupos[:]
        rng.shuffle(lg)
        G = len(lg)
        # Reparto de grupos garantizando, si hay suficientes, al menos 1 en cada conjunto.
        if G == 1:
            n_tr, n_va, n_te = 1, 0, 0
        elif G == 2:
            n_tr, n_va, n_te = 1, 1, 0
        else:
            n_tr = max(1, round(p_train * G))
            n_va = max(1, round(p_val * G))
            n_te = G - n_tr - n_va
            if n_te < 1:                       # asegurar 1 en test recortando de train
                n_te = 1
                n_tr = G - n_va - n_te
        # asignar las fotos de cada grupo al conjunto correspondiente
        for grupo in lg[:n_tr]:
            idx_train += grupo
        for grupo in lg[n_tr:n_tr + n_va]:
            idx_val += grupo
        for grupo in lg[n_tr + n_va:n_tr + n_va + n_te]:
            idx_test += grupo

    return sorted(idx_train), sorted(idx_val), sorted(idx_test)


def pesos_de_clase(samples, indices, num_clases, device):
    """Pesos inversamente proporcionales a la frecuencia (para CrossEntropyLoss)."""
    conteo = Counter(samples[i][1] for i in indices)
    total = sum(conteo.values())
    pesos = []
    for c in range(num_clases):
        n = conteo.get(c, 0)
        pesos.append(total / (num_clases * n) if n > 0 else 0.0)
    return torch.tensor(pesos, dtype=torch.float, device=device), conteo


def construir_modelo(num_clases, device):
    """EfficientNet-B0 preentrenado en ImageNet con la última capa adaptada."""
    modelo = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    n_entrada = modelo.classifier[1].in_features        # 1280 en B0
    modelo.classifier[1] = nn.Linear(n_entrada, num_clases)
    return modelo.to(device)


def congelar_backbone(modelo, congelar: bool) -> None:
    """Congela (o descongela) todo menos el clasificador final."""
    for nombre, p in modelo.named_parameters():
        if not nombre.startswith("classifier"):
            p.requires_grad = not congelar


# ----------------------------------------------------------------------------------------
# ÉPOCAS DE ENTRENAMIENTO Y EVALUACIÓN
# ----------------------------------------------------------------------------------------
def correr_epoca(modelo, loader, criterio, device, optim=None, scaler=None, use_amp=False):
    """Una pasada completa. Si optim != None entrena; si no, solo evalúa.
    Devuelve (loss_promedio, accuracy, y_true, y_pred).
    """
    entrenando = optim is not None
    modelo.train() if entrenando else modelo.eval()

    perdida_total, aciertos, n = 0.0, 0, 0
    y_true, y_pred = [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.set_grad_enabled(entrenando):
            with torch.cuda.amp.autocast(enabled=use_amp):
                salida = modelo(x)
                loss = criterio(salida, y)
            if entrenando:
                optim.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()

        perdida_total += loss.item() * x.size(0)
        pred = salida.argmax(1)
        aciertos += (pred == y).sum().item()
        n += x.size(0)
        y_true += y.detach().cpu().tolist()
        y_pred += pred.detach().cpu().tolist()

    return perdida_total / max(n, 1), aciertos / max(n, 1), y_true, y_pred


# ----------------------------------------------------------------------------------------
# GRÁFICAS
# ----------------------------------------------------------------------------------------
def graficar_curvas(historia, ruta_png):
    """Curvas de loss y accuracy (train/val) por época."""
    epocas = range(1, len(historia["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epocas, historia["train_loss"], "-o", label="train")
    ax1.plot(epocas, historia["val_loss"], "-o", label="val")
    ax1.set_title("Pérdida (loss)"); ax1.set_xlabel("época"); ax1.set_ylabel("loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(epocas, historia["train_acc"], "-o", label="train")
    ax2.plot(epocas, historia["val_acc"], "-o", label="val")
    ax2.set_title("Exactitud (accuracy)"); ax2.set_xlabel("época"); ax2.set_ylabel("accuracy"); ax2.legend(); ax2.grid(True)
    fig.tight_layout(); fig.savefig(ruta_png, dpi=130); plt.close(fig)


def graficar_matriz_confusion(cm, clases, ruta_png):
    """Matriz de confusión anotada."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(clases))); ax.set_xticklabels(clases, rotation=45, ha="right")
    ax.set_yticks(range(len(clases))); ax.set_yticklabels(clases)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusión (test)")
    umbral = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > umbral else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(ruta_png, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------------------
# PRINCIPAL
# ----------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento EfficientNet-B0 (frutas).")
    parser.add_argument("--data", default=str(DIR_ETIQUETADAS),
                        help="Raíz ImageFolder con las subcarpetas de clase (data/etiquetadas).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr-head", type=float, default=1e-3, help="LR de la etapa de calentamiento.")
    parser.add_argument("--lr-finetune", type=float, default=1e-4, help="LR del afinado completo.")
    parser.add_argument("--warmup", type=int, default=3, help="Épocas con el backbone congelado.")
    parser.add_argument("--patience", type=int, default=7, help="Paciencia del early stopping.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Desactiva precisión mixta.")
    args = parser.parse_args()

    fijar_semilla(args.seed)
    DIR_MODELS.mkdir(parents=True, exist_ok=True)
    DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    print(f"Dispositivo: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type != "cuda":
        print("  [AVISO] No se detectó GPU CUDA; el entrenamiento será lento en CPU.")

    raiz = Path(args.data)
    if not raiz.is_dir() or not any(raiz.iterdir()):
        sys.exit(f"No hay datos en {raiz}.\n"
                 f"Primero etiqueta y exporta:  python etiquetar.py   y   python etiquetar.py --exportar")

    # --- Carga con ImageFolder (dos instancias: una con augmentation, otra sin) ---
    tf_train, tf_eval = construir_transforms(args.img_size)
    try:
        ds_train_full = datasets.ImageFolder(str(raiz), transform=tf_train)
        ds_eval_full = datasets.ImageFolder(str(raiz), transform=tf_eval)
    except FileNotFoundError:
        sys.exit("ImageFolder no encontró imágenes en alguna clase. Cada subcarpeta de clase "
                 "debe tener al menos 1 imagen. Revisa data/etiquetadas/.")

    clases = ds_train_full.classes
    class_to_idx = ds_train_full.class_to_idx
    num_clases = len(clases)
    print(f"\nClases ({num_clases}): {class_to_idx}")
    print(f"Total de imágenes: {len(ds_train_full)}")

    # --- Split por fruto 70/15/15 ---
    idx_tr, idx_va, idx_te = dividir_por_fruto(ds_train_full.samples, 0.70, 0.15, args.seed)
    if not idx_va:
        print("[AVISO] El conjunto de validación quedó vacío (muy pocos frutos por clase).")
    print(f"\nReparto de imágenes -> train: {len(idx_tr)} | val: {len(idx_va)} | test: {len(idx_te)}")

    # Distribución por clase en cada conjunto (para detectar clases pequeñas)
    def dist(indices):
        c = Counter(ds_train_full.samples[i][1] for i in indices)
        return {clases[k]: c.get(k, 0) for k in range(num_clases)}
    print("  train:", dist(idx_tr))
    print("  val:  ", dist(idx_va))
    print("  test: ", dist(idx_te))

    train_loader = DataLoader(Subset(ds_train_full, idx_tr), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(Subset(ds_eval_full, idx_va), batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda")) if idx_va else None
    test_loader = DataLoader(Subset(ds_eval_full, idx_te), batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=(device.type == "cuda")) if idx_te else None

    # --- Modelo, pérdida (con pesos de clase), AMP ---
    modelo = construir_modelo(num_clases, device)
    pesos, conteo_tr = pesos_de_clase(ds_train_full.samples, idx_tr, num_clases, device)
    print(f"\nPesos de clase (desbalance): "
          f"{ {clases[c]: round(pesos[c].item(), 2) for c in range(num_clases)} }")
    criterio = nn.CrossEntropyLoss(weight=pesos)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    historia = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    mejor_val_acc, mejor_val_loss, epocas_sin_mejora = -1.0, float("inf"), 0
    ruta_modelo = DIR_MODELS / "mejor_modelo.pth"

    def evaluar_validacion():
        if val_loader is None:                 # sin val: usamos el propio train como referencia
            return correr_epoca(modelo, train_loader, criterio, device)[:2]
        return correr_epoca(modelo, val_loader, criterio, device)[:2]

    def guardar_mejor():
        torch.save({
            "model_state_dict": modelo.state_dict(),
            "arquitectura": "efficientnet_b0",
            "clases": clases,
            "class_to_idx": class_to_idx,
            "num_clases": num_clases,
            "img_size": args.img_size,
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std": IMAGENET_STD,
        }, ruta_modelo)

    # =========================== ENTRENAMIENTO EN 2 ETAPAS ===========================
    print("\n=== Etapa A: calentamiento (backbone congelado) ===")
    congelar_backbone(modelo, True)
    optim = torch.optim.Adam([p for p in modelo.parameters() if p.requires_grad],
                             lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler = None
    inicio = time.time()

    for epoca in range(1, args.epochs + 1):
        if epoca == args.warmup + 1:           # pasar a la etapa B
            print("\n=== Etapa B: afinado completo (backbone descongelado) ===")
            congelar_backbone(modelo, False)
            optim = torch.optim.Adam(modelo.parameters(), lr=args.lr_finetune,
                                     weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min",
                                                                   factor=0.3, patience=3)

        tr_loss, tr_acc, _, _ = correr_epoca(modelo, train_loader, criterio, device,
                                             optim=optim, scaler=scaler, use_amp=use_amp)
        va_loss, va_acc = evaluar_validacion()
        if scheduler is not None:
            scheduler.step(va_loss)

        historia["train_loss"].append(tr_loss); historia["train_acc"].append(tr_acc)
        historia["val_loss"].append(va_loss); historia["val_acc"].append(va_acc)
        lr_actual = optim.param_groups[0]["lr"]
        print(f"Época {epoca:02d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}  lr={lr_actual:.1e}")

        # ¿mejoró? (criterio: mayor val_acc; desempate por menor val_loss)
        mejora = (va_acc > mejor_val_acc) or (va_acc == mejor_val_acc and va_loss < mejor_val_loss)
        if mejora:
            mejor_val_acc, mejor_val_loss = va_acc, va_loss
            epocas_sin_mejora = 0
            guardar_mejor()
            print(f"      ↳ nuevo mejor modelo (val_acc={va_acc:.3f}) guardado.")
        elif epoca > args.warmup:              # early stopping solo en la etapa B
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= args.patience:
                print(f"\nEarly stopping en la época {epoca} (sin mejora en {args.patience} épocas).")
                break

    print(f"\nEntrenamiento terminado en {(time.time()-inicio)/60:.1f} min. "
          f"Mejor val_acc: {mejor_val_acc:.3f}")

    # Guardar historia y curvas
    with open(DIR_OUTPUTS / "historia_entrenamiento.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["epoca", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(historia["train_loss"])):
            w.writerow([i + 1, historia["train_loss"][i], historia["train_acc"][i],
                        historia["val_loss"][i], historia["val_acc"][i]])
    graficar_curvas(historia, DIR_OUTPUTS / "curvas_entrenamiento.png")

    # =========================== EVALUACIÓN FINAL EN TEST ===========================
    if test_loader is None:
        print("\n[AVISO] No hay conjunto de test (muy pocos frutos). Se omite la evaluación final.")
        _guardar_clases_json(clases, class_to_idx)
        return

    print("\n=== Evaluación final en TEST (con el mejor modelo) ===")
    ckpt = torch.load(ruta_modelo, map_location=device)
    modelo.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, y_true, y_pred = correr_epoca(modelo, test_loader, criterio, device)
    print(f"Accuracy en test: {test_acc:.3f}")

    etiquetas = list(range(num_clases))
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
    reporte = classification_report(y_true, y_pred, labels=etiquetas,
                                    target_names=clases, zero_division=0)
    print("\nReporte de clasificación:\n", reporte)

    with open(DIR_OUTPUTS / "reporte_clasificacion.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Accuracy en test: {test_acc:.4f}\n\n{reporte}\n")
    graficar_matriz_confusion(cm, clases, DIR_OUTPUTS / "matriz_confusion.png")
    _guardar_clases_json(clases, class_to_idx)

    print(f"\nListo. Salidas en: {DIR_OUTPUTS}")
    print(f"  - curvas_entrenamiento.png\n  - matriz_confusion.png\n  - reporte_clasificacion.txt")
    print(f"Mejor modelo: {ruta_modelo}")


def _guardar_clases_json(clases, class_to_idx):
    """Guarda las clases junto al modelo (útil para predict.py)."""
    with open(DIR_MODELS / "clases.json", "w", encoding="utf-8") as fh:
        json.dump({"clases": clases, "class_to_idx": class_to_idx}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
