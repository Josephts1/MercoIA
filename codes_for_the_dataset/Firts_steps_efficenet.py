from pathlib import Path; import torch; import torch.nn as nn; from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

DIR_DATA = Path(__file__).resolve().parent / "data"

#Se deben restar a original images. Media y std que usó ImageNet en entrenamiento. Obligatorios.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tf_entrenamiento = transforms.Compose([
    # Recorta una zona aleatoria entre 70%-100% del area y la lleva a 224x224.
    # Simula frutos capturados a distintas distancias y encuadres.
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),

    # Voltea con probabilidad 0.5. La enfermedad no cambia al espejarlo.
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),

    # Rota hasta +-20 grados. Las fotos de campo no siempre estan derechas.
    transforms.RandomRotation(20),

    # Varia brillo, contraste, saturacion y tono ligeramente.
    # Simula distintas condiciones de luz (manana, tarde, nublado).
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),

    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- Transformaciones para VALIDACION y TEST (sin augmentation) ---
# Evaluaciones estables y reproducibles, sin aleatoriedad.
tf_evaluacion = transforms.Compose([
    transforms.Resize(257),       # un poco mas grande que 224
    transforms.CenterCrop(224),   # recorte central exacto
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

#Una con cada transform. Ambas apuntan a las mismas imagenes en disco; solo difieren en como las procesan.
ds_train = datasets.ImageFolder(root=str(DIR_DATA), transform=tf_entrenamiento) #Para entrenar
ds_eval  = datasets.ImageFolder(root=str(DIR_DATA), transform=tf_evaluacion) #Para validar

# ds_train.samples es una lista [(ruta, clase_idx), ...] para cada imagen.
indices   = list(range(len(ds_train)))
etiquetas = [clase_idx for _, clase_idx in ds_train.samples]

# Primer corte: 15% para TEST
idx_trainval, idx_test = train_test_split(
    indices, test_size=0.15, stratify=etiquetas, random_state=42
)

# Segundo corte: del 85% restante, 17.6% para VAL  (= 15% del total)
etiquetas_trainval = [etiquetas[i] for i in idx_trainval]
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.176, stratify=etiquetas_trainval, random_state=42
)

# Subset de train usa ds_train (con augmentation).
# Subsets de val y test usan ds_eval (sin augmentation, evaluacion estable).
subset_train = Subset(ds_train, idx_train)
subset_val   = Subset(ds_eval,  idx_val)
subset_test  = Subset(ds_eval,  idx_test)

# El DataLoader empaqueta las imagenes en batches y las entrega al modelo.
# shuffle=True en train: orden aleatorio ayuda al entrenamiento.
# shuffle=False en val/test: orden fijo para metricas reproducibles.
BATCH_SIZE = 32
train_loader = DataLoader(subset_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(subset_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(subset_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Reporte del reparto
def contar_por_clase(indices_subset):
    conteo = Counter(etiquetas[i] for i in indices_subset)
    return {ds_train.classes[c]: conteo.get(c, 0) for c in range(len(ds_train.classes))}

print(f"\nReparto de imagenes:")
print(f"  Train : {len(idx_train):4d}  {contar_por_clase(idx_train)}")
print(f"  Val   : {len(idx_val):4d}  {contar_por_clase(idx_val)}")
print(f"  Test  : {len(idx_test):4d}  {contar_por_clase(idx_test)}")

EPOCHS = 90     # maximo de epocas; el early stopping puede cortar antes
LR_CABEZA = 1e-3   # learning rate etapa A (solo el clasificador)
LR_FINETUNE = 1e-4   # learning rate etapa B (todo el modelo, 10x menor)
EPOCAS_CALENTAMIENTO = 5     # cuantas epocas dura la etapa A antes de descongelar
PACIENCIA = 7      # early stopping: epocas sin mejora antes de parar
WEIGHT_DECAY = 1e-4   # penalizacion para pesos grandes, reduce overfitting
NUM_CLASES = len(ds_train.classes)   # 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargamos EfficientNet-B0 con los pesos preentrenados en ImageNet.
modelo = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# La arquitectura tiene dos partes principales:
#   modelo.features    → el backbone (capas convolucionales, NO lo tocamos)
#   modelo.classifier  → [Dropout, Linear(1280 → 1000)]
#
# Reemplazamos solo la capa Linear final para adaptarla a nuestras 6 clases.
# modelo.classifier[0] es el Dropout, lo dejamos intacto.
# modelo.classifier[1] es el Linear, ese lo reemplazamos.
n_entradas = modelo.classifier[1].in_features   # 1280 en B0
modelo.classifier[1] = nn.Linear(n_entradas, NUM_CLASES)

modelo = modelo.to(device)

# Verificacion: imprimimos la forma de la capa final para confirmar que quedo bien
print(f"\nCapa clasificadora final: {modelo.classifier[1]}")
print(f"  Entradas : {n_entradas}")
print(f"  Salidas  : {NUM_CLASES}  (una por clase)")

# =============================================================================
# PASO 5: Función de pérdida y optimizador
# =============================================================================

# Pesos de clase: inversion de frecuencia.
# Clases con menos imagenes en train reciben un peso mayor para que el modelo
# no las ignore por ser minoría.
conteo = Counter(etiquetas[i] for i in idx_train)
total_train = len(idx_train)
pesos = torch.tensor([
    total_train / (NUM_CLASES * conteo[c]) for c in range(NUM_CLASES)
], dtype=torch.float).to(device)

print("\nPesos de clase (mayor = más importante en la pérdida):")
for c, nombre in enumerate(ds_train.classes):
    print(f"  {nombre:25s}: {pesos[c].item():.3f}  ({conteo[c]} imgs en train)")

# CrossEntropyLoss: recibe los 6 puntajes crudos del modelo (logits) y la
# etiqueta real. Calcula internamente softmax + log + negativo.
criterio = nn.CrossEntropyLoss(weight=pesos)

# =============================================================================
# PASO 6: Bucle de entrenamiento (2 etapas)
# =============================================================================

def correr_epoca(loader, entrenando: bool, optimizador=None):
    """Una pasada completa por el loader. Devuelve (loss_promedio, accuracy)."""
    modelo.train() if entrenando else modelo.eval()
    perdida_total, aciertos, n = 0.0, 0, 0

    for imagenes, etiq in loader:
        imagenes, etiq = imagenes.to(device), etiq.to(device)

        with torch.set_grad_enabled(entrenando):
            puntajes = modelo(imagenes)          # shape: (batch, 6)
            loss = criterio(puntajes, etiq)

        if entrenando:
            optimizador.zero_grad()  # limpia gradientes del batch anterior
            loss.backward()          # calcula gradientes de todos los pesos
            optimizador.step()       # actualiza los pesos

        perdida_total += loss.item() * imagenes.size(0)
        predicciones  = puntajes.argmax(dim=1)   # clase con mayor puntaje
        aciertos      += (predicciones == etiq).sum().item()
        n             += imagenes.size(0)

    return perdida_total / n, aciertos / n


# --- Etapa A: solo el clasificador aprende (backbone congelado) ---
for nombre, param in modelo.named_parameters():
    if not nombre.startswith("classifier"):
        param.requires_grad = False

optimizador = torch.optim.Adam(
    [p for p in modelo.parameters() if p.requires_grad],
    lr=LR_CABEZA, weight_decay=WEIGHT_DECAY
)

mejor_val_acc    = 0.0
epocas_sin_mejora = 0
DIR_MODELS = Path(__file__).resolve().parent / "models"
DIR_MODELS.mkdir(exist_ok=True)
ruta_modelo = DIR_MODELS / "mejor_modelo.pth"

print(f"\n{'Época':>6}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>8}  {'val_acc':>7}  {'lr':>8}")

for epoca in range(1, EPOCHS + 1):

    # Al llegar a la época de fine-tuning: descongelar todo y bajar el LR
    if epoca == EPOCAS_CALENTAMIENTO + 1:
        print("\n  → Etapa B: descongelando backbone y bajando LR\n")
        for param in modelo.parameters():
            param.requires_grad = True
        optimizador = torch.optim.Adam(
            modelo.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY
        )

    tr_loss, tr_acc = correr_epoca(train_loader, entrenando=True,  optimizador=optimizador)
    va_loss, va_acc = correr_epoca(val_loader,   entrenando=False)

    lr_actual = optimizador.param_groups[0]["lr"]
    print(f"{epoca:>6}  {tr_loss:>10.4f}  {tr_acc:>9.3f}  {va_loss:>8.4f}  {va_acc:>7.3f}  {lr_actual:>8.1e}")

    # Guardar si es el mejor modelo hasta ahora
    if va_acc > mejor_val_acc:
        mejor_val_acc = va_acc
        epocas_sin_mejora = 0
        torch.save({
            "model_state_dict": modelo.state_dict(),
            "clases":           ds_train.classes,
            "class_to_idx":     ds_train.class_to_idx,
        }, ruta_modelo)
        print(f"         ↳ mejor modelo guardado (val_acc={va_acc:.3f})")
    elif epoca > EPOCAS_CALENTAMIENTO:
        epocas_sin_mejora += 1
        if epocas_sin_mejora >= PACIENCIA:
            print(f"\nEarly stopping en época {epoca} (sin mejora en {PACIENCIA} épocas).")
            break

print(f"\nEntrenamiento terminado. Mejor val_acc: {mejor_val_acc:.3f}")
