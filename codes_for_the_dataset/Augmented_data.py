import random
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Carpetas objetivo y cuantas imagenes nuevas generar en cada una
CLASES_OBJETIVO = {
    "mora_moho_gris":      50,
    "pina_fusariosis":     50,
    "pina_pudricion_negra": 50,
    "mora_antracnosis":    50,
}

DIR_DATA = Path(__file__).resolve().parent / "data"

# Transformaciones que se aplican de forma aleatoria a cada imagen fuente.
# No usamos ToTensor() ni Normalize() porque queremos guardar imagenes PIL, no tensores.
EXTENSIONES_VALIDAS = {".png", ".jpg", ".jpeg", ".bmp"}

transform_augmentacion = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04),
])


def obtener_imagenes_originales(carpeta: Path) -> list[Path]:
    """Devuelve solo las imagenes originales, ignorando las ya generadas por este script."""
    return [
        p for p in carpeta.iterdir()
        if p.suffix.lower() in EXTENSIONES_VALIDAS and not p.stem.startswith("aug_")
    ]

def generar_augmentaciones(carpeta: Path, cantidad: int) -> None:
    originales = obtener_imagenes_originales(carpeta)

    if not originales:
        print(f"  [AVISO] No se encontraron imagenes originales en {carpeta.name}")
        return

    print(f"\n  {carpeta.name}: {len(originales)} originales → generando {cantidad} nuevas")

    generadas = 0
    # Ciclamos sobre las originales tantas veces como sea necesario
    fuentes = (originales * ((cantidad // len(originales)) + 2))[:cantidad]
    random.shuffle(fuentes)

    for i, ruta_origen in enumerate(fuentes):
        imagen = Image.open(ruta_origen).convert("RGB")
        imagen_aug = transform_augmentacion(imagen)

        nombre_nuevo = f"aug_{i:04d}_{ruta_origen.stem}{ruta_origen.suffix}"
        ruta_destino = carpeta / nombre_nuevo
        imagen_aug.save(ruta_destino)
        generadas += 1

    print(f"  Guardadas {generadas} imagenes en {carpeta}")


def main():
    print("=== Generando augmentacion offline ===")
    for nombre_clase, cantidad in CLASES_OBJETIVO.items():
        carpeta = DIR_DATA / nombre_clase
        if not carpeta.is_dir():
            print(f"  [ERROR] Carpeta no encontrada: {carpeta}")
            continue
        generar_augmentaciones(carpeta, cantidad)

    print("\nConteo final por clase:")
    for carpeta in sorted(DIR_DATA.iterdir()):
        if carpeta.is_dir():
            total = sum(
                1 for p in carpeta.iterdir()
                if p.suffix.lower() in EXTENSIONES_VALIDAS
            )
            originales = sum(
                1 for p in carpeta.iterdir()
                if p.suffix.lower() in EXTENSIONES_VALIDAS and not p.stem.startswith("aug_")
            )
            print(f"  {carpeta.name:25s}: {originales} originales + {total - originales} aug = {total} total")


if __name__ == "__main__":
    main()
