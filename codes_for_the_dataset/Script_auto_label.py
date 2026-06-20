"""
etiquetar.py — Herramienta de etiquetado asistido (Proyecto MERCOIA / Clasificación de frutas por enfermedad)
==============================================================================================================

Las imágenes están en  dataset/Moras  y  dataset/Pinas. Cada imagen recibe su propia
etiqueta de forma individual.

CLASES (9):
    moras   ->  1 = mora_sana             2 = mora_antracnosis       3 = mora_moho_gris
    pinas   ->  4 = pina_sana             5 = pina_pudricion_negra   6 = pina_fusariosis
    bananos ->  7 = banano_racimo_sano    8 = banano_racimo_enfe_1   9 = banano_racimo_enfe_2

TECLAS DURANTE EL ETIQUETADO:
    1..9  asignar clase a la imagen (solo se aceptan las del tipo de fruta mostrado)
    s     saltar esta imagen (queda sin etiquetar, lo puedes retomar luego)
    u     deshacer el último etiquetado y volver atrás
    q     guardar y salir (puedes continuar después: el progreso se guarda solo)

USO:
    1) Etiquetar (abre la ventana interactiva):
           python etiquetar.py
    2) Exportar a la estructura ImageFolder (cuando termines o cuando quieras):
           python etiquetar.py --exportar
       Esto copia cada foto a  data/etiquetadas/<clase>/  listo para train.py (Fase 4).

REQUISITOS: Python 3.9+, matplotlib y Pillow.
       pip install matplotlib Pillow
"""

import sys
import csv
import shutil
from pathlib import Path
from collections import defaultdict

DIR_SCRIPT = Path(__file__).resolve().parent  # /home/joseph/Documents/MERCOIA/Algorithm
DIR_PROYECTO = DIR_SCRIPT.parent              # /home/joseph/Documents/MERCOIA
DIR_DATASET = DIR_PROYECTO / "dataset"
DIR_DATA = DIR_SCRIPT / "data"
DIR_ETIQUETADAS = DIR_DATA / "etiquetadas"    # destino estilo ImageFolder
ARCHIVO_CSV = DIR_DATA / "etiquetas.csv"      # progreso (fuente de verdad)

# Carpetas de fruta dentro del dataset
CARPETAS_FRUTA = {"Moras": "mora", "Pinas": "pina", "Bananos": "Ba1nano"}

# Mapeo tecla -> clase, separado por tipo de fruta para evitar errores
CLASES_POR_FRUTA = {
    "Moras": {
        "1": "mora_sana",
        "2": "mora_antracnosis",
        "3": "mora_moho_gris",
    },
    "Pinas": {
        "4": "pina_sana",
        "5": "pina_pudricion_negra",
        "6": "pina_fusariosis",
    },
    "Bananos": {
        "7": "banano_racimo_sano",
        "8": "banano_racimo_enfe_1",
        "9": "banano_racimo_enfe_2",
    },
}
TODAS_LAS_CLASES = [c for d in CLASES_POR_FRUTA.values() for c in d.values()]


# ----------------------------------------------------------------------------------------
# LISTAR IMÁGENES INDIVIDUALES
# ----------------------------------------------------------------------------------------
def listar_imagenes() -> list:
    """Recorre dataset/Moras y dataset/Pinas y devuelve una lista ordenada de dicts,
    una entrada por imagen: {llave, fruta, archivos:[Path]}.
    """
    objetos = []
    for carpeta in CARPETAS_FRUTA:
        ruta = DIR_DATASET / carpeta
        if not ruta.is_dir():
            print(f"[AVISO] No se encontró la carpeta: {ruta}")
            continue
        for img in sorted(ruta.glob("*.png")):
            llave = f"{carpeta}/{img.name}"
            objetos.append({
                "llave": llave,
                "fruta": carpeta,
                "archivos": [img],
            })
    return objetos


# ----------------------------------------------------------------------------------------
# PROGRESO (CSV)
# ----------------------------------------------------------------------------------------
def cargar_progreso() -> dict:
    """Lee data/etiquetas.csv y devuelve {llave: clase} para poder reanudar."""
    etiquetas = {}
    if ARCHIVO_CSV.exists():
        with open(ARCHIVO_CSV, newline="", encoding="utf-8") as fh:
            for fila in csv.DictReader(fh):
                etiquetas[fila["objeto"]] = fila["clase"]
    return etiquetas


def guardar_progreso(etiquetas: dict, objetos: list) -> None:
    """Escribe todo el progreso a data/etiquetas.csv."""
    DIR_DATA.mkdir(parents=True, exist_ok=True)
    fruta_de = {o["llave"]: o["fruta"] for o in objetos}
    with open(ARCHIVO_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["objeto", "fruta", "clase"])
        for llave, clase in etiquetas.items():
            w.writerow([llave, fruta_de.get(llave, ""), clase])


# ----------------------------------------------------------------------------------------
# MODO ETIQUETADO (interactivo con matplotlib)
# ----------------------------------------------------------------------------------------
def etiquetar() -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
    except ImportError:
        sys.exit("Falta matplotlib. Instala con:  pip install matplotlib Pillow")

    objetos = listar_imagenes()
    if not objetos:
        sys.exit(f"No se encontraron imágenes en {DIR_DATASET}. Revisa la ruta del dataset.")

    etiquetas = cargar_progreso()
    total = len(objetos)
    print(f"Imágenes a etiquetar: {total}")
    print(f"Ya etiquetadas: {len(etiquetas)}")

    # Empezar en la primera imagen sin etiquetar
    estado = {"idx": 0}
    while estado["idx"] < total and objetos[estado["idx"]]["llave"] in etiquetas:
        estado["idx"] += 1

    fig, ax = plt.subplots(figsize=(7, 8))
    plt.subplots_adjust(top=0.84, bottom=0.18)

    def dibujar():
        ax.clear()
        # Limpiar textos de figura anteriores para evitar solapamiento
        for txt in fig.texts[:]:
            txt.remove()

        if estado["idx"] >= total:
            ax.text(0.5, 0.5, "¡Listo! Todas las imágenes etiquetadas.\n"
                              "Cierra la ventana y ejecuta:  python etiquetar.py --exportar",
                    ha="center", va="center", fontsize=13)
            ax.axis("off")
            fig.canvas.draw_idle()
            return

        obj = objetos[estado["idx"]]
        img_path = obj["archivos"][0]
        try:
            ax.imshow(imread(str(img_path)))
        except Exception as e:
            ax.text(0.5, 0.5, f"No se pudo abrir:\n{img_path.name}\n{e}", ha="center")
        ax.axis("off")

        clase_actual = etiquetas.get(obj["llave"], "—")
        ax.set_title(
            f"Imagen {estado['idx']+1}/{total}   |   tipo: {obj['fruta']}   |   "
            f"etiquetadas: {len(etiquetas)}\n"
            f"{img_path.name}   [clase actual: {clase_actual}]",
            fontsize=10)

        if obj["fruta"] == "Moras":
            teclas = "1=mora_sana   2=mora_antracnosis   3=mora_moho_gris"
        elif obj["fruta"] == "Pinas":
            teclas = "4=pina_sana   5=pina_pudricion_negra   6=pina_fusariosis"
        else:
            teclas = "7=banano_racimo_sano   8=banano_racimo_enfe_1   9=banano_racimo_enfe_2"
        fig.text(0.5, 0.02,
                 f"{teclas}\n s=saltar   u=deshacer   q=guardar y salir",
                 ha="center", fontsize=9, family="monospace")
        fig.canvas.draw_idle()

    def avanzar():
        estado["idx"] += 1
        # Saltar las que ya estén etiquetadas (por si se reanuda en medio)
        while estado["idx"] < total and objetos[estado["idx"]]["llave"] in etiquetas:
            estado["idx"] += 1

    def on_key(event):
        k = (event.key or "").lower()
        if estado["idx"] >= total:
            if k == "q":
                plt.close(fig)
            return
        obj = objetos[estado["idx"]]
        validas = CLASES_POR_FRUTA[obj["fruta"]]

        if k in validas:                        # asignar clase y avanzar
            etiquetas[obj["llave"]] = validas[k]
            guardar_progreso(etiquetas, objetos)
            avanzar()
            dibujar()
        elif k == "s":                          # saltar sin etiquetar
            avanzar()
            dibujar()
        elif k == "u":                          # deshacer el último etiquetado
            estado["idx"] = max(0, estado["idx"] - 1)
            etiquetas.pop(objetos[estado["idx"]]["llave"], None)
            guardar_progreso(etiquetas, objetos)
            dibujar()
        elif k == "q":                          # guardar y salir
            guardar_progreso(etiquetas, objetos)
            plt.close(fig)
        # cualquier otra tecla (incl. una clase no válida para esta fruta) se ignora

    fig.canvas.mpl_connect("key_press_event", on_key)
    dibujar()
    plt.show()  # bloquea hasta cerrar la ventana

    guardar_progreso(etiquetas, objetos)
    print(f"\nProgreso guardado en: {ARCHIVO_CSV}")
    print(f"Imágenes etiquetadas: {len(etiquetas)}/{total}")
    _resumen_por_clase(etiquetas, objetos)


# ----------------------------------------------------------------------------------------
# MODO EXPORTAR (construir data/etiquetadas/<clase>/ para ImageFolder)
# ----------------------------------------------------------------------------------------
def exportar() -> None:
    etiquetas = cargar_progreso()
    if not etiquetas:
        sys.exit("No hay etiquetas todavía. Ejecuta primero:  python etiquetar.py")

    objetos = {o["llave"]: o for o in listar_imagenes()}

    # Crear carpetas de las 9 clases
    for clase in TODAS_LAS_CLASES:
        (DIR_ETIQUETADAS / clase).mkdir(parents=True, exist_ok=True)

    copiadas = 0
    for llave, clase in etiquetas.items():
        obj = objetos.get(llave)
        if not obj:
            continue
        destino = DIR_ETIQUETADAS / clase
        for img in obj["archivos"]:
            dst = destino / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
                copiadas += 1

    print(f"Exportación lista en: {DIR_ETIQUETADAS}")
    print(f"Fotos copiadas en esta corrida: {copiadas}")
    _resumen_por_clase(etiquetas, list(objetos.values()))


def _resumen_por_clase(etiquetas: dict, objetos: list) -> None:
    """Imprime cuántas imágenes hay por clase."""
    conteo = defaultdict(int)
    for _, clase in etiquetas.items():
        conteo[clase] += 1
    print("\nResumen por clase (imágenes):")
    for clase in TODAS_LAS_CLASES:
        print(f"   {clase:28s}  {conteo.get(clase, 0):5d} imágenes")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--exportar":
        exportar()
    else:
        etiquetar()
