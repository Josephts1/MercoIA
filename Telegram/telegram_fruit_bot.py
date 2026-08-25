"""Bot de Telegram: segmentación con SAM 3 + diagnóstico con EfficientNet-B0.

Recibe una foto por Telegram y responde con el DIAGNÓSTICO de la red entrenada.
La segmentación ocurre por dentro: los recortes se usan para alimentar al
clasificador pero NO se envían al chat (salvo que se active /recortes).

Pipeline por imagen:
    foto -> SAM 3 (segmenta los frutos) -> EfficientNet-B0 (clasifica cada
    recorte) -> mensaje de texto con la clase y su probabilidad.

Uso:
    export TELEGRAM_BOT_TOKEN="123456:ABC..."
    python telegram_fruit_bot.py

Comandos en el chat:
    /banana | /mora | /pina   -> cambia el tipo de fruta activo
    /estado                   -> muestra la configuración actual
    /recortes                 -> activa/desactiva el envío de los recortes
                                 (depuración de la segmentación; por defecto NO)
    (enviar una foto)         -> procesa y devuelve el diagnóstico

También puedes poner el nombre de la fruta como pie de foto (caption) para
procesar esa imagen con ese tipo sin cambiar el modo global.
"""

import sys


def paso(msg):
    """Print inmediato y sin buffer: sobrevive a cualquier config de logging."""
    print(f">>> {msg}", flush=True)


paso("Arrancando telegram_fruit_bot.py")

import asyncio
import io
import logging
import os
import re
import threading
from collections import Counter
import time
import traceback
from pathlib import Path

paso("Importando cv2...")
import cv2

paso("Importando python-telegram-bot...")
from telegram import InputMediaDocument, ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

paso("Importando unified_fruit_detection (esto carga torch/ultralytics)...")
import unified_fruit_detection as ufd

paso("Importando clasificador (EfficientNet-B0 entrenado)...")
import clasificador

paso("Imports completos.")

# force=True: torch/ultralytics ya configuraron el logger raíz al importarse,
# y sin esto basicConfig se ignora silenciosamente y no vemos ningún log.
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("fruit-bot")
logging.getLogger("httpx").setLevel(logging.WARNING)

# --------------------------------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------------------------------

# .strip(): al copiar y pegar el token suele colarse un \n o espacios. Ese
# carácter acaba dentro de la URL de la API y httpx la rechaza con un
# "InvalidURL: non-printable ASCII character" que no dice nada del token.
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

# Solo estos IDs de usuario de Telegram pueden usar el bot.
# Déjalo vacío para permitir a cualquiera (no recomendado).
# Para saber tu ID: escríbele al bot y mira el log, lo imprime en cada mensaje.
ALLOWED_USER_IDS: set[int] = set()

# Fruta por defecto al arrancar
DEFAULT_FRUIT = "banana"

# Máximo de recortes a devolver por imagen cuando /recortes está activo
MAX_CROPS = 20

# El mensaje al usuario no muestra probabilidades, solo la clase más probable
# (top-1) de cada recorte.
TOPK = 1

# --------------------------------------------------------------------------
# MODELO: se carga UNA sola vez al arrancar (pesa 3.4 GB en VRAM)
# --------------------------------------------------------------------------

PREDICTOR = None
GPU_LOCK = threading.Lock()  # serializa el acceso a la GPU (12 GB no dan para 2 a la vez)


def cargar_modelo():
    global PREDICTOR
    import numpy as np
    import torch

    # --- Diagnóstico de GPU -------------------------------------------------
    paso(f"torch {torch.__version__} | CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        paso(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        paso("*** ATENCIÓN: corriendo en CPU. Será MUY lento (~2 min/imagen). ***")
        paso("*** Revisa el driver NVIDIA con: nvidia-smi                    ***")

    pesos = Path(ufd.SAM3_WEIGHTS)
    paso(f"Ruta de pesos resuelta: {pesos}")
    if not pesos.is_file():
        raise FileNotFoundError(
            f"No existe el archivo de pesos en {pesos}\n"
            f"    Mueve sam3.pt ahí, o edita SAM3_WEIGHTS en unified_fruit_detection.py"
        )
    paso(f"Pesos encontrados ({pesos.stat().st_size / 1e9:.1f} GB). Construyendo...")
    PREDICTOR = ufd.build_predictor(DEFAULT_FRUIT)

    # SAM 3 carga los pesos de forma perezosa (en la primera inferencia).
    # Hacemos un warmup con una imagen negra para que el primer mensaje real
    # no tenga que esperar la carga completa del modelo.
    paso("Warmup: cargando pesos en memoria (puede tardar ~1 min)...")
    t0 = time.time()
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        ufd.segment_array(PREDICTOR, dummy, DEFAULT_FRUIT)
    except Exception as e:
        paso(f"Warmup falló (no es fatal, pero revísalo): {e}")
    paso(f"SAM 3 listo en {time.time() - t0:.1f} s")

    # --- Clasificador entrenado (EfficientNet-B0, ~16 MB) -------------------
    paso("Cargando el clasificador de enfermedades...")
    clases, dev = clasificador.cargar_modelo()
    paso(f"Clasificador listo en {dev} — {len(clases)} clases: {', '.join(clases)}")


def procesar(image_bytes: bytes, fruit_type: str):
    """Segmenta y clasifica una imagen recibida por Telegram.

    Devuelve (predicciones, n_descartadas, recortes):
      - predicciones: lista paralela a los recortes aceptados; cada elemento es
        [(clase, prob), ...] ordenado de mayor a menor.
      - n_descartadas: cuántas detecciones se filtraron por oclusión.
      - recortes: los recortes BGR aceptados (solo se envían si /recortes).
    """
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No pude decodificar la imagen")

    # Un solo lock para las dos redes: comparten la misma GPU.
    with GPU_LOCK:
        aceptadas, descartadas = ufd.segment_array(PREDICTOR, img, fruit_type)
        predicciones = clasificador.clasificar_lote(
            aceptadas,
            topk=TOPK,
            # Restringe a las clases del fruto pedido: si se segmentaron
            # bananos, no tiene sentido diagnosticar una enfermedad de mora.
            clases_permitidas=clasificador.clases_de_fruta(fruit_type),
        )

    return predicciones, len(descartadas), aceptadas


# --------------------------------------------------------------------------
# HANDLERS DE TELEGRAM
# --------------------------------------------------------------------------


def autorizado(update: Update) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return update.effective_user and update.effective_user.id in ALLOWED_USER_IDS


# Etiqueta bonita -> clave interna de FRUIT_CONFIG
BOTONES = {"🍌 Banana": "banana", "🫐 Mora": "mora", "🍍 Piña": "pina"}

# Cómo llamar a cada detección en el chat. "g" es la vocal de género, para que
# los participios concuerden (racimo detectado / mora detectada).
UNIDAD = {
    "banana": {"sing": "racimo", "plur": "racimos", "g": "o"},
    "mora": {"sing": "mora", "plur": "moras", "g": "a"},
    "pina": {"sing": "piña", "plur": "piñas", "g": "a"},
}
_UNIDAD_GENERICA = {"sing": "fruto", "plur": "frutos", "g": "o"}
EMOJI = {"banana": "🍌", "mora": "🫐", "pina": "🍍"}


def _es_sano(clase):
    # "sano" (banano_racimo_sano) o "sana" (mora_sana, pina_sana) según género.
    return clase.lower().split("_")[-1] in ("sano", "sana")


def _frase_singular(clase, u):
    """'banano_racimo_sano' -> 'El racimo está *sano*'; 'mora_antracnosis' -> 'La mora tiene *antracnosis*'."""
    articulo = "El" if u["g"] == "o" else "La"
    if _es_sano(clase):
        adjetivo = "sano" if u["g"] == "o" else "sana"
        return f"{articulo} {u['sing']} está *{adjetivo}*"
    etiqueta = clasificador.nombre_bonito(clase).split(" (")[0]
    return f"{articulo} {u['sing']} tiene *{etiqueta}*"


def _frase_plural(clase, u):
    """'mora_sana' -> 'moras sanas'; 'mora_antracnosis' -> 'moras con antracnosis'."""
    if _es_sano(clase):
        return f"{u['plur']} sanas" if u["g"] != "o" else f"{u['plur']} sanos"
    etiqueta = clasificador.nombre_bonito(clase).split(" (")[0]
    return f"{u['plur']} con {etiqueta}"


def _porcentajes(predicciones):
    """[(clase, prob), ...] por recorte -> [(clase, pct_entero), ...] de mayor a menor.

    El % se calcula sobre cuántos recortes cayeron en cada clase (top-1), no
    sobre la probabilidad del modelo. El último se ajusta para que la suma dé
    exactamente 100 pese al redondeo.
    """
    conteo = Counter(p[0][0] for p in predicciones if p)
    total = sum(conteo.values())
    items = conteo.most_common()
    pcts = [round(100 * c / total) for _, c in items]
    if pcts:
        pcts[-1] = 100 - sum(pcts[:-1])
    return [(clase, pct) for (clase, _c), pct in zip(items, pcts)]


def formatear_diagnostico(fruta, predicciones, n_descartadas, dt):
    """Arma el mensaje de texto con el diagnóstico de la red.

    Sin probabilidades: banano da un único racimo (segmentación single_region)
    y el mensaje es una frase con su clase. Mora y piña pueden traer varios
    frutos, así que se agrega el % de cada clase sobre el total detectado.
    """
    u = UNIDAD.get(fruta, _UNIDAD_GENERICA)
    g = u["g"]
    n = len(predicciones)
    etiqueta = u["sing"] if n == 1 else u["plur"]
    detectado = f"detectad{g}" + ("" if n == 1 else "s")

    cabecera = f"{EMOJI.get(fruta, '')} *{n} {etiqueta}* {detectado}"
    if n_descartadas:
        descartado = f"descartad{g}" + ("" if n_descartadas == 1 else "s")
        cabecera += f" · {n_descartadas} {descartado}"
    cabecera += f" · {dt:.1f}s"

    if n == 1:
        clase = predicciones[0][0][0]
        return f"{cabecera}\n\n{_frase_singular(clase, u)}."

    partes = [f"un {pct}% de {_frase_plural(clase, u)}"
              for clase, pct in _porcentajes(predicciones)]
    texto = partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + " y " + partes[-1]

    return f"{cabecera}\n\nSe detecta {texto}."

TECLADO = ReplyKeyboardMarkup(
    [list(BOTONES.keys())], resize_keyboard=True, is_persistent=True
)


def clave_fruta(texto: str):
    """Traduce lo que escribió el usuario a una clave de FRUIT_CONFIG.

    Acepta variaciones: mora, Mora, MORAS, pina, Piña, BANANA, /banana, etc.
    """
    t = (texto or "").strip().lower()
    if t in BOTONES:  # improbable (case), pero por si acaso
        return BOTONES[t]
    for etiqueta, clave in BOTONES.items():
        if t == etiqueta.lower():
            return clave
    # Acepta también texto suelto: "pina", "piña", "banana", "mora", "moras", etc.
    # Normaliza: minúsculas, quita / al inicio, reemplaza ñ->n, quita plurales
    t = t.lstrip("/").replace("ñ", "n").rstrip("s")
    return t if t in ufd.FRUIT_CONFIG else None


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    log.info("Mensaje de user_id=%s", update.effective_user.id)
    if not autorizado(update):
        return
    ctx.chat_data.setdefault("fruta", DEFAULT_FRUIT)
    await update.message.reply_text(
        "Bot de diagnóstico de enfermedades listo.\n\n"
        f"Fruta activa: *{ctx.chat_data['fruta']}*\n\n"
        "Envíame una foto y te respondo con la enfermedad detectada. "
        "La segmentación se hace por dentro; en el chat solo verás el "
        "diagnóstico.\n\n"
        "Usa los botones de abajo para cambiar de fruta, o escribe la fruta "
        "como pie de foto para procesar solo esa imagen con ese tipo.\n\n"
        "Tip: envía la imagen como *archivo* (no como foto) para conservar "
        "la resolución original.\n"
        "Depuración: /recortes muestra también las imágenes segmentadas.",
        parse_mode="Markdown",
        reply_markup=TECLADO,
    )


async def cmd_recortes(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Activa/desactiva el envío de los recortes segmentados (depuración)."""
    if not autorizado(update):
        return
    activo = not ctx.chat_data.get("enviar_recortes", False)
    ctx.chat_data["enviar_recortes"] = activo
    await update.message.reply_text(
        f"Envío de recortes: *{'activado' if activo else 'desactivado'}*.\n"
        + ("Verás las imágenes segmentadas junto al diagnóstico."
           if activo else "Solo verás el diagnóstico de la red."),
        parse_mode="Markdown",
    )


async def cmd_fruta(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Cambia la fruta activa, sea por comando (/pina) o por botón (🍍 Piña)."""
    if not autorizado(update):
        return
    fruta = clave_fruta(update.message.text)
    if fruta is None:
        await update.message.reply_text(
            "No reconozco esa fruta. Usa los botones de abajo.",
            reply_markup=TECLADO,
        )
        return
    ctx.chat_data["fruta"] = fruta
    await update.message.reply_text(
        f"Modo activo: *{fruta}*. Envíame una foto.",
        parse_mode="Markdown",
        reply_markup=TECLADO,
    )


async def cmd_estado(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update):
        return
    fruta = ctx.chat_data.get("fruta", DEFAULT_FRUIT)
    cfg = ufd.FRUIT_CONFIG[fruta]
    permitidas = clasificador.clases_de_fruta(fruta) or []
    await update.message.reply_text(
        f"Fruta: {fruta}\n"
        f"Prompts: {', '.join(cfg['prompts'])}\n"
        f"conf: {cfg['conf']} | iou: {cfg['iou']} | max_white: {cfg['max_white_pct']}\n"
        f"Agrupa en racimo: {'sí' if cfg.get('merge_instances') else 'no'}\n"
        f"Clases del diagnóstico: {', '.join(permitidas) or '(todas)'}\n"
        f"Envío de recortes: {'sí' if ctx.chat_data.get('enviar_recortes') else 'no'}"
    )


async def on_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    log.info("Imagen de user_id=%s", update.effective_user.id)
    if not autorizado(update):
        await update.message.reply_text("No autorizado.")
        return

    msg = update.message
    # El caption tiene prioridad sobre el modo activo del chat.
    fruta = clave_fruta(msg.caption) or ctx.chat_data.get("fruta", DEFAULT_FRUIT)

    # Foto comprimida (photo) o archivo original (document)
    if msg.photo:
        tg_file = await msg.photo[-1].get_file()  # mayor resolución disponible
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        tg_file = await msg.document.get_file()
    else:
        await msg.reply_text("Envíame una imagen.")
        return

    aviso = await msg.reply_text(f"Analizando como *{fruta}*...", parse_mode="Markdown")
    await ctx.bot.send_chat_action(msg.chat_id, ChatAction.TYPING)

    try:
        raw = bytes(await tg_file.download_as_bytearray())
        t0 = time.time()
        predicciones, n_descartadas, recortes = await asyncio.to_thread(
            procesar, raw, fruta
        )
        dt = time.time() - t0
    except Exception as e:
        log.exception("Error procesando imagen")
        await aviso.edit_text(f"Error: {e}")
        return

    if not predicciones:
        u = UNIDAD.get(fruta, _UNIDAD_GENERICA)
        articulo = "ningún" if u["g"] == "o" else "ninguna"
        texto = f"No detecté {articulo} {u['sing']} en la imagen"
        if n_descartadas:
            texto += f" ({n_descartadas} descartad{u['g']}s por oclusión)"
        await aviso.edit_text(f"{texto} — {dt:.1f}s")
        return

    await aviso.edit_text(
        formatear_diagnostico(fruta, predicciones, n_descartadas, dt),
        parse_mode="Markdown",
    )

    # Los recortes solo se envían en modo depuración (/recortes).
    if not ctx.chat_data.get("enviar_recortes"):
        return

    # Un racimo completo puede pesar varios MB en PNG; codificarlo aquí
    # bloquearía el bucle de eventos y el bot dejaría de responder.
    def a_png(crops):
        out = []
        for c in crops:
            ok, buf = cv2.imencode(".png", c)
            if ok:
                out.append(buf.tobytes())
        return out

    pngs = await asyncio.to_thread(a_png, recortes[:MAX_CROPS])

    # Telegram acepta media groups de máximo 10 elementos
    for i in range(0, len(pngs), 10):
        lote = pngs[i : i + 10]
        media = [
            InputMediaDocument(
                media=io.BytesIO(png), filename=f"{fruta}_{i + j:02d}.png"
            )
            for j, png in enumerate(lote)
        ]
        await ctx.bot.send_media_group(msg.chat_id, media)

    if len(recortes) > MAX_CROPS:
        await msg.reply_text(
            f"(Se enviaron {MAX_CROPS} de {len(recortes)} recortes)"
        )


async def on_error(update, ctx):
    log.exception("Excepción no controlada", exc_info=ctx.error)


# --------------------------------------------------------------------------


def main():
    paso(f"Python: {sys.executable}")
    if not TOKEN:
        raise SystemExit(
            "Falta la variable TELEGRAM_BOT_TOKEN.\n"
            "    Ejecuta:  export TELEGRAM_BOT_TOKEN='tu_token'"
        )
    # Un token válido es <id_numérico>:<~35 caracteres>. Comprobarlo aquí
    # evita esperar a que cargue el modelo (3.4 GB) para morir en la primera
    # llamada a la API con un error que no menciona el token.
    if not re.fullmatch(r"\d+:[A-Za-z0-9_-]{30,}", TOKEN):
        raise SystemExit(
            f"El token no tiene el formato esperado (largo={len(TOKEN)}).\n"
            f"    Debe ser  123456789:AAE...  sin espacios ni saltos de línea.\n"
            f"    Revísalo con:  echo \"[$TELEGRAM_BOT_TOKEN]\"\n"
            f"    Si el ] sale en otro renglón, hay un salto de línea dentro."
        )
    paso(f"Token detectado (termina en ...{TOKEN[-6:]})")

    cargar_modelo()

    paso("Construyendo la aplicación de Telegram...")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("estado", cmd_estado))
    app.add_handler(CommandHandler("recortes", cmd_recortes))
    for fruta in ufd.FRUIT_CONFIG:
        app.add_handler(CommandHandler(fruta, cmd_fruta))
    # Botones del teclado permanente (llegan como mensajes de texto normales)
    app.add_handler(
        MessageHandler(filters.Text(list(BOTONES.keys())), cmd_fruta)
    )
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, on_image))
    app.add_error_handler(on_error)

    paso("Bot escuchando (polling). Ctrl+C para detener.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        paso("Detenido por el usuario.")
    except BaseException:
        paso("EL BOT MURIÓ. Traceback completo:")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
# python /home/joseph/Documents/MERCOIA/Algorithm/codes/Dataset_codes/Deteccion_y_filtrado/telegram_fruit_bot.py
