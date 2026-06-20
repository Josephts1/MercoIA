import cv2; from pathlib import Path; import torch; from matplotlib import pyplot as plt; import numpy as np; import requests
from ultralytics.models.sam import SAM3SemanticPredictor; from ultralytics.utils.plotting import save_one_box

# Centroides
C1 = np.array([-6.1667,  39.7121]) # Etapa verde
C2 = np.array([38.3822,  25.9969]) # Etapa roja
C3 = np.array([ 9.8719, -18.1026]) # Etapa morada
T1 = 18 # días de verde a roja
T2 = 30 # días de roja a morada

# Ventana de tolerancia en días para harvest_window
HARVEST_W = 3

_LEN_ALPHA1 = float(np.linalg.norm(C2 - C1)) # |α1| = |C2 - C1|
_LEN_ALPHA2 = float(np.linalg.norm(C3 - C2)) # |α2| = |C3 - C2|

ZONE_LABEL = {
    1: "Verde→Roja  (inmadura)",
    2: "Roja→Morada (pre-madura)",
    3: "Madura ✓",
}

def crops(image_dir, labels_dir):

    labels_dir = Path(labels_dir)
    save_path = labels_dir / "crops"
    #SAM
    overrides = dict(conf=0.1, task="segment", mode="predict",
        model="/home/joseph/Documents/Confe_MercoIA/checkpoints/sam3.pt",
        half=True, save=False, save_txt=True,
        project=str(labels_dir), imgsz=644,
    )
    img = cv2.imread(image_dir)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_dir}")

    #SAM
    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(img)
    mask, boxes = predictor.inference_features(
        predictor.features,
        src_shape=img.shape[:2],
        text=["Blackberry"],
    )

    if mask is None or boxes is None or len(boxes) == 0:
        raise RuntimeError(
            f"SAM no detectó ninguna mora en {image_dir} "
            f"(conf={overrides['conf']}). Prueba bajar el umbral o cambiar el prompt."
        )

    #print(boxes)
    #Coordenada y1 arriba hacia abajo
    index = torch.argsort(boxes[:, 1], descending=False)
    boxes = boxes[index]
    mask  = mask[index]

    # Overlay compuesto: se acumula fuera del loop
    seg_overlay = img.copy()

    for i, box in enumerate(boxes):
        xyxy = box[:4]
        m = mask[i].cpu().numpy().astype(bool)

        # Crop con fondo blanco
        isolated_img = img.copy()
        isolated_img[~m] = [255, 255, 255]
        save_one_box(xyxy, isolated_img,
                     file=save_path / f"blackberry12_{i}.png", save=True)

    return save_path

def estimate_delta_t(a_mean: float, b_mean: float) -> tuple:
    """
    Estima días hasta cosecha proyectando q = (a*, b*) ortogonalmente
    sobre los segmentos a1 (C1→C2) y a2 (C2→C3) en el espacio CIELAB.

    --- Variables ---
      β  = |C_fin - proj| distancia CIELAB restante al centroide final:
               En a1: β = |C2 - proj_q_en_a1|
               En a2: β = |C3 - proj_q_en_a2|
      δt = T / |a| ## (días por unidad CIELAB del segmento) ##el número:
               En a1: δt = T1 / |a1|
               En a2: δt = T2 / |a2|
      τ  = días reales hasta la cosecha óptima (valor final entregado):
               En a1: τ = δt · β + T2   (recorre lo que falta de a1, más todo a2)
               En a2: τ = δt · β         (solo recorre lo que falta de a2)

    ── Verificación de valores límite ──────────────────────────────────
      q = C1 (muy verde):  β = |a1|, τ = (T1/|a1|)·|a1| + T2 = T1+T2 = 48 d ✓
      q = C2 (roja):       β = 0,    τ = 0 + T2             = T2     = 30 d ✓  (en a2)
      q = C3 (madura):     β = 0,    τ = 0                  = 0      =  0 d ✓

    ── Algoritmo paso a paso ───────────────────────────────────────────
    1. Proyectar q sobre a1:
         t1    = clip( dot(q-C1, a1) / |a1|² , 0, 1 )   ← parámetro escalar
         proj1 = C1 + t1 · a1                            ← punto proyectado
         dist1 = |q - proj1|                             ← distancia perpendicular
    2. Proyectar q sobre a2 (igual, desde C2):
         t2, proj2, dist2  (análogo al paso 1)
    3. Asignar al tramo con menor distancia perpendicular.
    4. Calcular β y δt para el tramo asignado.
    5. Calcular τ con la fórmula del tramo.

    ── Retorna ─────────────────────────────────────────────────────────
    (tau, delta_t, beta, tramo, t_proj, perp_dist, zone_label)
      tau       : días estimados hasta cosecha (valor principal del pipeline)
      delta_t   : δt = T/|a|, tasa [días/unidad-CIELAB] — valor intermedio
      beta      : β = distancia CIELAB restante al centroide final [unidades CIELAB]
      tramo     : 1 (a1, verde→rojo) ó 2 (a2, rojo→morado)
      t_proj    : parámetro de proyección ∈ [0,1] (progreso dentro del tramo)
      perp_dist : distancia perpendicular al tramo asignado (calidad del ajuste)
      zone_label: etiqueta de zona legible para reportes
    """
    q  = np.array([a_mean, b_mean])
    d1 = C2 - C1   # vector del tramo α1
    d2 = C3 - C2   # vector del tramo α2

    # ── Paso 1: proyección sobre α1 ─────────────────────────────────
    t1    = float(np.clip(np.dot(q - C1, d1) / np.dot(d1, d1), 0.0, 1.0))
    proj1 = C1 + t1 * d1
    dist1 = float(np.linalg.norm(q - proj1))   # distancia perpendicular a α1

    # ── Paso 2: proyección sobre α2 ─────────────────────────────────
    t2    = float(np.clip(np.dot(q - C2, d2) / np.dot(d2, d2), 0.0, 1.0))
    proj2 = C2 + t2 * d2
    dist2 = float(np.linalg.norm(q - proj2))   # distancia perpendicular a α2

    # ── Paso 3, 4 y 5: asignación, β, δt y τ ────────────────────────
    if dist1 <= dist2:
        # ── Tramo α1 (verde → roja) ──────────────────────────────────
        beta    = float(np.linalg.norm(proj1 - C2))  # β = |C2 - proj1| [CIELAB]
        delta_t = T1 / _LEN_ALPHA1                   # δt = T1/|α1| [días/CIELAB]
        tau     = delta_t * beta + T2                # τ = δt·β + T2  [días]
        return tau, delta_t, beta, 1, t1, dist1, ZONE_LABEL[1]
    else:
        # ── Tramo α2 (roja → morada) ─────────────────────────────────
        beta    = float(np.linalg.norm(proj2 - C3))  # β = |C3 - proj2| [CIELAB]
        delta_t = T2 / _LEN_ALPHA2                   # δt = T2/|α2| [días/CIELAB]
        tau     = max(0.0, delta_t * beta)            # τ = δt·β       [días]
        # Sub-etiqueta: t2 ≥ 0.85 → β ≤ 15% del tramo → mora casi madura
        label   = ZONE_LABEL[3] if t2 >= 0.85 else ZONE_LABEL[2]
        return tau, delta_t, beta, 2, t2, dist2, label

def harvest_window(delta_t_list: list, W: int = HARVEST_W) -> dict:
    """
    Para cada día candidato d, cuenta cuántas moras tienen su Δt dentro
    de la ventana [d - W/2, d + W/2].

    Retorna el día d* que maximiza el número de moras cosechables
    simultáneamente, junto con estadísticas de aprovechamiento.

    Fórmula:
        N(d) = Σᵢ 𝟙[ Δtᵢ ∈ [d - W/2, d + W/2] ]
        d* = argmax N(d)
    """
    if not delta_t_list:
        return {}

    dt      = np.array(delta_t_list)
    half_W  = W / 2.0
    max_day = int(np.ceil(dt.max())) + W + 1
    days    = np.arange(0, max_day + 1)

    N = np.array([
        int(np.sum((dt >= d - half_W) & (dt <= d + half_W)))
        for d in days
    ])

    d_star  = int(days[np.argmax(N)])
    n_star  = int(N[d_star])
    pct     = n_star / len(dt) * 100

    return {
        "d_star"             : d_star,
        "n_cosechables"      : n_star,
        "pct_aprovechamiento": pct,
        "distribution"       : list(zip(days.tolist(), N.tolist())),
    }

def color_analysis(crops_dir: str) -> tuple:
    """
    Para cada crop calcula IC = mean(a* / -b*) usando SOLO píxeles del fruto,
    clasifica la zona de madurez y estima Δt.

    Retorna (IC_array, berry_data_list) donde berry_data_list es una lista
    de dicts con toda la información por mora.

    CORRECCIONES respecto al código original:
      1. Máscara de fondo: el fondo es BLANCO [255,255,255] (asignado en crops()),
         NO negro. Se corrige de (img <= 6) a (img >= 249).
      2. División por cero: se excluyen píxeles con b* == 0 antes de calcular IC.
      3. Histograma: ahora grafica solo píxeles del fruto (a_px, b_px),
         no todo el array incluyendo el fondo.
      4. Directorio analysis/ se crea con mkdir antes de savefig().
      5. La función retorna (IC_arr, berry_data) para integrar con el pipeline.
      6. Se calcula a_mean y b_mean por mora para la clasificación de zona.
    """
    crops_dir   = Path(crops_dir)
    analysis_dir = crops_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)   # FIX 4: crear antes de savefig

    # Excluir archivos de overlay y máscaras; ordenar numéricamente.
    # NOTA: crops() guarda los archivos como "blackberry12_{i}.png",
    # por eso el patrón es "blackberry12_*.png" (no "blackberry_*.jpg").
    crops_paths = sorted(
        [p for p in crops_dir.glob("blackberry12_*.jpg")
         if "_mask" not in p.name and "_seg" not in p.name],
        key=lambda x: int(x.stem.split("_")[1]),
    )
    n_crops = len(crops_paths)
    if n_crops == 0:
        raise ValueError(f"No se encontraron crops en: {crops_dir}")

    berry_data = []
    # squeeze=False garantiza que axes sea siempre 2D (n_crops × 2),
    # evitando el IndexError "axes[i][0]" cuando n_crops == 1.
    fig, axes = plt.subplots(nrows=n_crops, ncols=2,
                             figsize=(10, 4 * n_crops), squeeze=False)

    for i, img_path in enumerate(crops_paths):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # FIX 1: fondo es BLANCO (crops() usa isolated_img[~m] = [255,255,255])
        mask_bac   = np.all(img_bgr >= 249, axis=2)   # antes: img <= 6 (incorrecto)
        fruit_mask = ~mask_bac

        if fruit_mask.sum() == 0:
            print(f"[WARN] Mora {i+1}: máscara vacía, se omite.")
            berry_data.append(None)
            continue
        
        #plt.imshow(fruit_mask) VER MASK
        #plt.show()

        # CIELAB, centrar en 0
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        a_full  = img_lab[:, :, 1].astype(float) - 128.0
        b_full  = img_lab[:, :, 2].astype(float) - 128.0

        a_px = a_full[fruit_mask]
        b_px = b_full[fruit_mask]

        valid = b_px != 0
        if valid.sum() < 5:
            IC_val = np.nan
        else:
            IC_val = float(np.mean(a_px[valid] / (-b_px[valid])))

        a_mean = float(np.mean(a_px))
        b_mean = float(np.mean(b_px))
        std_a  = float(np.std(a_px))
        std_b  = float(np.std(b_px))

        # ── Estimación de días hasta cosecha (proyección geométrica CIELAB) ──
        # Retorna: (tau, delta_t, beta, tramo, t_proj, perp_dist, zone_label)
        # • tau       : τ = días reales hasta cosecha (valor principal del pipeline)
        # • delta_t   : δt = T/|α|, tasa [días/unidad-CIELAB], valor intermedio
        # • beta      : β = distancia CIELAB restante al centroide final
        # • tramo     : 1 = verde→roja (α1), 2 = roja→morada (α2)
        # • t_proj    : parámetro de proyección ∈ [0,1] (progreso dentro del tramo)
        # • perp_dist : distancia perpendicular al tramo (calidad del ajuste)
        # • zone_label: etiqueta legible para reportes y LLM
        tau, delta_t, beta, tramo, t_proj, perp_dist, zone_label = estimate_delta_t(a_mean, b_mean)

        berry_data.append({
            "index"    : i + 1,
            "IC"       : IC_val,
            "a_mean"   : a_mean,
            "b_mean"   : b_mean,
            "std_a"    : std_a,
            "std_b"    : std_b,
            "tramo"    : tramo,      # 1 ó 2 (tramo de la trayectoria CIELAB)
            "t_proj"   : t_proj,     # parámetro de proyección ∈ [0,1]
            "delta_t"  : delta_t,    # δt = T/|α| (tasa, valor intermedio)
            "beta"     : beta,       # β = distancia CIELAB restante al centroide final
            "tau"      : tau,        # τ = días hasta cosecha (valor principal)
            "perp_dist": perp_dist,  # distancia perpendicular al tramo
            "label"    : zone_label, # etiqueta de zona legible
        })

        print(
            f"Mora {i+1:>2} | IC={IC_val:+.3f} | "
            f"a*={a_mean:+.1f} | b*={b_mean:+.1f} | "
            f"tramo=α{tramo} t={t_proj:.2f} β={beta:.2f} | "
            f"{zone_label:<25} | τ={tau:.1f} días"
        )

        # Subplot izquierdo: imagen con anotación
        ax_img = axes[i][0]
        ax_img.imshow(img_rgb)
        ax_img.set_title(
            f"Mora {i+1}  ·  IC = {IC_val:+.3f}\n"
            f"{zone_label}  ·  τ = {tau:.1f} días",
            fontsize=9,
        )
        ax_img.axis("off")

        # Subplot derecho: histograma SOLO píxeles del fruto 
        ax_hist = axes[i][1]
        ax_hist.hist(a_px, bins=64, alpha=0.6, color="red",  label="a*")
        ax_hist.hist(b_px, bins=64, alpha=0.5, color="blue", label="b*")
        ax_hist.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        ax_hist.set_title(f"std_a = {std_a:.2f}  |  std_b = {std_b:.2f}", fontsize=9)
        ax_hist.set_xlabel("Valor CIELAB")
        ax_hist.legend(loc="upper right")

    plt.tight_layout()
    out_fig = analysis_dir / "resultados_color_analisis.png"
    plt.savefig(str(out_fig), dpi=150)
    plt.close(fig)

    valid_data = [d for d in berry_data if d is not None and not np.isnan(d["IC"])]
    IC_arr     = np.array([d["IC"]  for d in valid_data])
    return IC_arr, valid_data

def plot_harvest_window(hw: dict, berry_data: list, save_dir: Path) -> None:
    """
    Stacked bar chart showing, for each candidate harvest day d, the
    percentage of berries in each ripeness category:
      - Stage 1  (Unripe)    : tramo α1, not yet in harvest window
      - Stage 2  (Developing): tramo α2, not yet in harvest window  
      - Ripe                 : τ ∈ [d − w/2, d + w/2]
      - Overripe             : τ < d − w/2 (past optimal peak)
    
    The optimal harvest day d* is the column where the Ripe slice is tallest.
    """
    tau_arr = np.array([d["tau"] for d in berry_data])
    tramo_arr = np.array([d["tramo"] for d in berry_data])
    n = len(berry_data)
    half_W = HARVEST_W / 2.0
    
    days, counts = zip(*hw["distribution"])
    days = np.array(days)
    d_star = hw["d_star"]
    days = np.arange(min(days), max(days) + 1)
    days = days[days <= 50]
    
    # For each day d, compute the 4-category percentages
    pct_stage1 = []
    pct_stage2 = []
    pct_ripe   = []
    pct_over   = []
    
    for d in days:
        in_window  = (tau_arr >= d - half_W) & (tau_arr <= d + half_W)
        overripe   = tau_arr < d - half_W
        stage1_not_ready = (~in_window) & (~overripe) & (tramo_arr == 1)
        stage2_not_ready = (~in_window) & (~overripe) & (tramo_arr == 2)
        
        pct_stage1.append(stage1_not_ready.sum() / n * 100)
        pct_stage2.append(stage2_not_ready.sum() / n * 100)
        pct_ripe.append(in_window.sum() / n * 100)
        pct_over.append(overripe.sum() / n * 100)
    
    pct_stage1 = np.array(pct_stage1)
    pct_stage2 = np.array(pct_stage2)
    pct_ripe   = np.array(pct_ripe)
    pct_over   = np.array(pct_over)
    
    # Colors: green, red/orange, purple, brown
    colors = {
        'stage1': '#4CAF50',   # green - unripe
        'stage2': '#E53935',   # red - developing
        'ripe':   '#6A1B9A',   # purple - ripe
        'over':   '#795548',   # brown - overripe
    }
    
    fig, ax = plt.subplots(figsize=(max(10, len(days) * 0.6), 6))
    
    bars_stage1 = ax.bar(days, pct_stage1, color=colors['stage1'], label='Stage 1 – Unripe')
    bars_stage2 = ax.bar(days, pct_stage2, bottom=pct_stage1, color=colors['stage2'], label='Stage 2 – Developing')
    bars_ripe   = ax.bar(days, pct_ripe,   bottom=pct_stage1+pct_stage2, color=colors['ripe'], label='Stage 3 – Ripe (Harvest)')
    bars_over   = ax.bar(days, pct_over,   bottom=pct_stage1+pct_stage2+pct_ripe, color=colors['over'], label='Overripe')
    
    # Add percentage labels inside bars (only if segment > 5%)
    for i, d in enumerate(days):
        bottoms = [0, pct_stage1[i], pct_stage1[i]+pct_stage2[i], pct_stage1[i]+pct_stage2[i]+pct_ripe[i]]
        values = [pct_stage1[i], pct_stage2[i], pct_ripe[i], pct_over[i]]
        for bot, val in zip(bottoms, values):
            if val > 5:
                ax.text(d, bot + val/2, f'{val:.1f}%', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    
    # Mark d* with vertical line and annotation
    ax.axvline(x=d_star, color='gold', linestyle='--', linewidth=2, label=f'Optimal Day d* = {d_star}')
    
    ax.set_xlabel('Days from Today', fontsize=11)
    ax.set_ylabel('% of Total Berries', fontsize=11)
    ax.set_title('Optimal Harvest Window – Ripeness Stage Distribution per Day', fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(days)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()
    
    out = save_dir / "ventana_cosecha.png"
    plt.savefig(str(out), dpi=150)
    plt.close(fig)

def agro_recomen_llama(berry_data: list, hw: dict) -> str:
    """
    Construye un prompt con datos de madurez + ventana óptima de cosecha
    y obtiene una recomendación del LLM local (Ollama / llama3.1:8b).

    CORRECCIONES respecto al código original:
      - El prompt ahora incluye pct_diseased y la info de Δt por mora.
      - Se añade manejo de errores para fallas de conexión.
      - La función retorna el texto generado.
    """
    n = len(berry_data)

    # ── 1. Clasificación de madurez ──────────────────────────────────────
    # Basada en el modelo de proyección CIELAB (tramo + t_proj):
    #   Madura    → tramo α2, t_proj ≥ 0.85  (β ≤ 15% del tramo, lista ya)
    #   Pre-madura → tramo α2, t_proj < 0.85  (aún en tránsito rojo→morado)
    #   Inmadura  → tramo α1                  (etapa verde→roja, aún lejos)
    ripe     = sum(1 for d in berry_data if d["tramo"] == 2 and d["t_proj"] >= 0.85)
    pre_ripe = sum(1 for d in berry_data if d["tramo"] == 2 and d["t_proj"] <  0.85)
    unripe   = sum(1 for d in berry_data if d["tramo"] == 1)
    pct_ripe     = ripe     / n * 100
    pct_pre_ripe = pre_ripe / n * 100
    pct_unripe   = unripe   / n * 100

    d_star              = hw["d_star"]
    n_cosechables       = hw["n_cosechables"]
    pct_aprovechamiento = hw["pct_aprovechamiento"]

    # ── 2. Métricas adicionales calculadas en Python ─────────────────────
    # Todas estas cifras se pasan pre-calculadas al LLM para que las use
    # directamente en la recomendación sin hacer ningún cálculo propio.
    tau_arr  = np.array([d["tau"] for d in berry_data])
    half_W   = HARVEST_W / 2.0

    # Estadísticos descriptivos del tiempo de madurez (τ)
    tau_mean = float(np.mean(tau_arr))
    tau_std  = float(np.std(tau_arr))
    tau_min  = float(np.min(tau_arr))
    tau_max  = float(np.max(tau_arr))
    rango_dias = tau_max - tau_min          # dispersión total del lote

    # ¿Cuántas moras estarían listas si se cosechara HOY (día 0)?
    n_hoy   = int(np.sum(tau_arr <= half_W))
    pct_hoy = n_hoy / n * 100

    # Moras que quedarían sin cosechar si se cosecha hoy (τ > W/2 días)
    n_perdidas_hoy   = n - n_hoy
    pct_perdidas_hoy = n_perdidas_hoy / n * 100

    # Eficiencia un día antes y un día después de d_star (sensibilidad)
    n_antes  = int(np.sum((tau_arr >= (d_star-1) - half_W) & (tau_arr <= (d_star-1) + half_W)))
    n_despues= int(np.sum((tau_arr >= (d_star+1) - half_W) & (tau_arr <= (d_star+1) + half_W)))
    pct_antes   = n_antes   / n * 100
    pct_despues = n_despues / n * 100

    # Moras que en d_star aún no han madurado (τ > d_star + W/2)
    n_inmad_en_dstar   = int(np.sum(tau_arr > d_star + half_W))
    pct_inmad_en_dstar = n_inmad_en_dstar / n * 100

    # Moras que en d_star ya habrían sobremaduro (τ < d_star - W/2)
    n_sobre_en_dstar   = int(np.sum(tau_arr < d_star - half_W))
    pct_sobre_en_dstar = n_sobre_en_dstar / n * 100

    # Justificación matemática del d_star: es el día con mayor N(d)
    # N(d_star) = n_cosechables > N(d_star±1) → es el pico de la distribución
    ganancia_vs_hoy    = n_cosechables - n_hoy       # moras extra vs cosechar hoy

    # ── 3. Tabla individual de moras ─────────────────────────────────────
    mora_lines = "\n".join([
        f"  Mora {d['index']:>2}: {d['label']:<26} → lista en {d['tau']:>4.0f} día(s)"
        for d in berry_data
    ])

    url   = "http://localhost:11434/api/chat"
    model = "llama3.1:8b"

    # ── 4. Construcción del prompt ────────────────────────────────────────
    # Estrategia:
    #   a) d_star va primero, encuadrado, como dato inamovible.
    #   b) Se entregan las métricas matemáticas ya calculadas para que el LLM
    #      las cite directamente — no recalcula, solo redacta.
    #   c) La justificación del d_star está explícita: es el día en que N(d)
    #      es máximo (más moras simultáneamente listas), con comparativas
    #      vs. cosechar hoy, un día antes y un día después.
    #   d) El formato obliga a mencionar porcentajes, justificación y acción.

    system_prompt = f"""Eres un agrónomo especialista en mora castilla (Rubus glaucus).
Tu única tarea es redactar una recomendación de cosecha usando EXCLUSIVAMENTE
los datos pre-calculados que se te proporcionan a continuación.

DÍA ÓPTIMO DE COSECHA  : día {d_star} desde hoy
Moras listas ese día   : {n_cosechables} de {n} ({pct_aprovechamiento:.1f}% del lote)
Ventana de cosecha     : días {max(0, d_star - int(half_W))} – {d_star + int(half_W)} desde hoy (±{int(half_W)} días)

── Estado actual del lote (hoy) ──────────────────────
  Maduras ahora              : {ripe:>2}  ({pct_ripe:.1f}%)
  En maduración avanzada     : {pre_ripe:>2}  ({pct_pre_ripe:.1f}%)
  Inmaduras                  : {unripe:>2}  ({pct_unripe:.1f}%)
  Si se cosechara HOY        : solo {n_hoy} mora(s) ({pct_hoy:.1f}%) estarían en punto

── Por qué el día {d_star} es el óptimo (comparativa N(d)) ──
  Cosechar hoy (día 0)       : {n_hoy}  moras listas ({pct_hoy:.1f}%)  → {ganancia_vs_hoy:+d} vs día {d_star}
  Cosechar día {d_star:<2} ← ÓPTIMO    : {n_cosechables}  moras listas ({pct_aprovechamiento:.1f}%)  ← máximo
  En día {d_star}: moras aún inmaduras : {n_inmad_en_dstar} ({pct_inmad_en_dstar:.1f}%)  (no alcanzarán a madurar)
  En día {d_star}: moras sobremaduradas: {n_sobre_en_dstar} ({pct_sobre_en_dstar:.1f}%)  (habrán pasado su punto óptimo)

── Estado individual de cada mora ────────────────────
{mora_lines}

REGLAS (incumplirlas invalida la respuesta):
1. El día óptimo es el DÍA {d_star} — menciónalo exactamente así en el primer párrafo.
   JAMÁS uses "hoy", "mañana", "ahora mismo" ni ningún otro día.
2. Usa los porcentajes y cifras de arriba tal como están — no recalcules nada.
3. Explica brevemente por qué el día {d_star} es mejor que cosechar hoy
   (usando las diferencias de N(d) de la tabla comparativa).
4. Máximo 250 palabras. Sin títulos. Sin viñetas. Solo párrafos en español.
   Léxico claro para un agricultor, sin tecnicismos innecesarios.

Formato de la respuesta (tres párrafos):
  Párrafo 1 — Estado actual: menciona cuántas moras hay maduras ({pct_ripe:.1f}%),
    en maduración ({pct_pre_ripe:.1f}%) e inmaduras ({pct_unripe:.1f}%) hoy en el lote.
  Párrafo 2 — Recomendación justificada: indica que el día óptimo de cosecha
    es el día {d_star}; explica que ese día estarán listas {n_cosechables} moras
    ({pct_aprovechamiento:.1f}%), {ganancia_vs_hoy} más que si se cosechara hoy.
    Menciona que cosechar un día antes o después reduce el aprovechamiento.
  Párrafo 3 — Acción concreta y consecuencias: qué debe hacer el agricultor
    entre hoy y el día {d_star} (revisar, preparar cosecha), y qué ocurriría
    si no cosecha en la ventana días {max(0, d_star-int(half_W))}–{d_star+int(half_W)}
    (sobremaduración, pérdida de calidad y valor comercial).
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": (
             f"El sistema calculó que el día óptimo de cosecha es el día {d_star} "
             f"desde hoy, cuando {n_cosechables} de {n} moras ({pct_aprovechamiento:.1f}%) "
             f"estarán simultáneamente en su punto. Redacta la recomendación completa "
             f"para el agricultor siguiendo el formato de tres párrafos."
         )},
    ]
    payload = {
        "model"  : model,
        "messages": messages,
        "stream" : False,
        # temperature=0 + seed=42 + top_k=1 → salida determinista:
        # el mismo lote siempre produce la misma recomendación.
        "options": {"temperature": 0, "seed": 42, "top_k": 1, "num_predict": 500},
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        text = response.json()["message"]["content"]
        print("\n─── Recomendación LLM ───────────────────────")
        print(text)
        print("─────────────────────────────────────────────")
        return text
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] No se pudo conectar con el LLM local: {e}")
        return ""

if __name__ == "__main__":
    IMAGE  = "/home/joseph/Documents/Confe_MercoIA/images/moras_moradas/cap1_13may_2.png"
    OUTPUT = "/home/joseph/Documents/Confe_MercoIA/images/siii"

    # 1 Block (Crops)
    Run_segm = True
    if Run_segm:
        crops_dir = crops(IMAGE, OUTPUT)

    # 2 Block (Color Analysis)
    IC_arr, berry_data = color_analysis(str(crops_dir))
    # 3 Block (Harvest Window)
    # Se usa d["tau"] porque τ es el tiempo real de cosecha estimado por el modelo
    tau_list = [d["tau"] for d in berry_data]
    hw = harvest_window(tau_list, W=HARVEST_W)
    print(
        f"\nVentana óptima: día {hw['d_star']} desde hoy → "
        f"{hw['n_cosechables']} moras cosechables "
        f"({hw['pct_aprovechamiento']:.1f}% del lote)"
    )
    plot_harvest_window(hw, berry_data, Path(OUTPUT) / "crops" / "analysis")    
    # 4 Block (LLM)
    agro_recomen_llama(berry_data, hw)
