## 📖 ¿Qué es MercoIA?

**MercoIA** es el módulo de clasificación de enfermedades del proyecto homónimo, desarrollado en el marco del *Proyecto de Investigación — Convenio 268-2025*. Su propósito es asistir a agricultores de la región de **Santander (Colombia)** en la identificación temprana de enfermedades en cultivos de **piña, mora y banano** a partir de fotografías tomadas con su propio celular, sin necesidad de equipos especializados ni acceso a laboratorio.

El problema que motiva este trabajo es concreto: las enfermedades fúngicas en estas tres especies son una de las principales causas de pérdida poscosecha en la región, pero su diagnóstico oportuno depende de técnicos agrícolas que no siempre están disponibles a tiempo. MercoIA busca cerrar esa brecha poniendo una herramienta de diagnóstico directamente en el bolsillo del productor.


## ✨ Características principales

- 📱 **Pensado para celulares** — funciona con fotos tomadas en condiciones reales de campo (luz y ángulos variables, distintas distancias).
- 🔬 **9 clases de diagnóstico** cubriendo las enfermedades más frecuentes en tres cultivos regionales.
- 🤖 **EfficientNet-B0** preentrenado en ImageNet con fine-tuning especializado en frutas.
- 🏷️ **Etiquetado asistido** con SAM3 (Segment Anything Model) para disminuir el proceso de entrenamiento de la red.
- 📊 **Augmentación de datos** adaptada a condiciones agronómicas reales (rotaciones, variación de luz, recortes aleatorios).
- 🔬 **Artículo enviado** a la conferencia STSIVA 2026.

![Resumen del artículo presentado a la conferencia STSIVA 2026.](https://github.com/Josephts1/MercoIA/blob/main/images/pipeline.png)


## 🍍🫐🍌 Clases de diagnóstico
El módulo distingue **9 clases** distribuidas en tres cultivos. Las enfermedades seleccionadas corresponden a las de mayor incidencia en la región según revisión bibliográfica y consulta con técnicos locales:

| # | Cultivo | Clase | Descripción |
|---|---------|-------|-------------|
| 1 | 🫐 Mora | `mora_sana` | Fruto sin enfermedad visible |
| 2 | 🫐 Mora | `mora_antracnosis` | Manchas oscuras por *Colletotrichum* spp. |
| 3 | 🫐 Mora | `mora_moho_gris` | Pudrición por *Botrytis cinerea* |
| 4 | 🍍 Piña | `pina_sana` | Fruto sin enfermedad visible |
| 5 | 🍍 Piña | `pina_pudricion_negra` | Pudrición negra por *Thielaviopsis* spp. |
| 6 | 🍍 Piña | `pina_fusariosis` | Infección por *Fusarium* spp. |
| 7 | 🍌 Banano | `banano_racimo_sano` | Racimo sin enfermedad visible |
| 8 | 🍌 Banano | `banano_racimo_enfe_1` | Enfermedad tipo 1 en racimo |
| 9 | 🍌 Banano | `banano_racimo_enfe_2` | Enfermedad tipo 2 en racimo |

> **Nota sobre los bananos:** Las enfermedades de banano (clases 8 y 9) están aún en proceso de caracterización taxonómica con apoyo de especialistas. Los nombres definitivos se actualizarán en versiones futuras del dataset.
---

## 🗂️ Estructura del repositorio

```
MercoIA/
│
├── 📁 STSIVA/                        # Código para el artículo STSIVA 2026
│   └── main.py                       # Pipeline principal (SAM3 + clasificación + etapas de madurez)
│
├── 📁 codes_for_the_dataset/         # Preparación y curaduría del dataset
│   ├── Augmented_data.py             # Augmentación de imágenes para entrenamiento
│   ├── Firts_steps_efficenet.py      # Entrenamiento inicial con EfficientNet-B0
│   ├── Script_auto_label.py          # Etiquetado asistido con SAM3
│   ├── detecti_banana.py             # Detección de enfermedades en banano
│   ├── detecti_filter_mora.py        # Detección de enfermedades en mora
│   ├── detecti_filter_pina.py        # Detección de enfermedades en piña
│   ├── pdf2image.py                  # Conversión de PDFs de campo a imágenes
│   └── segment_SAM.py                # Segmentación con SAM
│
├── 📁 network/                       # Arquitectura y pesos del modelo
│   ├── First_try_network.py          # Primera iteración de la red neuronal
│   └── mejor_modelo.pth              # Mejor modelo entrenado (checkpoint)
│
├── 📁 images/                        # Recursos gráficos
│   └── pipeline.png                  # Diagrama general del pipeline
│
└── README.md
```

---
## 🔬 ¿Cómo funciona el sistema?

El pipeline está diseñado en tres etapas independientes que se encadenan:

### Etapa 1 — Construcción del dataset

Las imágenes de campo llegan en bruto (fotos de celular con múltiples frutos por encuadre, fondos ruidosos, variaciones de iluminación). El primer reto es construir un dataset limpio y equilibrado. Para esto, cada script de detección (`detecti_filter_*.py`) aplica **cuatro filtros en cascada** sobre los recortes generados por SAM3:

| Filtro | Criterio | Motivo de descarte |
|--------|----------|--------------------|
| **Forma** | Relación largo/ancho ≤ 1.6 | Elimina ramas, hojas y objetos alargados que SAM3 confunde con frutos |
| **Tamaño** | Lado mínimo ≥ 120 px | Descarta frutos muy lejanos o parcialmente visibles |
| **Oclusión** | Fondo blanco < 50–58% del recorte | Filtra frutos tapados por otros o mal segmentados |
| **Nitidez** | Ratio de energías Sobel ≥ 0.3 | Rechaza imágenes borrosas (mano temblorosa, movimiento del fruto) |

Los recortes descartados se guardan en una carpeta separada para revisión manual, lo que permite refinar los umbrales iterativamente.

### Etapa 2 — Entrenamiento del clasificador

Se utiliza **EfficientNet-B0** preentrenado en ImageNet. La decisión de usar esta arquitectura responde a tres factores: (1) su tamaño reducido la hace ejecutable en GPUs pequeñas y futura inferencia en dispositivos móviles; (2) sus pesos preentrenados capturan texturas genéricas útiles para distinguir superficies de frutos enfermos; (3) su diseño con compound scaling la hace más precisa por parámetro que alternativas como ResNet-18.

El entrenamiento incluye división por **identidad de fruto** (no por imagen), evitando que distintas fotos del mismo fruto aparezcan a la vez en train y validation —una fuga de datos frecuente en datasets de campo donde se toman varias fotos del mismo ejemplar.

### Etapa 3 — Pipeline de inferencia completo

El script `STSIVA/main.py` integra el sistema final: dada una imagen de entrada, SAM3 localiza y recorta cada fruto, el clasificador predice la clase, y un módulo colorimétrico adicional estima la **etapa de madurez** según la posición del color promedio del fruto respecto a tres centroides calibrados (verde, roja, morada) y un modelo de transición temporal entre etapas.

---

## ⚙️ Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Josephts1/MercoIA.git
cd MercoIA

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install torch torchvision opencv-python matplotlib scikit-learn ultralytics
```

---


### Pipeline completo (STSIVA 2026)

```python
# Ejecuta el pipeline principal: segmentación + clasificación + estimación de madurez
python STSIVA/main.py
```

---
