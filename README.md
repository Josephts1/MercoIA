---

## 📖 ¿Qué es MercoIA?

**MercoIA** es el módulo de clasificación de enfermedades del proyecto homónimo, desarrollado en el marco del *Proyecto de Investigación — Convenio 268-2025*. Su propósito es asistir a agricultores de la región de **Santander (Colombia)** en la identificación temprana de enfermedades en cultivos de **piña, mora y banano** a partir de fotografías tomadas con su propio celular, sin necesidad de equipos especializados ni acceso a laboratorio.

El problema que motiva este trabajo es concreto: las enfermedades fúngicas en estas tres especies son una de las principales causas de pérdida poscosecha en la región, pero su diagnóstico oportuno depende de técnicos agrícolas que no siempre están disponibles a tiempo. MercoIA busca cerrar esa brecha poniendo una herramienta de diagnóstico directamente en el bolsillo del productor.

---

## ✨ Características principales

- 📱 **Pensado para celulares** — funciona con fotos tomadas en condiciones reales de campo (luz y ángulos variables, distintas distancias).
- 🔬 **9 clases de diagnóstico** cubriendo las enfermedades más frecuentes en tres cultivos regionales.
- 🤖 **EfficientNet-B0** preentrenado en ImageNet con fine-tuning especializado en frutas.
- 🏷️ **Etiquetado asistido** con SAM3 (Segment Anything Model) para disminuir el proceso de entrenamiento de la red.
- 📊 **Augmentación de datos** adaptada a condiciones agronómicas reales (rotaciones, variación de luz, recortes aleatorios).
- 🔬 **Artículo enviado** a la conferencia STSIVA 2026.

![Resumen del artículo presentado a la conferencia STSIVA 2026.](https://github.com/Josephts1/MercoIA/blob/main/images/pipeline.png)


## 🍍🫐🍌 Clases de diagnóstico

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

---

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
│   └── mejor_modelo.pth              # ✅ Mejor modelo entrenado (checkpoint)
│
├── 📁 images/                        # Recursos gráficos
│   └── pipeline.png                  # Diagrama general del pipeline
│
└── README.md
```

---

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
