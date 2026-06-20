import re
import pymupdf
from pathlib import Path


def pdf2ima(pdf_path, save_path, prefix):
    doc = pymupdf.open(pdf_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(doc):
        pixmap = page.get_pixmap(dpi=300)
        pixmap.save(save_path / f"{prefix}_{i + 1}.png")

    doc.close()

save_path = "/home/joseph/Documents/MERCOIA/Algorithm/Our_dataset/bananos/Original"

pdfs = [
    ("/home/joseph/Documents/MERCOIA/Algorithm/Our_dataset/bananos/Bananos-20260610T024525Z-3-001/Bananos/Archivo_escaneado_20260526-2054.pdf", "cap2_26may"),
    ("/home/joseph/Documents/MERCOIA/Algorithm/Our_dataset/bananos/Bananos-20260610T024525Z-3-001/Bananos/Archivo_escaneado_20260608-1854.pdf", "cap3_8jun"),
    ("/home/joseph/Documents/MERCOIA/Algorithm/Our_dataset/bananos/Bananos-20260610T024525Z-3-001/Bananos/finca San José Svte.pdf", "cap1_25may"),
]

for pdf_path, prefix in pdfs:
    pdf2ima(pdf_path, save_path, prefix)


def rename_jpgs(folder, prefix):
    folder = Path(folder)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.(png|jpg)$")

    existing_nums = [
        int(m.group(1)) for f in folder.iterdir() if (m := pattern.match(f.name))
    ]
    next_num = max(existing_nums, default=0) + 1

    jpgs = sorted(f for f in folder.glob("*.jpg") if not pattern.match(f.name))

    for i, jpg in enumerate(jpgs):
        jpg.rename(folder / f"{prefix}_{next_num + i}.jpg")

save_path = "/home/joseph/Documents/MERCOIA/Algorithm/Our_dataset/bananos/Original"
rename_jpgs(save_path, "cap1_19may")
