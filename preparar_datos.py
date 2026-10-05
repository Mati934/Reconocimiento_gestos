"""Reparte las fotos de images/ en data/train y data/test, reducidas a 512 px.

Uso: uv run python preparar_datos.py

Las fotos se agrupan en "tandas": fotos consecutivas del mismo gesto, casi
iguales entre sí (máximo MAX_RUN fotos por tanda). Una tanda completa va a
train o a test, nunca se divide, para que el test tenga fotos realmente nuevas
para el modelo.
"""

import csv
import random
import re
from pathlib import Path

from PIL import Image, ImageOps

IMAGES_DIR = Path("images")
DATA_DIR = Path("data")
CLASSES = ["ok", "otros", "paz", "pulgar_arriba", "stop"]
MAX_SIDE = 512
TEST_FRACTION = 0.2
MIN_TEST = 10
# Las ráfagas largas se cortan en tandas de este tamaño máximo; si no, una
# ráfaga de 90 fotos obligaría a mandar todas las demás fotos de la clase a test.
MAX_RUN = 15
SEED = 42

NAME_PATTERN = re.compile(r"^(?P<label>.+)_(?P<id>\d{4})\.jpe?g$", re.IGNORECASE)


def read_photos():
    photos = []
    for path in IMAGES_DIR.iterdir():
        match = NAME_PATTERN.match(path.name)
        if match and match["label"] in CLASSES:
            photos.append((int(match["id"]), match["label"], path))
    return sorted(photos)


def group_runs(photos):
    """Agrupa fotos consecutivas (por número) con el mismo gesto, en tandas de hasta MAX_RUN."""
    runs = []
    for photo_id, label, path in photos:
        if (
            runs
            and runs[-1]["label"] == label
            and runs[-1]["last_id"] == photo_id - 1
            and len(runs[-1]["paths"]) < MAX_RUN
        ):
            runs[-1]["paths"].append(path)
            runs[-1]["last_id"] = photo_id
        else:
            runs.append({"label": label, "first_id": photo_id, "last_id": photo_id, "paths": [path]})
    return runs


def choose_test_runs(runs, rng):
    """Elige tandas al azar para test hasta llegar a ~20 % (mínimo 10 fotos)."""
    total = sum(len(run["paths"]) for run in runs)
    target = max(MIN_TEST, round(TEST_FRACTION * total))
    shuffled = runs[:]
    rng.shuffle(shuffled)

    chosen, count = [], 0
    for run in shuffled:
        if count >= target:
            break
        if count + len(run["paths"]) <= target + 2:
            chosen.append(run)
            count += len(run["paths"])

    # Si las tandas pequeñas no alcanzan, se completa con la más chica restante.
    for run in sorted(shuffled, key=lambda r: len(r["paths"])):
        if count >= target:
            break
        if run not in chosen:
            chosen.append(run)
            count += len(run["paths"])
    return chosen


def save_resized(source, destination):
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        image.save(destination, "JPEG", quality=90)


def main():
    rng = random.Random(SEED)
    photos = read_photos()
    rows = []

    for split in ("train", "test"):
        for label in CLASSES:
            folder = DATA_DIR / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for old_file in folder.glob("*.jpg"):
                old_file.unlink()

    for label in CLASSES:
        runs = group_runs([photo for photo in photos if photo[1] == label])
        test_runs = choose_test_runs(runs, rng)

        for run_number, run in enumerate(runs, start=1):
            split = "test" if run in test_runs else "train"
            for path in run["paths"]:
                destination = DATA_DIR / split / label / f"{path.stem}.jpg"
                save_resized(path, destination)
                rows.append([destination.as_posix(), path.name, label, split, f"{label}_tanda{run_number:02d}"])

    with open(DATA_DIR / "particion.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["archivo", "original", "clase", "conjunto", "tanda"])
        writer.writerows(rows)

    print(f"{'clase':<15}{'train':>7}{'test':>7}")
    for label in CLASSES:
        train = sum(1 for row in rows if row[2] == label and row[3] == "train")
        test = sum(1 for row in rows if row[2] == label and row[3] == "test")
        warning = "  <- faltan fotos (mínimo 50 train / 10 test)" if train < 50 or test < 10 else ""
        print(f"{label:<15}{train:>7}{test:>7}{warning}")


if __name__ == "__main__":
    main()
