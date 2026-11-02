"""
Entraine les deux modeles FastText (service, sous_service) a partir des
fichiers generes par data_augmentation.py, et les sauvegarde dans models/.

Usage:
    python train_model.py
"""
import os

import fasttext

MODELS_DIR = "models"
DATA_DIR = "data"


def train_and_evaluate(label_name, epoch):
    train_path = os.path.join(DATA_DIR, f"train_{label_name}.txt")
    test_path = os.path.join(DATA_DIR, f"test_{label_name}.txt")

    model = fasttext.train_supervised(
        train_path,
        epoch=epoch,
        lr=0.7,
        wordNgrams=2,
        dim=50,
        bucket=20000,  # vocabulaire synthetique limite: pas besoin des 2M buckets par defaut
        loss="softmax",
    )

    n, precision, recall = model.test(test_path)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"[{label_name}] test examples: {n}")
    print(f"[{label_name}] precision@1 (avant quantization): {precision:.4f}")
    print(f"[{label_name}] recall@1    (avant quantization): {recall:.4f}")
    print(f"[{label_name}] f1@1        (avant quantization): {f1:.4f}")

    # quantization: reduit fortement la taille du modele sur disque (utile pour
    # le distribuer dans un repo git), au prix d'une legere perte de precision
    model.quantize(input=train_path, retrain=True, qnorm=True, cutoff=100000)
    n, precision, recall = model.test(test_path)
    print(f"[{label_name}] precision@1 (apres quantization):  {precision:.4f}")
    print(f"[{label_name}] recall@1    (apres quantization):  {recall:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"{label_name}.bin")
    model.save_model(out_path)
    size_mb = os.path.getsize(out_path) / 1_000_000
    print(f"[{label_name}] modele sauvegarde: {out_path} ({size_mb:.1f} MB)\n")
    return model


def main():
    # sous_service : 750 classes, texte plus specifique -> plus d'epochs
    train_and_evaluate("sous_service", epoch=40)
    # service : 250 classes, un peu moins d'epochs suffit
    train_and_evaluate("service", epoch=25)


if __name__ == "__main__":
    main()
