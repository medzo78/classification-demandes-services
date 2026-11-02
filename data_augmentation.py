"""
Reequilibrage des classes par augmentation de texte, avant entrainement FastText.

Reprend la logique du pipeline original (uid_service / uid_subservice, calcul
d'un facteur de multiplication par classe par rapport a la classe majoritaire,
puis generation de variantes de texte pour les classes minoritaires) adaptee
au nouveau schema (description / service / sous_service) et a un jeu de
donnees local (pas de S3).

Difference volontaire avec l'original : le projet en entreprise utilisait
nlpaug (SynonymAug sur wordnet francais + RandomWordAug swap/delete), qui
necessite de telecharger les corpus NLTK/wordnet-fra a l'execution. Pour que
cette demo reste 100% reproductible hors-ligne et rapide sur une machine
modeste (CPU seul, 4 Go de RAM), l'augmentation par synonymes est remplacee
par un petit dictionnaire de synonymes "maison" ; la strategie globale
(swap / delete / synonym en rotation, facteur de multiplication plafonne au
niveau de la classe majoritaire) est identique.

Usage:
    python generate_taxonomy.py
    python generate_synthetic_data.py
    python data_augmentation.py
Produit, dans data/ :
    train_service.txt, test_service.txt
    train_sous_service.txt, test_sous_service.txt
"""
import math
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
THRESHOLD = 4  # taille mini d'une classe pour etre conservee (200 dans l'original, a l'echelle d'un tres gros jeu de donnees)
MAX_MULTIPLICATION_FACTOR = 15  # plafond du sur-echantillonnage (absent/illimite dans l'original, ajoute ici vu l'echelle du jeu synthetique)

SYNONYMS = {
    "reparer": ["depanner", "remettre en etat", "arranger"],
    "installer": ["poser", "mettre en place", "monter"],
    "entretenir": ["faire l'entretien de", "verifier", "revisioner"],
    "renover": ["refaire", "moderniser", "retaper"],
    "remplacer": ["changer", "substituer"],
    "obtenir": ["avoir", "recevoir", "demander"],
    "urgence": ["urgent", "en priorite"],
    "rapidement": ["vite", "sans tarder"],
    "devis": ["estimation", "tarif"],
    "professionnel": ["artisan", "prestataire", "expert"],
    "maison": ["logement", "domicile"],
    "appartement": ["logement"],
    "petit": ["petite", "modeste"],
    "grand": ["grande", "important"],
    "bonjour": ["bonjour a vous", "bonjour,"],
    "merci": ["merci bien", "merci d'avance"],
    "cherche": ["recherche", "suis a la recherche de"],
    "besoin": ["nécessité", "demande"],
}


def synonym_augment(rng, text):
    words = text.split(" ")
    out = []
    changed = False
    for w in words:
        key = w.lower().strip(".,?!:;")
        if not changed and key in SYNONYMS and rng.random() < 0.7:
            out.append(rng.choice(SYNONYMS[key]))
            changed = True
        else:
            out.append(w)
    return " ".join(out)


def swap_augment(rng, text):
    words = text.split(" ")
    if len(words) < 3:
        return text
    i = rng.randint(0, len(words) - 2)
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def delete_augment(rng, text):
    words = text.split(" ")
    if len(words) < 4:
        return text
    i = rng.randint(1, len(words) - 2)  # garde le premier et le dernier mot
    del words[i]
    return " ".join(words)


def df_to_fasttext_file(df, filename, label_col, text_col):
    with open(filename, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            label = str(row[label_col]).strip().replace(" ", "_")
            text = str(row[text_col]).strip().replace("\n", " ")
            f.write(f"__label__{label} {text}\n")


def multiplication_factors(df, label_col):
    counts = df[label_col].value_counts()  # trie decroissant
    list_label = list(counts.index)
    taux_multiple = [math.floor(counts.max() / c) for c in counts]
    taux_multiple = [min(t, MAX_MULTIPLICATION_FACTOR) for t in taux_multiple]
    # classes deja proches de la classe majoritaire (facteur 1) : pas d'augmentation
    elim = max(i for i in range(len(taux_multiple)) if taux_multiple[i] == 1)
    return list_label[: elim + 1], list_label[elim + 1 :], taux_multiple[elim + 1 :]


def augment_class(rng, df, label_col, text_col, extra_cols, class_lab, nb):
    rows = df[df[label_col] == class_lab]
    out_texts = []
    out_extra = {c: [] for c in extra_cols}

    for i, (_, row) in enumerate(rows.iterrows()):
        text = str(row[text_col])
        out_texts.append(text)
        for c in extra_cols:
            out_extra[c].append(row[c])

        n_variants = nb - 1
        if n_variants <= 0:
            continue
        for k in range(n_variants):
            mod = (i + k) % 3
            if mod == 0:
                variant = synonym_augment(rng, text)
            elif mod == 1:
                variant = swap_augment(rng, text)
            else:
                variant = delete_augment(rng, text)
            out_texts.append(variant)
            for c in extra_cols:
                out_extra[c].append(row[c])

    result = {text_col: out_texts, label_col: [class_lab] * len(out_texts)}
    result.update(out_extra)
    return pd.DataFrame(result)


def build_balanced_split(rng, df, label_col, text_col, extra_cols, out_prefix):
    df = df.dropna(subset=[text_col, label_col]).copy()
    df = df.groupby(label_col).filter(lambda x: len(x) >= THRESHOLD)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df[label_col].to_numpy()
    )

    df_to_fasttext_file(test_df, os.path.join("data", f"test_{out_prefix}.txt"), label_col, text_col)

    intact_labels, labels_to_augment, factors = multiplication_factors(train_df, label_col)

    parts = [train_df[train_df[label_col].isin(intact_labels)]]
    for label, factor in zip(labels_to_augment, factors):
        parts.append(augment_class(rng, train_df, label_col, text_col, extra_cols, label, factor))

    augmented_train_df = pd.concat(parts, axis=0).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_to_fasttext_file(
        augmented_train_df, os.path.join("data", f"train_{out_prefix}.txt"), label_col, text_col
    )

    print(
        f"[{out_prefix}] classes: {df[label_col].nunique()} | "
        f"train brut: {len(train_df)} -> train augmente: {len(augmented_train_df)} | "
        f"test: {len(test_df)}"
    )


def main():
    rng = random.Random(SEED)
    raw_path = os.path.join("data", "raw_descriptions.csv")
    df = pd.read_csv(raw_path)

    # I. prediction du sous_service
    build_balanced_split(
        rng, df, label_col="sous_service", text_col="description",
        extra_cols=["service"], out_prefix="sous_service",
    )

    # II. prediction du service
    build_balanced_split(
        rng, df, label_col="service", text_col="description",
        extra_cols=[], out_prefix="service",
    )


if __name__ == "__main__":
    main()
