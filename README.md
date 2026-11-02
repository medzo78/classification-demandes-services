# Classification des demandes de services (FastText + FastAPI)

## Contexte

Reconstruction, a des fins de demonstration pour un entretien, d'un projet de
NLP realise en entreprise pour une plateforme de mise en relation entre
particuliers et entrepreneurs (demandes de travaux a domicile). La plateforme
recoit un texte libre decrivant un besoin ("je cherche quelqu'un pour reparer
ma chaudiere en urgence") et doit le classer automatiquement dans :

- une categorie de **service** (250 classes, ex: "Plomberie - Reparation")
- une categorie de **sous-service**, plus fine (750 classes, ex:
  "Reparation - chaudiere")

Les donnees et le code originaux (proprietaires, stockes sur S3) ne sont pas
disponibles ici. Ce depot reconstitue le pipeline complet (generation de
donnees synthetiques -> augmentation -> entrainement FastText -> API) a partir
de deux scripts recuperes du projet d'origine (`data_augmentation.py`,
`train_model.py`), adaptes pour fonctionner de maniere autonome et
reproductible sur une machine modeste (CPU seul, 4 Go de RAM).

**Difference assumee avec l'original** : la version en entreprise utilisait
`nlpaug` (augmentation par synonymes via WordNet francais) ; cette demo utilise
un petit augmenteur "maison" (synonymes/permutation/suppression de mots) pour
rester 100% reproductible hors-ligne et rapide a l'execution. La strategie de
reequilibrage des classes (sur-echantillonnage des classes minoritaires par
rapport a la classe majoritaire, plafonne) est identique a l'original.

## Pipeline

```
generate_taxonomy.py        -> data/taxonomy.csv            (250 services, 750 sous-services)
generate_synthetic_data.py  -> data/raw_descriptions.csv     (description, service, sous_service)
data_augmentation.py        -> data/train_*.txt, test_*.txt  (format FastText, classes reequilibrees)
train_model.py               -> models/service.bin, models/sous_service.bin
api/main.py                  -> API FastAPI de prediction
```

## Installation

Le projet utilise un environnement virtuel dedie. `fasttext-wheel` ne fournit
pas de wheel precompile pour Python 3.13 sur Windows (compilation C++
necessaire) : ce projet utilise donc **Python 3.11**, pour lequel un wheel
precompile existe (pas de compilateur necessaire).

```bash
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Demarrage rapide (modeles deja entraines fournis dans le depot)

Les fichiers `models/service.bin` et `models/sous_service.bin` sont inclus
(moins de 1 Mo chacun) : pas besoin de reentrainer pour tester l'API.

```bash
.venv\Scripts\uvicorn api.main:app --port 8000
```

Puis ouvrir **http://127.0.0.1:8000/docs** pour tester `/predict` directement
depuis le navigateur (Swagger UI), sans ecrire de code.

## Reproduire le pipeline depuis zero (regenerer donnees + reentrainer)

```bash
.venv\Scripts\python generate_taxonomy.py
.venv\Scripts\python generate_synthetic_data.py
.venv\Scripts\python data_augmentation.py
.venv\Scripts\python train_model.py
```

## Lancer l'API

```bash
.venv\Scripts\uvicorn api.main:app --reload --port 8000
```

Puis ouvrir **http://127.0.0.1:8000/docs** : documentation interactive
(Swagger UI) permettant de tester `/predict` directement depuis le navigateur,
sans ecrire de code.

Exemple d'appel :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"description\": \"Bonjour, je cherche quelqu'un pour reparer ma chaudiere le plus vite possible.\", \"top_k\": 3}"
```

Reponse (exemple reel obtenu avec ce depot) :

```json
{
  "description": "Bonjour, je cherche quelqu'un pour reparer ma chaudiere le plus vite possible.",
  "service": [
    {"label": "Chauffage - Reparation", "confidence": 0.9175},
    {"label": "Serrurerie - Reparation", "confidence": 0.0557},
    {"label": "Maconnerie - Reparation", "confidence": 0.0248}
  ],
  "sous_service": [
    {"label": "Reparation - chaudiere", "confidence": 0.6407},
    {"label": "Reparation - serrure", "confidence": 0.0303},
    {"label": "Reparation - porte blindee", "confidence": 0.0243}
  ]
}
```

Autres endpoints utiles : `GET /health`, `GET /services`,
`GET /services/{service}/sous_services`.

## Choix du modele : FastText

FastText a ete retenu (comme dans le projet original) car il est tres rapide a
entrainer et a l'inference sur CPU seul, avec une empreinte memoire faible
(adapte a une machine sans GPU et avec 4 Go de RAM), tout en restant performant
sur de la classification de texte court multi-classes (750 classes ici).

## Resultats et points techniques

- precision@1 sur le jeu de test : **~99.8%** (sous-service, 750 classes) et
  **100%** (service, 250 classes). Ces scores tres eleves s'expliquent par la
  nature synthetique/templatee des donnees (cf. limites ci-dessous) : a
  interpreter comme une validation du pipeline, pas comme une performance
  transposable a des donnees reelles.
- **Taille des modeles** : la configuration FastText par defaut (`bucket`
  a 2 000 000, `dim=100`) produisait deux modeles de ~800 Mo chacun, trop
  volumineux pour un depot de demo. Reduire `bucket` a 20 000 et `dim` a 50
  (le vocabulaire synthetique est restreint, pas besoin d'autant de buckets
  de hachage), puis appliquer `model.quantize()`, ramene chaque modele a
  **moins de 1 Mo**, pour une perte de precision negligeable (<0.2 point).
- **Compatibilite fasttext / numpy** : `fasttext-wheel` 0.9.2 appelle en
  interne `np.array(probs, copy=False)`, qui leve une `ValueError` avec
  numpy>=2.0 (semantique stricte du parametre `copy`). Plutot que de figer
  numpy<2 (ce qui serait incompatible avec la version de `pandas` utilisee
  par ailleurs), l'API applique un monkeypatch cible et documente dans
  `api/main.py` (ne reecrit que le cas `copy=False`, retombe sur `np.asarray`).
- **Python 3.11 dedie** : `fasttext-wheel` n'a pas de wheel precompile pour
  Python 3.13 sur Windows (compilation C++ requise, pas de compilateur
  disponible sur la machine de dev) ; le projet installe et utilise Python 3.11,
  pour lequel un wheel precompile existe.

## Limites de cette demo

- Donnees entierement synthetiques (templates de phrases), generees pour
  reproduire une repartition de classes realiste (desequilibree) ; les scores
  de precision ne refletent donc pas des performances en production sur des
  donnees reelles d'utilisateurs.
- Les 250 services / 750 sous-services sont generes par combinaison
  (domaine x type d'intervention) plutot qu'issus d'un referentiel metier reel.
- Sur des descriptions qui melangent plusieurs domaines dans une meme phrase
  (ex: un mariage qui mentionne a la fois un traiteur et un photographe), le
  modele de sous-service peut se tromper : chaque exemple d'entrainement ne
  porte qu'un seul besoin a la fois, ce cas n'est donc pas bien couvert.
