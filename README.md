---
title: Classification Demandes Services
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Classification des demandes de services (FastText + FastAPI)

Classification automatique d'une demande de travaux/prestation (texte libre)
en categorie de **service** (250 classes) et **sous-service** (750 classes),
pour une plateforme de mise en relation entre particuliers et entrepreneurs.

**Demo en ligne :** _[a completer apres deploiement, voir `DEPLOIEMENT.md`]_

## Tester en 2 minutes

Les modeles sont deja entraines et inclus dans le depot (`models/*.bin`,
< 1 Mo chacun) : aucune etape de generation de donnees ni d'entrainement n'est
necessaire pour tester l'API.

```bash
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn api.main:app --port 8000
```

Ouvrir **http://127.0.0.1:8000/** : interface web (champ de texte libre,
exemples cliquables, resultats avec score de confiance). La documentation
technique interactive (Swagger) est sur **http://127.0.0.1:8000/docs**.

Ou en ligne de commande :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"description\": \"Bonjour, je cherche quelqu'un pour reparer ma chaudiere le plus vite possible.\", \"top_k\": 3}"
```

Reponse :

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

Autres endpoints : `GET /health`, `GET /services`,
`GET /services/{service}/sous_services`.

*Remarque : `fasttext-wheel` n'a pas de wheel precompile pour Python 3.13 sur
Windows. Ce projet utilise Python 3.11 (`py -3.11`), pour lequel un wheel
precompile existe.*

## Contexte du projet

Reconstruction, a des fins de portfolio, d'un projet de NLP realise en
entreprise pour une plateforme de mise en relation particuliers/entrepreneurs
(demandes de travaux a domicile). La plateforme recoit un texte libre decrivant
un besoin et doit le classer automatiquement dans une categorie de service et
de sous-service.

Les donnees et le code originaux (proprietaires) ne sont pas disponibles ici :
ce depot reconstitue le pipeline complet (generation de donnees synthetiques,
augmentation/reequilibrage des classes, entrainement FastText, API) a partir de
deux scripts du projet d'origine (`data_augmentation.py`, `train_model.py`),
adaptes pour etre autonomes et reproductibles sur une machine sans GPU (4 Go de
RAM).

## Architecture du pipeline

| Script | Entree | Sortie |
|---|---|---|
| `generate_taxonomy.py` | - | `data/taxonomy.csv` (250 services, 750 sous-services) |
| `generate_synthetic_data.py` | `data/taxonomy.csv` | `data/raw_descriptions.csv` (description, service, sous_service) |
| `data_augmentation.py` | `data/raw_descriptions.csv` | `data/train_*.txt`, `data/test_*.txt` (format FastText, classes reequilibrees) |
| `train_model.py` | `data/train_*.txt`, `data/test_*.txt` | `models/service.bin`, `models/sous_service.bin` |
| `api/main.py` + `static/index.html` | `models/*.bin`, `data/taxonomy.csv` | API FastAPI + interface web de prediction |

Pour regenerer les donnees et reentrainer depuis zero :

```bash
.venv\Scripts\python generate_taxonomy.py
.venv\Scripts\python generate_synthetic_data.py
.venv\Scripts\python data_augmentation.py
.venv\Scripts\python train_model.py
```

## Choix techniques

- **Modele : FastText.** Entrainement et inference rapides sur CPU seul, faible
  empreinte memoire, bien adapte a de la classification de texte court sur un
  grand nombre de classes (750 ici) sans necessiter de GPU.
- **Taille des modeles.** `bucket=20000` et `dim=50` (au lieu des valeurs par
  defaut, `2000000` et `100`, surdimensionnees pour un vocabulaire synthetique
  restreint) combines a une quantization (`model.quantize()`) : modeles finaux
  de moins de 1 Mo chacun, pour une perte de precision negligeable.
- **Augmentation des donnees.** Reequilibrage des classes minoritaires par
  sur-echantillonnage (facteur plafonne par rapport a la classe majoritaire),
  via trois transformations de texte en rotation (synonymes, permutation,
  suppression de mots) — meme strategie que le script original, qui s'appuyait
  sur `nlpaug`/WordNet francais ; remplace ici par un augmenteur leger afin que
  la demo reste reproductible hors-ligne.
- **Python 3.11 dedie au projet** (voir `requirements.txt`), car `fasttext-wheel`
  ne fournit pas de wheel precompile pour Python 3.13 sur Windows.
- **Coherence service / sous-service.** Les deux modeles sont entraines
  independamment et peuvent se contredire (ex: le modele service propose un
  domaine different de celui du sous-service predit). L'API derive le service
  affiche a partir du parent du sous-service predit (via `data/taxonomy.csv`),
  ce qui garantit un resultat toujours coherent entre les deux niveaux.
- **Interface web.** Page unique en HTML/CSS/JS natif (`static/index.html`,
  servie par l'API), sans framework ni etape de build.

## Deploiement

Ce projet est conteneurise (`Dockerfile`) pour etre deployable gratuitement
avec une URL publique (Hugging Face Spaces ou Render). Voir **`DEPLOIEMENT.md`**
pour les etapes detaillees (publication GitHub + mise en ligne).

## Resultats

precision@1 sur le jeu de test : ~99.8% (sous-service, 750 classes), 100%
(service, 250 classes). A interpreter comme une validation du pipeline de bout
en bout : les donnees etant synthetiques et generees par templates, ces scores
ne sont pas representatifs d'une performance en production sur des donnees
utilisateurs reelles (voir limites ci-dessous).

## Limites et perimetre de la demo

- Donnees entierement synthetiques (templates de phrases), avec une
  repartition de classes volontairement desequilibree pour simuler un cas
  realiste.
- Les 250 services / 750 sous-services sont generes par combinaison
  (domaine x type d'intervention), et non issus d'un referentiel metier reel.
- Sur des descriptions qui melangent plusieurs domaines dans une meme phrase
  (ex: un mariage mentionnant a la fois un traiteur et un photographe), le
  modele de sous-service peut se tromper : les exemples d'entrainement ne
  portent chacun que sur un seul besoin.
