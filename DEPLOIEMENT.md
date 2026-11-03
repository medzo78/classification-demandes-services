# Guide de deploiement

Ce document decrit, pas a pas, comment publier ce projet sur GitHub puis le
deployer gratuitement sur une URL accessible (Hugging Face Spaces), pour
qu'un tiers (ex: un recruteur) puisse le tester sans rien installer.

## 0. Pre-requis

- Un compte [GitHub](https://github.com) (gratuit).
- Un compte [Hugging Face](https://huggingface.co) (gratuit) — utilise pour
  l'hebergement de l'API/frontend (solution recommandee ici car le Docker
  SDK des Spaces heberge gratuitement n'importe quelle application
  conteneurisee, y compris une API FastAPI).
- Git installe en local (deja le cas sur ce poste).

## 1. Verifier/corriger l'identite Git locale

Les commits de ce depot doivent porter votre nom et votre vrai email (visible
publiquement une fois pousse sur GitHub) :

```bash
cd "chemin\vers\classification-demandes-services"
git config user.name "Votre Nom"
git config user.email "votre.email@example.com"
```

Si des commits existent deja avec une mauvaise identite, corrigez le dernier
avec :

```bash
git commit --amend --reset-author --no-edit
```

## 2. Publier le code sur GitHub

1. Aller sur [github.com/new](https://github.com/new), creer un depot **vide**
   (sans README/licence, pour eviter un conflit avec les fichiers existants),
   par exemple nomme `classification-demandes-services`.
2. Dans le dossier du projet :

```bash
git remote add origin https://github.com/<votre-utilisateur>/classification-demandes-services.git
git branch -M main
git push -u origin main
```

Le code (y compris les modeles entraines, < 1 Mo chacun) est maintenant
visible sur GitHub.

## 3. Deployer sur Hugging Face Spaces (Docker)

1. Aller sur [huggingface.co/new-space](https://huggingface.co/new-space).
2. Renseigner : nom du Space (ex: `classification-demandes-services`), SDK
   **Docker**, visibilite **Public**.
3. Une fois le Space cree, Hugging Face fournit l'URL git du Space, de la
   forme `https://huggingface.co/spaces/<votre-utilisateur>/<nom-du-space>`.
4. Ajouter ce Space comme second remote Git et y pousser le meme code :

```bash
git remote add space https://huggingface.co/spaces/<votre-utilisateur>/<nom-du-space>
git push space main
```

   Lors de l'authentification, utiliser votre nom d'utilisateur Hugging Face
   et, comme mot de passe, un **access token** (a generer dans
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
   permission "write").

5. Hugging Face construit automatiquement l'image Docker (fichier
   `Dockerfile` a la racine) et demarre l'application. Suivre la progression
   dans l'onglet "Logs" du Space (quelques minutes la premiere fois).
6. Une fois le build termine, l'application est accessible publiquement a
   l'adresse `https://<votre-utilisateur>-<nom-du-space>.hf.space` (interface
   web) et `https://<votre-utilisateur>-<nom-du-space>.hf.space/docs` (API).

Le fichier `README.md` contient deja l'en-tete YAML attendu par Hugging Face
(`sdk: docker`, `app_port: 7860`) : aucune configuration supplementaire n'est
necessaire.

### Mettre a jour le Space apres une modification

```bash
git add -A
git commit -m "votre message"
git push origin main   # met a jour GitHub
git push space main    # redeploie sur Hugging Face
```

## 4. Alternative : Render.com

Si vous preferez ne pas creer de compte Hugging Face, Render.com propose un
niveau gratuit qui deploie directement depuis un depot GitHub :

1. Pousser le code sur GitHub (etape 2 ci-dessus).
2. Sur [render.com](https://render.com), "New +" -> "Web Service", connecter
   le depot GitHub.
3. Render detecte le `Dockerfile` automatiquement (environnement "Docker").
   Laisser le champ "Port" vide (l'image lit la variable `PORT` fournie par
   Render).
4. Choisir le plan gratuit et deployer.

Limite du plan gratuit Render : le service se met en veille apres 15 minutes
sans trafic, la premiere requete suivante prend alors ~30-60 secondes (temps
de reveil). Hugging Face Spaces n'a pas cette contrainte au meme degre pour
un usage de demonstration ponctuelle.

## 5. Verification post-deploiement

Une fois l'URL publique obtenue, verifier :

```bash
curl https://<votre-url>/health
curl -X POST https://<votre-url>/predict \
  -H "Content-Type: application/json" \
  -d "{\"description\": \"Je cherche un plombier pour une fuite d'eau urgente.\", \"top_k\": 3}"
```

Et ouvrir `https://<votre-url>/` dans un navigateur pour tester l'interface.
