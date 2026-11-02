"""
Genere un jeu de donnees synthetique (texte de demande -> service / sous-service)
pour simuler les demandes de travaux/prestations recues par la plateforme.

Les volumes par classe sont volontairement desequilibres (certaines categories
tres frequentes, d'autres rares) pour reproduire une situation réaliste que
``data_augmentation.py`` doit ensuite corriger avant l'entrainement.

Usage:
    python generate_taxonomy.py          # (prealable)
    python generate_synthetic_data.py
Produit:
    data/raw_descriptions.csv  (description, service, sous_service)
"""
import csv
import os
import random

from generate_taxonomy import SERVICE_DOMAINS, SERVICE_SUBTYPES, TECH_DOMAINS, TECH_SUBTYPES, build_taxonomy

SEED = 42

# phrase naturelle (avec article) utilisee dans les phrases generees, par objet
OBJECT_PHRASES = {
    # --- domaines techniques ---
    "sanitaires": "mes sanitaires",
    "chauffe-eau": "mon chauffe-eau",
    "tuyauterie": "ma tuyauterie",
    "tableau electrique": "mon tableau electrique",
    "prises et interrupteurs": "mes prises et interrupteurs",
    "eclairage": "mon eclairage",
    "murs interieurs": "mes murs interieurs",
    "facade exterieure": "ma facade exterieure",
    "plafonds": "mes plafonds",
    "portes": "mes portes",
    "fenetres": "mes fenetres",
    "placards sur mesure": "mes placards sur mesure",
    "pelouse": "ma pelouse",
    "haies et arbustes": "mes haies et arbustes",
    "arrosage automatique": "mon arrosage automatique",
    "menage courant": "le menage courant de mon logement",
    "vitres": "mes vitres",
    "fin de chantier": "le nettoyage de fin de chantier",
    "climatiseur mural": "mon climatiseur mural",
    "pompe a chaleur": "ma pompe a chaleur",
    "ventilation": "ma ventilation",
    "chaudiere": "ma chaudiere",
    "radiateurs": "mes radiateurs",
    "plancher chauffant": "mon plancher chauffant",
    "tuiles": "mes tuiles",
    "gouttieres": "mes gouttieres",
    "etancheite": "l'etancheite de mon toit",
    "mur porteur": "mon mur porteur",
    "dalle beton": "ma dalle en beton",
    "ouverture de mur": "une ouverture dans mon mur",
    "sol interieur": "le carrelage de mon sol interieur",
    "faience salle de bain": "la faience de ma salle de bain",
    "terrasse exterieure": "le carrelage de ma terrasse",
    "serrure": "ma serrure",
    "porte blindee": "ma porte blindee",
    "coffre-fort": "mon coffre-fort",
    "lave-linge": "mon lave-linge",
    "refrigerateur": "mon refrigerateur",
    "four et plaques": "mon four et mes plaques de cuisson",
    "isolation combles": "l'isolation de mes combles",
    "isolation murs": "l'isolation de mes murs",
    "fenetres double vitrage": "mes fenetres en double vitrage",
    "montage de meubles": "le montage de mes meubles",
    "pose d'etageres": "la pose de mes etageres",
    "petites reparations": "de petites reparations diverses",
    # --- domaines prestations de service ---
    "cartons et emballage": "l'emballage de mes cartons",
    "transport de meubles": "le transport de mes meubles",
    "garde-meuble": "une solution de garde-meuble",
    "ordinateur": "mon ordinateur",
    "reseau wifi": "mon reseau wifi",
    "sauvegarde de donnees": "la sauvegarde de mes donnees",
    "mathematiques": "des cours particuliers de mathematiques",
    "francais": "des cours particuliers de francais",
    "langues etrangeres": "des cours de langues etrangeres",
    "remise en forme": "un programme de remise en forme",
    "musculation": "un coaching en musculation",
    "preparation physique": "une preparation physique",
    "mariage": "l'organisation de mon mariage",
    "anniversaire": "l'organisation d'un anniversaire",
    "soiree d'entreprise": "l'organisation d'une soiree d'entreprise",
    "coiffure a domicile": "une prestation de coiffure a domicile",
    "massage": "un massage a domicile",
    "manucure": "une manucure a domicile",
    "papier peint": "la pose de papier peint",
    "amenagement de piece": "l'amenagement d'une piece",
    "conseil deco": "un conseil en decoration",
    "portrait": "une seance photo portrait",
    "evenement": "la photographie d'un evenement",
    "immobilier": "des photos immobilieres",
    "buffet": "un buffet traiteur",
    "repas assis": "un repas assis traiteur",
    "cocktail": "un cocktail traiteur",
    "livraison de colis": "la livraison d'un colis",
    "transport de personnes": "le transport de personnes",
    "location avec chauffeur": "une location de vehicule avec chauffeur",
}

VERB_TECH = {
    "Installation": "installer {obj}",
    "Reparation": "reparer {obj}",
    "Entretien": "entretenir {obj}",
    "Renovation": "renover {obj}",
    "Depannage urgent": "depanner en urgence {obj}",
    "Devis et conseil": "obtenir un devis pour {obj}",
    "Mise aux normes": "mettre aux normes {obj}",
    "Remplacement": "remplacer {obj}",
    "Diagnostic": "faire diagnostiquer {obj}",
    "Petits travaux": "effectuer de petits travaux sur {obj}",
}

VERB_SERVICE = {
    "Devis et conseil": "obtenir un devis pour {obj}",
    "Prestation standard": "reserver {obj}",
    "Prestation premium": "reserver {obj} en formule premium",
    "Intervention ponctuelle": "organiser {obj} ponctuellement",
    "Formule decouverte": "essayer {obj} en formule decouverte",
    "Abonnement regulier": "souscrire un abonnement regulier pour {obj}",
    "Prestation express": "obtenir {obj} en formule express",
    "Intervention a domicile": "beneficier a domicile de {obj}",
    "Organisation complete": "faire organiser integralement {obj}",
    "Accompagnement personnalise": "beneficier d'un accompagnement personnalise pour {obj}",
}

OPENINGS = [
    "Bonjour, je cherche quelqu'un pour {verb}.",
    "Je souhaite {verb}.",
    "Recherche un professionnel pour {verb}.",
    "Besoin d'un artisan pour {verb}.",
    "Je voudrais {verb}.",
    "Pouvez-vous m'aider a {verb} ?",
    "Je suis a la recherche d'un prestataire pour {verb}.",
    "Demande : {verb}.",
    "Bonjour, il me faudrait {verb}.",
    "J'aimerais {verb}.",
]

ROOMS = [
    "dans ma salle de bain",
    "dans ma cuisine",
    "dans mon appartement",
    "dans ma maison",
    "au sous-sol",
    "dans le jardin",
    "sur la terrasse",
    "dans mon garage",
    "au rez-de-chaussee",
    "a l'etage",
    "",
]

URGENCY = [
    "le plus vite possible",
    "dans les prochains jours",
    "en urgence",
    "avant la fin du mois",
    "sans urgence particuliere",
    "",
]

CLOSINGS = [
    "Merci de me faire un devis.",
    "Quel est votre tarif ?",
    "Disponible rapidement svp.",
    "Budget limite, merci de me proposer une solution adaptee.",
    "Pouvez-vous intervenir cette semaine ?",
    "Merci d'avance pour votre reponse.",
    "",
]

# poids de popularite (classes desequilibrees, comme dans un vrai jeu de donnees)
TECH_DOMAIN_WEIGHT = {
    "Plomberie": 5, "Electricite": 5, "Peinture": 4, "Menuiserie": 3, "Jardinage": 4,
    "Nettoyage": 4, "Climatisation": 3, "Chauffage": 4, "Toiture": 2, "Maconnerie": 2,
    "Carrelage": 3, "Serrurerie": 2, "Reparation electromenager": 3,
    "Renovation energetique": 2, "Bricolage general": 4,
}
SERVICE_DOMAIN_WEIGHT = {
    "Demenagement": 3, "Informatique": 3, "Cours particuliers": 3, "Coaching sportif": 2,
    "Evenementiel": 2, "Beaute et bien-etre": 2, "Decoration interieure": 2,
    "Photographie": 1, "Traiteur": 1, "Transport": 2,
}
TECH_SUBTYPE_WEIGHT = {
    "Installation": 3, "Reparation": 4, "Entretien": 3, "Renovation": 2,
    "Depannage urgent": 2, "Devis et conseil": 4, "Mise aux normes": 1,
    "Remplacement": 2, "Diagnostic": 1, "Petits travaux": 2,
}
SERVICE_SUBTYPE_WEIGHT = {
    "Devis et conseil": 4, "Prestation standard": 3, "Prestation premium": 1,
    "Intervention ponctuelle": 2, "Formule decouverte": 2, "Abonnement regulier": 1,
    "Prestation express": 2, "Intervention a domicile": 2, "Organisation complete": 1,
    "Accompagnement personnalise": 2,
}


def class_sample_count(rng, domain_weight, subtype_weight):
    base = domain_weight * subtype_weight
    jitter = rng.uniform(0.6, 1.4)
    return max(4, min(200, int(base * 4 * jitter)))


def make_description(rng, verb_phrase, allow_room):
    opening = rng.choice(OPENINGS).format(verb=verb_phrase)
    sentences = [opening]
    if allow_room and rng.random() < 0.6:
        room = rng.choice(ROOMS)
        if room:
            sentences.append(f"C'est {room}.")
    if rng.random() < 0.5:
        urgency = rng.choice(URGENCY)
        if urgency:
            sentences.append(f"Si possible {urgency}.")
    if rng.random() < 0.6:
        closing = rng.choice(CLOSINGS)
        if closing:
            sentences.append(closing)
    return " ".join(sentences)


def generate_rows(rng):
    rows = []
    tech_domains = {name: objs for name, objs in TECH_DOMAINS}
    service_domains = {name: objs for name, objs in SERVICE_DOMAINS}

    for taxo_row in build_taxonomy():
        domain = taxo_row["domain"]
        subtype = taxo_row["subtype"]
        obj = taxo_row["object"]
        service = taxo_row["service"]
        sous_service = taxo_row["sous_service"]
        is_tech = domain in tech_domains

        verb_template = (VERB_TECH if is_tech else VERB_SERVICE)[subtype]
        verb_phrase = verb_template.format(obj=OBJECT_PHRASES[obj])

        if is_tech:
            n = class_sample_count(rng, TECH_DOMAIN_WEIGHT[domain], TECH_SUBTYPE_WEIGHT[subtype])
        else:
            n = class_sample_count(rng, SERVICE_DOMAIN_WEIGHT[domain], SERVICE_SUBTYPE_WEIGHT[subtype])

        for _ in range(n):
            description = make_description(rng, verb_phrase, allow_room=is_tech)
            rows.append({"description": description, "service": service, "sous_service": sous_service})
    return rows


def main():
    rng = random.Random(SEED)
    rows = generate_rows(rng)
    rng.shuffle(rows)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "raw_descriptions.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["description", "service", "sous_service"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"lignes generees: {len(rows)}")
    print(f"services distincts: {len({r['service'] for r in rows})}")
    print(f"sous_services distincts: {len({r['sous_service'] for r in rows})}")
    print(f"ecrit: {out_path}")


if __name__ == "__main__":
    main()
