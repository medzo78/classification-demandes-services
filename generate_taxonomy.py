"""
Genere la taxonomie du projet : 250 "services" et 750 "sous-services"
(3 sous-services par service), pour une plateforme de mise en relation
particuliers / entrepreneurs (demandes de travaux et de prestations a domicile).

Deux familles de domaines :
  - domaines "metiers techniques" (plomberie, electricite, peinture, ...)
    combines avec des types d'intervention (installation, reparation, ...)
  - domaines "prestations de service" (cours particuliers, evenementiel, ...)
    combines avec des formules de prestation (devis, formule decouverte, ...)

Usage:
    python generate_taxonomy.py
Produit:
    data/taxonomy.csv  (service_id, service, domain, subtype, object,
                         sous_service_id, sous_service)
"""
import csv
import os

# (domaine, [3 objets/taches types du domaine])
TECH_DOMAINS = [
    ("Plomberie", ["sanitaires", "chauffe-eau", "tuyauterie"]),
    ("Electricite", ["tableau electrique", "prises et interrupteurs", "eclairage"]),
    ("Peinture", ["murs interieurs", "facade exterieure", "plafonds"]),
    ("Menuiserie", ["portes", "fenetres", "placards sur mesure"]),
    ("Jardinage", ["pelouse", "haies et arbustes", "arrosage automatique"]),
    ("Nettoyage", ["menage courant", "vitres", "fin de chantier"]),
    ("Climatisation", ["climatiseur mural", "pompe a chaleur", "ventilation"]),
    ("Chauffage", ["chaudiere", "radiateurs", "plancher chauffant"]),
    ("Toiture", ["tuiles", "gouttieres", "etancheite"]),
    ("Maconnerie", ["mur porteur", "dalle beton", "ouverture de mur"]),
    ("Carrelage", ["sol interieur", "faience salle de bain", "terrasse exterieure"]),
    ("Serrurerie", ["serrure", "porte blindee", "coffre-fort"]),
    ("Reparation electromenager", ["lave-linge", "refrigerateur", "four et plaques"]),
    ("Renovation energetique", ["isolation combles", "isolation murs", "fenetres double vitrage"]),
    ("Bricolage general", ["montage de meubles", "pose d'etageres", "petites reparations"]),
]

TECH_SUBTYPES = [
    "Installation",
    "Reparation",
    "Entretien",
    "Renovation",
    "Depannage urgent",
    "Devis et conseil",
    "Mise aux normes",
    "Remplacement",
    "Diagnostic",
    "Petits travaux",
]

SERVICE_DOMAINS = [
    ("Demenagement", ["cartons et emballage", "transport de meubles", "garde-meuble"]),
    ("Informatique", ["ordinateur", "reseau wifi", "sauvegarde de donnees"]),
    ("Cours particuliers", ["mathematiques", "francais", "langues etrangeres"]),
    ("Coaching sportif", ["remise en forme", "musculation", "preparation physique"]),
    ("Evenementiel", ["mariage", "anniversaire", "soiree d'entreprise"]),
    ("Beaute et bien-etre", ["coiffure a domicile", "massage", "manucure"]),
    ("Decoration interieure", ["papier peint", "amenagement de piece", "conseil deco"]),
    ("Photographie", ["portrait", "evenement", "immobilier"]),
    ("Traiteur", ["buffet", "repas assis", "cocktail"]),
    ("Transport", ["livraison de colis", "transport de personnes", "location avec chauffeur"]),
]

SERVICE_SUBTYPES = [
    "Devis et conseil",
    "Prestation standard",
    "Prestation premium",
    "Intervention ponctuelle",
    "Formule decouverte",
    "Abonnement regulier",
    "Prestation express",
    "Intervention a domicile",
    "Organisation complete",
    "Accompagnement personnalise",
]


def build_taxonomy():
    rows = []
    service_id = 1
    sous_service_id = 1
    for domains, subtypes in [(TECH_DOMAINS, TECH_SUBTYPES), (SERVICE_DOMAINS, SERVICE_SUBTYPES)]:
        for domain, objects in domains:
            for subtype in subtypes:
                service_name = f"{domain} - {subtype}"
                for obj in objects:
                    sous_service_name = f"{subtype} - {obj}"
                    rows.append(
                        {
                            "service_id": f"SRV{service_id:03d}",
                            "service": service_name,
                            "domain": domain,
                            "subtype": subtype,
                            "object": obj,
                            "sous_service_id": f"SUB{sous_service_id:03d}",
                            "sous_service": sous_service_name,
                        }
                    )
                    sous_service_id += 1
                service_id += 1
    return rows


def main():
    rows = build_taxonomy()
    n_services = len({r["service_id"] for r in rows})
    n_sous = len({r["sous_service_id"] for r in rows})
    print(f"services: {n_services}, sous_services: {n_sous}")
    assert n_services == 250, n_services
    assert n_sous == 750, n_sous

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "taxonomy.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "service_id",
                "service",
                "domain",
                "subtype",
                "object",
                "sous_service_id",
                "sous_service",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"ecrit: {out_path}")


if __name__ == "__main__":
    main()
