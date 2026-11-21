"""Usage: uv run python scripts/seed_data.py  -> writes data/sql/*.csv"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sql"
TODAY = date(2026, 11, 20)
rng = random.Random(42)

MITARBEITER = [
    "Peter Gruber",
    "Elisabeth Rainer",
    "Klaus Obermair",
    "Daniela Pichler",
    "Stefan Hofbauer",
    "Hannes Brandstetter",
    "Sabine Wimmer",
    "Markus Leitner",
]
TAETIGKEITEN = [
    "Rohrmontage",
    "Schweißen",
    "Vorfertigung",
    "Druckprüfung",
    "Dokumentation",
    "Projektleitung",
    "Bauleitung",
]


def write_csv(name: str, rows: list[dict], fieldnames: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    facts = yaml.safe_load((ROOT / "data" / "facts.yaml").read_text(encoding="utf-8"))
    kunden = facts["kunden"]
    projekte = facts["projekte"]
    offene_punkte = [
        {
            **item,
            "faellig_am": "" if item["faellig_am"] is None else item["faellig_am"],
        }
        for item in facts["offene_punkte"]
    ]

    write_csv("kunden.csv", kunden, ["id", "name", "ort"])
    write_csv(
        "projekte.csv",
        projekte,
        [
            "id",
            "name",
            "kunde_id",
            "status",
            "projektleiter",
            "start_datum",
            "ende_geplant",
            "budget_eur",
        ],
    )
    write_csv(
        "offene_punkte.csv",
        offene_punkte,
        [
            "id",
            "projekt_id",
            "beschreibung",
            "prioritaet",
            "status",
            "faellig_am",
            "verantwortlich",
        ],
    )

    active_projects = []
    for project in projekte:
        if project["status"] not in {"In Ausführung", "Abgeschlossen"}:
            continue
        start = date.fromisoformat(project["start_datum"])
        end = min(date.fromisoformat(project["ende_geplant"]), TODAY)
        workdays = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                workdays.append(current)
            current += timedelta(days=1)
        if workdays:
            active_projects.append((project, workdays))

    hours_options = [hour / 2 for hour in range(8, 21)]
    stunden = []
    for row_id in range(1, 251):
        project, workdays = rng.choice(active_projects)
        stunden.append(
            {
                "id": row_id,
                "projekt_id": project["id"],
                "mitarbeiter": rng.choice(MITARBEITER),
                "datum": rng.choice(workdays).isoformat(),
                "stunden": rng.choice(hours_options),
                "taetigkeit": rng.choice(TAETIGKEITEN),
            }
        )
    write_csv(
        "stunden.csv",
        stunden,
        ["id", "projekt_id", "mitarbeiter", "datum", "stunden", "taetigkeit"],
    )

    lieferanten = [
        "Flanschtechnik Hofer GmbH",
        "Rohrhandel Mayr",
        "Rohrwerk Steyrtal GmbH",
        "Armaturen Koller GmbH",
    ]
    material = []
    durchmesser = [50, 80, 100, 150, 200, 300]
    for row_id in range(1, 31):
        project = projekte[(row_id - 1) % len(projekte)]
        material_typ = (row_id - 1) % 4
        dn = rng.choice(durchmesser)
        if material_typ == 0:
            bezeichnung = f"Edelstahlrohr 1.4571 DN {dn}"
            menge = rng.choice([6, 12, 24, 48, 60, 100])
            einheit = "m"
            einzelkosten = rng.uniform(35, 180)
        elif material_typ == 1:
            bezeichnung = f"Edelstahlflansch 1.4571 DN {dn}"
            menge = rng.randint(4, 40)
            einheit = "Stk"
            einzelkosten = rng.uniform(35, 350)
        elif material_typ == 2:
            bezeichnung = f"Flanschdichtung DN {dn}"
            menge = rng.randint(4, 40)
            einheit = "Stk"
            einzelkosten = rng.uniform(5, 45)
        else:
            bezeichnung = f"Edelstahl-Absperrarmatur DN {dn}"
            menge = rng.randint(2, 12)
            einheit = "Stk"
            einzelkosten = rng.uniform(350, 1800)

        if project["id"] == "P-2041":
            lieferant = rng.choice(lieferanten[:2])
        elif project["id"] == "P-2043" and material_typ == 0:
            lieferant = "Rohrwerk Steyrtal GmbH"
        else:
            lieferant = rng.choice(lieferanten)

        material.append(
            {
                "id": row_id,
                "projekt_id": project["id"],
                "bezeichnung": bezeichnung,
                "menge": menge,
                "einheit": einheit,
                "kosten_eur": round(menge * einzelkosten, 2),
                "lieferant": lieferant,
            }
        )
    write_csv(
        "material.csv",
        material,
        [
            "id",
            "projekt_id",
            "bezeichnung",
            "menge",
            "einheit",
            "kosten_eur",
            "lieferant",
        ],
    )


if __name__ == "__main__":
    main()
