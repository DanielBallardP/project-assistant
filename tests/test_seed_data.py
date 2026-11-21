import csv
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from scripts import seed_data


@pytest.fixture
def seeded_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, list[dict[str, str]]]:
    monkeypatch.setattr(seed_data, "OUT", tmp_path)
    seed_data.rng.seed(42)
    seed_data.main()

    tables = {}
    for name in ("projekte", "stunden", "material", "offene_punkte"):
        with (tmp_path / f"{name}.csv").open(encoding="utf-8", newline="") as file:
            tables[name] = list(csv.DictReader(file))
    return tables


def test_every_project_id_exists_in_projekte_csv(seeded_tables):
    project_ids = {row["id"] for row in seeded_tables["projekte"]}

    for table_name in ("stunden", "material", "offene_punkte"):
        assert all(
            row["projekt_id"] in project_ids for row in seeded_tables[table_name]
        )


def test_no_worklog_entry_exceeds_12_hours(seeded_tables):
    assert all(Decimal(row["stunden"]) <= 12 for row in seeded_tables["stunden"])


def test_hours_are_not_booked_before_project_start(seeded_tables):
    projects = {row["id"]: row for row in seeded_tables["projekte"]}

    assert all(
        date.fromisoformat(row["datum"])
        >= date.fromisoformat(projects[row["projekt_id"]]["start_datum"])
        for row in seeded_tables["stunden"]
    )


def test_material_cost_per_project_stays_below_budget(seeded_tables):
    project_budgets = {
        row["id"]: Decimal(row["budget_eur"]) for row in seeded_tables["projekte"]
    }
    material_costs: defaultdict[str, Decimal] = defaultdict(Decimal)
    for row in seeded_tables["material"]:
        material_costs[row["projekt_id"]] += Decimal(row["kosten_eur"])

    assert all(
        cost < project_budgets[project_id]
        for project_id, cost in material_costs.items()
    )
