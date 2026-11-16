import pytest

from amoeba.items import Item


def test_item_is_a_record():
    it = Item(name="arm3.j1.angle", kind="sense", low=-3.14, high=3.14, unit="rad")
    assert it.name == "arm3.j1.angle"
    assert it.kind == "sense"
    assert it.low == -3.14 and it.high == 3.14
    assert it.unit == "rad"


def test_item_rejects_bad_kind():
    with pytest.raises(ValueError):
        Item(name="x", kind="motor", low=0.0, high=1.0, unit="rad")


def test_item_rejects_bad_range():
    with pytest.raises(ValueError):
        Item(name="x", kind="sense", low=1.0, high=1.0, unit="rad")
    with pytest.raises(ValueError):
        Item(name="x", kind="sense", low=2.0, high=1.0, unit="rad")


def test_item_rejects_empty_name():
    with pytest.raises(ValueError):
        Item(name="", kind="sense", low=0.0, high=1.0, unit="rad")


def test_preset_constants_are_the_brief_numbers():
    from amoeba.feelings import PRESETS

    lean, rich = PRESETS["lean"], PRESETS["rich"]
    print(
        f"lean={lean.basal_rate!r},{lean.effort_rate!r},{lean.intake_rate!r},"
        f"{lean.strength_floor!r},{lean.energy_set_point!r}"
    )
    print(
        f"rich={rich.basal_rate!r},{rich.effort_rate!r},{rich.intake_rate!r},"
        f"{rich.strength_floor!r},{rich.energy_set_point!r}"
    )
    assert lean.basal_rate == 1 / 600 and lean.effort_rate == 1 / 600
    assert lean.intake_rate == 0.05 and lean.strength_floor == 0.3
    assert lean.energy_set_point == 0.8
    assert rich.basal_rate == 1 / 1800 and rich.effort_rate == 1 / 1800
    assert rich.intake_rate == 0.15 and rich.strength_floor == 0.5
    assert rich.energy_set_point == 0.8
