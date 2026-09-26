"""Check v12's frozen rule universe without private case facts."""

from itertools import combinations, permutations, product

from jev_arena.authored_v12_pilot import MISSING, RULES, checked


def test_complete_rule_universes_and_missing_evidence() -> None:
    shelves = ("Iris", "Juniper", "Lotus")
    destinations = ("North", "East", "West")
    for order in permutations(destinations):
        assignment = dict(zip(shelves, order))
        for entry in shelves:
            assert (
                checked(
                    RULES["provenance-hop"], {"entry": entry, "assignment": assignment}
                )
                in destinations
            )
    for flag, pulse in product("ABC", (1, 2, 3)):
        assert checked(RULES["dual-index-code"], {"flag": flag, "pulse": pulse}) in {
            "Cedar",
            "Delta",
            "Echo",
        }
    names = ("Aster", "Birch", "Clover")
    subsets = [list(part) for count in range(4) for part in combinations(names, count)]
    for first, second in product(subsets, repeat=2):
        assert checked(
            RULES["exclusive-register"],
            {"first_register": first, "second_register": second},
        ) in {*names, "hold"}
    for lineup, swap in product(
        permutations(("Oak", "Pine", "Yew")), ((1, 2), (1, 3), (2, 3))
    ):
        assert checked(
            RULES["single-swap"], {"lineup": list(lineup), "swap": list(swap)}
        ) in {"Oak", "Pine", "Yew"}
    for batch, seal in product(range(7), repeat=2):
        assert (
            type(
                checked(
                    RULES["ceramic-checksum"], {"batch_code": batch, "seal_code": seal}
                )
            )
            is bool
        )
    for burst, reply in product(range(60), repeat=2):
        assert (
            type(
                checked(
                    RULES["burst-handshake"],
                    {"burst_second": burst, "reply_second": reply},
                )
            )
            is bool
        )
    for reference, measured in product(range(-10, 11), repeat=2):
        assert checked(
            RULES["pitch-drift"],
            {"reference_cents": reference, "measured_cents": measured},
        ) in range(5)
    for provenance, condition in product("ABC", ("stable", "fragile", "failed")):
        assert checked(
            RULES["curation-matrix"], {"provenance": provenance, "condition": condition}
        ) in range(5)
    for rule in RULES.values():
        assert checked(rule, dict.fromkeys(rule.fields)) == MISSING
