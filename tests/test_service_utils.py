from pga_sim.service import _extract_rows, _probability_from_keys


def test_extract_rows_prefers_player_payload() -> None:
    payload = {
        "meta": {"event_name": "Sample"},
        "predictions": {
            "players": [
                {"player_name": "A", "win": 1.2},
                {"player_name": "B", "win": 0.7},
            ]
        },
        "other_rows": [{"foo": "bar"}],
    }
    rows = _extract_rows(payload, ("pred", "player"))
    assert len(rows) == 2
    assert rows[0]["player_name"] == "A"


def test_probability_parses_percentages_and_decimals() -> None:
    row_pct = {"top_10": 12.5}
    row_dec = {"top_10": 0.125}
    assert _probability_from_keys(row_pct, ("top_10",)) == 0.125
    assert _probability_from_keys(row_dec, ("top_10",)) == 0.125

