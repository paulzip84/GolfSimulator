from pga_sim.service import (
    _extract_live_score_snapshot,
    _extract_rows,
    _probability_from_keys,
    _score_to_par_from_value,
)


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


def test_score_to_par_parser_handles_even_and_signed_values() -> None:
    assert _score_to_par_from_value("E") == 0.0
    assert _score_to_par_from_value("+3") == 3.0
    assert _score_to_par_from_value("-4") == -4.0
    assert _score_to_par_from_value("3 UNDER") == -3.0


def test_extract_live_score_snapshot_parses_nested_scorecard_payload() -> None:
    row = {
        "player_id": "77",
        "player_name": "Golfer",
        "position": "T5",
        "score_to_par": "-6",
        "status": {"thru": "F"},
        "today": "+1",
        "scorecard": {
            "round_scores": [68, 70, 69],
            "holes": {"hole_1": 4, "hole_2": 5, "hole_3": 3, "hole_4": 4},
        },
    }

    snapshot = _extract_live_score_snapshot(row)
    assert snapshot.current_position == "T5"
    assert snapshot.current_score_to_par == -6.0
    assert snapshot.current_thru == "F"
    assert snapshot.today_score_to_par == 1.0
    assert snapshot.round_scores == [68.0, 70.0, 69.0]
    assert snapshot.hole_scores == [4, 5, 3, 4]
