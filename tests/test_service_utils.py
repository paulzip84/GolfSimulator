from pga_sim.service import (
    SimulationService,
    _derive_total_to_par_from_round_data,
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


def test_extract_live_score_snapshot_derives_total_from_round_to_par_data() -> None:
    row = {
        "player_id": "77",
        "player_name": "Golfer",
        "position": "T5",
        # Intentionally stale/missing total score to par.
        "score_to_par": "-2",
        "thru": "9",
        "today": "+1",
        # These are to-par-by-round values.
        "round_scores": [-3, -2],
    }

    snapshot = _extract_live_score_snapshot(row)
    # Completed rounds sum to -5, plus in-progress today (+1) => -4.
    assert snapshot.current_score_to_par == -4.0


def test_extract_live_score_snapshot_ignores_nested_hole_to_par_for_total() -> None:
    row = {
        "player_id": "88",
        "player_name": "Nested Totals",
        "thru": "F",
        "score_to_par": "-7",
        "scorecard": {
            "holes": [
                {"hole": 1, "to_par": "E"},
                {"hole": 2, "to_par": "+1"},
            ]
        },
    }
    snapshot = _extract_live_score_snapshot(row)
    assert snapshot.current_score_to_par == -7.0


def test_extract_live_score_snapshot_parses_round_score_strings_for_total() -> None:
    row = {
        "player_id": "91",
        "player_name": "String Rounds",
        "thru": "7",
        "today": "+2",
        "round_scores": ["-3", "E"],
    }
    snapshot = _extract_live_score_snapshot(row)
    # Completed rounds: -3 + 0, in-progress today: +2 => total -1.
    assert snapshot.current_score_to_par == -1.0


def test_extract_live_score_snapshot_derives_total_from_gross_rounds_when_complete() -> None:
    row = {
        "player_id": "99",
        "player_name": "Gross Rounds",
        "thru": "18",
        "today": "-6",
        # Gross strokes by round.
        "round_scores": [74, 68, 66, 65],
    }
    snapshot = _extract_live_score_snapshot(row)
    # Infer par 71 from final round: 65 - (-6) = 71
    # Total to par = (74+68+66+65) - (71*4) = 273 - 284 = -11
    assert snapshot.current_score_to_par == -11.0


def test_derive_total_from_gross_rounds_in_progress_with_inferred_par() -> None:
    derived = _derive_total_to_par_from_round_data(
        round_scores=[70, 71, 69],
        today_score_to_par=-3.0,
        current_thru="12",
        inferred_round_par=72.0,
    )
    assert derived == -9.0


def test_merge_player_records_derives_in_progress_totals_from_cohort_par() -> None:
    service = SimulationService(object())  # type: ignore[arg-type]
    field_rows = [
        {
            "player_id": "1",
            "player_name": "Finished Player",
            "score_to_par": "-10",
            "thru": "18",
            "today": "-4",
            "round_scores": [70, 71, 69, 68],
        },
        {
            "player_id": "2",
            "player_name": "In Progress Player",
            "thru": "12",
            "today": "-3",
            "round_scores": [70, 71, 69],
        },
    ]
    live_rows = [
        {
            "player_id": "2",
            "player_name": "In Progress Player",
            "score_to_par": "E",
            "thru": "12",
            "today": "-3",
        }
    ]

    records = service._merge_player_records(field_rows, [], [], live_rows)
    by_name = {record.player_name: record for record in records}
    assert by_name["Finished Player"].current_score_to_par == -10.0
    # Sum of completed gross rounds = 210. Inferred event par = 72 from finished player.
    # Completed-round to-par = 210 - (72*3) = -6; add today's -3 => -9.
    assert by_name["In Progress Player"].current_score_to_par == -9.0
