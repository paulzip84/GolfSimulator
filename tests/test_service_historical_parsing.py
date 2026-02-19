from pga_sim.service import (
    _extract_player_metric_observations,
    _historical_event_descriptors,
    _parse_season_value,
)


def test_parse_season_value_handles_multiple_formats() -> None:
    assert _parse_season_value("2025") == 2025
    assert _parse_season_value("2024-2025") == 2025
    assert _parse_season_value("24-25") == 2025
    assert _parse_season_value("Season 2023/24") == 2024


def test_historical_event_descriptors_filters_by_requested_year() -> None:
    payload = {
        "event_list": [
            {"event_id": "A", "year": 2024},
            {"event_id": "B", "year": 2025},
            {"event_id": "C", "season": "2025"},
            {"event_id": "D", "start_date": "2025-05-02"},
        ]
    }
    descriptors = _historical_event_descriptors(payload, season=2025)
    assert ("A", 2024) not in descriptors
    assert ("B", 2025) in descriptors
    assert ("C", 2025) in descriptors
    assert ("D", 2025) in descriptors


def test_extract_player_metric_observations_from_round_score_arrays() -> None:
    payload = {
        "event_id": "X1",
        "players": [
            {"player_id": "p1", "player_name": "Player 1", "round_scores": [68, 70, 69, 71]},
            {"player_id": "p2", "player_name": "Player 2", "round_scores": [72, 74, 71, 73]},
        ],
    }
    observations = _extract_player_metric_observations(payload)
    assert len(observations) == 8

    by_player = {}
    for key, _id, name, metric in observations:
        by_player.setdefault((key, _id, name), []).append(metric)
    p1_metrics = next(v for k, v in by_player.items() if k[1] == "p1")
    p2_metrics = next(v for k, v in by_player.items() if k[1] == "p2")

    # Lower scores should map to positive relative metrics.
    assert sum(p1_metrics) > 0
    assert sum(p2_metrics) < 0


def test_extract_player_metric_observations_from_event_stats_payload() -> None:
    payload = {
        "event_id": "14",
        "year": 2025,
        "event_stats": [
            {
                "dg_id": 10091,
                "player_name": "McIlroy, Rory",
                "dg_points": 28.0,
                "fec_points": 750,
                "earnings": 4200000,
                "fin_text": "1",
            },
            {
                "dg_id": 6093,
                "player_name": "Rose, Justin",
                "dg_points": 13.8649,
                "fec_points": 500,
                "earnings": 2268000,
                "fin_text": "2",
            },
        ],
    }
    observations = _extract_player_metric_observations(payload)
    assert len(observations) == 2
    by_name = {name: metric for _, _, name, metric in observations}
    assert "McIlroy, Rory" in by_name
    assert "Rose, Justin" in by_name
    assert by_name["McIlroy, Rory"] > by_name["Rose, Justin"]
