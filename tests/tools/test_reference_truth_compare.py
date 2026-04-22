from __future__ import annotations

from app.tools.reference_truth_compare import compare_truth_to_parsed


def test_compare_truth_to_parsed_detects_turn_and_guess_mismatches() -> None:
    truth = {
        "source": "test",
        "games": {
            "game1": {
                "winner_team": "blue",
                "turns": [
                    {
                        "turn_index": 0,
                        "team_color": "red",
                        "spymaster_name": "arker",
                        "clue_text": "TEETH",
                        "clue_count": "1",
                        "events": [
                            {
                                "event_type": "guess",
                                "word": "DENTIST",
                                "card_color": "red",
                                "result": "correct",
                                "selector_name": "fembluca",
                            }
                        ],
                    }
                ],
            }
        },
    }
    parsed = {
        "game1": {
            "winner_team": "red",
            "turns": [
                {
                    "team_color": "red",
                    "spymaster_name": "arker",
                    "clue_text": "TEETH",
                    "clue_count": "2",
                    "guesses": [
                        {
                            "word": "STREAM",
                            "card_color": "red",
                            "result": "enemy",
                            "selector_name": "ty",
                        }
                    ],
                }
            ],
        }
    }

    report = compare_truth_to_parsed(truth_payload=truth, parsed_payload=parsed)
    mismatches = report["games"]["game1"]["mismatches"]

    assert report["total_mismatches"] >= 4
    assert any(item["kind"] == "winner_team" for item in mismatches)
    assert any(item["kind"] == "turn_field" and item["field"] == "clue_count" for item in mismatches)
    assert any(item["kind"] == "guess_field" and item["field"] == "word" for item in mismatches)
    assert any(item["kind"] == "guess_field" and item["field"] == "selector_name" for item in mismatches)


def test_compare_truth_to_parsed_treats_turn_end_markers_as_advisory() -> None:
    truth = {
        "source": "test",
        "games": {
            "game1": {
                "winner_team": "blue",
                "turns": [
                    {
                        "turn_index": 0,
                        "team_color": "blue",
                        "spymaster_name": "rush",
                        "clue_text": "RIFF",
                        "clue_count": "2",
                        "events": [
                            {
                                "event_type": "turn_end_marker",
                                "marker": "green_checkmark",
                                "selector_name": "Drew",
                            }
                        ],
                    }
                ],
            }
        },
    }
    parsed = {
        "game1": {
            "winner_team": "blue",
            "turns": [
                {
                    "team_color": "blue",
                    "spymaster_name": "rush",
                    "clue_text": "RIFF",
                    "clue_count": "2",
                    "guesses": [],
                }
            ],
        }
    }

    report = compare_truth_to_parsed(truth_payload=truth, parsed_payload=parsed)

    assert report["total_mismatches"] == 0
    assert report["games"]["game1"]["advisory"] == [
        {
            "kind": "turn_end_markers_present_in_truth",
            "turn_index": 0,
            "count": 1,
        }
    ]
