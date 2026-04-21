from app.core.models import PlayerRole, PlayerRosterEntry, TeamColor
from app.parsers.selectors.resolution import (
    _resolve_name_from_initial_hint,
    _resolve_name_from_text,
    merge_selector_results_into_analysis,
    resolve_selector_fields_inplace,
)


def _roster() -> list[PlayerRosterEntry]:
    return [
        PlayerRosterEntry(player_name="ty", display_name="ty", team_color=TeamColor.BLUE, role=PlayerRole.OPERATIVE),
        PlayerRosterEntry(player_name="drew", display_name="Drew", team_color=TeamColor.BLUE, role=PlayerRole.OPERATIVE),
        PlayerRosterEntry(player_name="rush", display_name="rush", team_color=TeamColor.BLUE, role=PlayerRole.SPYMASTER),
        PlayerRosterEntry(player_name="cherry", display_name="Cherry", team_color=TeamColor.RED, role=PlayerRole.OPERATIVE),
        PlayerRosterEntry(player_name="fembluca", display_name="fembluca", team_color=TeamColor.RED, role=PlayerRole.OPERATIVE),
        PlayerRosterEntry(player_name="arker", display_name="arker", team_color=TeamColor.RED, role=PlayerRole.SPYMASTER),
    ]


def test_attach_selector_artifacts_and_resolve_selector_name() -> None:
    analysis = {
        "roster": [player.model_dump(mode="json") for player in _roster()],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "guesses": [
                    {"word": "ATLANTIS", "timestamp_sec": 10.0},
                ],
            }
        ],
    }
    selector_results = {
        "guesses": [
            {
                "turn_index": 0,
                "guess_index": 0,
                "crop_path": "crop.png",
                "crop_x4_path": "crop-x4.png",
                "candidate_scores": [
                    {"display_name": "tv", "composite_score": 0.28},
                    {"display_name": "Drew", "composite_score": 0.18},
                ],
            }
        ]
    }

    merge_selector_results_into_analysis(analysis, selector_results)
    resolve_selector_fields_inplace(analysis)

    guess = analysis["turns"][0]["guesses"][0]
    assert guess["selector_crop_path"] == "crop.png"
    assert guess["selector_name"] == "ty"
    assert guess["selector_source"] == "selector_candidates"
    assert guess["selector_confidence"] >= 0.75


def test_merge_selector_results_clears_stale_weak_selector_fields() -> None:
    analysis = {
        "roster": [player.model_dump(mode="json") for player in _roster()],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "guesses": [
                    {
                        "word": "BACON",
                        "timestamp_sec": 10.0,
                        "selector_name": "Drew",
                        "selector_source": "turn_continuity",
                        "selector_confidence": 0.66,
                        "resolved_selector_candidates": [{"display_name": "Drew", "composite_score": 0.31}],
                    },
                ],
            }
        ],
    }
    selector_results = {
        "guesses": [
            {
                "turn_index": 0,
                "guess_index": 0,
                "crop_path": "crop.png",
                "crop_x4_path": "crop-x4.png",
                "candidate_scores": [
                    {"display_name": "Cherry", "composite_score": 0.41},
                    {"display_name": "Drew", "composite_score": 0.39},
                ],
            }
        ]
    }

    merge_selector_results_into_analysis(analysis, selector_results)

    guess = analysis["turns"][0]["guesses"][0]
    assert guess.get("selector_name") is None
    assert guess.get("selector_source") is None
    assert guess.get("resolved_selector_candidates") is None
    assert guess["selector_candidates"][0]["display_name"] == "Cherry"


def test_continuity_fallback_only_uses_high_confidence_previous_guess() -> None:
    analysis = {
        "roster": [player.model_dump(mode="json") for player in _roster()],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "red",
                "guesses": [
                    {
                        "word": "DENTIST",
                        "timestamp_sec": 10.0,
                        "selector_candidates": [
                            {"display_name": "Cherry", "composite_score": 0.31},
                        ],
                    },
                    {
                        "word": "POP",
                        "timestamp_sec": 12.0,
                        "selector_candidates": [],
                    },
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    first_guess, second_guess = analysis["turns"][0]["guesses"]
    assert first_guess["selector_name"] == "Cherry"
    assert second_guess["selector_name"] == "Cherry"
    assert second_guess["selector_source"] == "turn_continuity"


def test_short_selector_ocr_requires_clear_best_match() -> None:
    assert _resolve_name_from_text("Deny", ["Drew", "Cherry", "luke"])[0] is None
    assert _resolve_name_from_text("Dery", ["Drew", "Cherry", "luke"])[0] == "Cherry"


def test_initial_hint_rejects_single_letter_and_length_mismatch_noise() -> None:
    assert _resolve_name_from_initial_hint("A", ["ty", "Drew", "arker"])[0] is None
    assert _resolve_name_from_initial_hint("R", ["near", "revel", "shadowsn"])[0] is None
    assert _resolve_name_from_initial_hint("SISN", ["near", "revel", "shadowsn"])[0] is None


def test_initial_hint_keeps_short_valid_selector_reads() -> None:
    assert _resolve_name_from_initial_hint("Tt", ["ty", "Drew", "arker"])[0] == "ty"
    assert _resolve_name_from_initial_hint("Der", ["ty", "Drew", "arker"])[0] == "Drew"


def test_guess_player_promotes_selector_for_two_operative_team() -> None:
    analysis = {
        "roster": [
            {"display_name": "near", "team_color": "red", "role": "operative"},
            {"display_name": "revel", "team_color": "red", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "red",
                "guesses": [
                    {"word": "GLOVE", "timestamp_sec": 10.0, "player_name": "near"},
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    guess = analysis["turns"][0]["guesses"][0]
    assert guess["selector_name"] == "near"
    assert guess["selector_source"] == "guess_player"


def test_team_dominance_backfills_ambiguous_same_team_guesses() -> None:
    analysis = {
        "roster": [
            {"display_name": "near", "team_color": "red", "role": "operative"},
            {"display_name": "revel", "team_color": "red", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "red",
                "guesses": [
                    {"word": "SUIT", "timestamp_sec": 10.0, "selector_name": "near", "selector_confidence": 0.9},
                ],
            },
            {
                "turn_index": 1,
                "team_color": "red",
                "guesses": [
                    {"word": "HEART", "timestamp_sec": 20.0, "selector_name": "near", "selector_confidence": 0.88},
                ],
            },
            {
                "turn_index": 2,
                "team_color": "red",
                "guesses": [
                    {
                        "word": "CROWN",
                        "timestamp_sec": 30.0,
                        "resolved_selector_candidates": [
                            {"display_name": "revel"},
                            {"display_name": "near"},
                        ],
                    },
                    {"word": "STREET", "timestamp_sec": 35.0, "player_name": "unknown"},
                ],
            },
        ],
    }

    resolve_selector_fields_inplace(analysis)

    first_guess, second_guess = analysis["turns"][2]["guesses"]
    assert first_guess["selector_name"] == "near"
    assert first_guess["selector_source"] == "team_dominance"
    assert second_guess["selector_name"] == "near"
    assert second_guess["selector_source"] == "team_dominance"


def test_turn_continuity_can_use_close_runner_up_candidate() -> None:
    analysis = {
        "roster": [
            {"display_name": "Drew", "team_color": "blue", "role": "operative"},
            {"display_name": "Cherry", "team_color": "blue", "role": "operative"},
            {"display_name": "luke", "team_color": "blue", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "guesses": [
                    {"word": "PENTAGON", "timestamp_sec": 10.0, "selector_name": "Cherry", "selector_confidence": 0.9},
                    {
                        "word": "PAGE",
                        "timestamp_sec": 12.0,
                        "resolved_selector_candidates": [
                            {"display_name": "Drew", "composite_score": 0.2785},
                            {"display_name": "Cherry", "composite_score": 0.2546},
                            {"display_name": "luke", "composite_score": 0.2496},
                        ],
                    },
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    guess = analysis["turns"][0]["guesses"][1]
    assert guess["selector_name"] == "Cherry"
    assert guess["selector_source"] == "turn_continuity"


def test_turn_continuity_does_not_fill_blank_guess_without_selector_evidence() -> None:
    analysis = {
        "roster": [
            {"display_name": "Drew", "team_color": "blue", "role": "operative"},
            {"display_name": "Cherry", "team_color": "blue", "role": "operative"},
            {"display_name": "luke", "team_color": "blue", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "guesses": [
                    {
                        "word": "DRAGON",
                        "timestamp_sec": 10.0,
                        "selector_name": "Drew",
                        "selector_source": "selector_crop",
                        "selector_confidence": 0.9,
                    },
                    {
                        "word": "BACON",
                        "timestamp_sec": 12.0,
                    },
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    guess = analysis["turns"][0]["guesses"][1]
    assert guess.get("selector_name") is None


def test_resolve_selector_fields_reruns_weak_existing_selector() -> None:
    analysis = {
        "roster": [
            {"display_name": "Drew", "team_color": "blue", "role": "operative"},
            {"display_name": "Cherry", "team_color": "blue", "role": "operative"},
            {"display_name": "luke", "team_color": "blue", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "guesses": [
                    {
                        "word": "BACON",
                        "timestamp_sec": 10.0,
                        "selector_name": "Drew",
                        "selector_source": "turn_continuity",
                        "selector_confidence": 0.66,
                        "selector_candidates": [
                            {"display_name": "Cherry", "composite_score": 0.41},
                            {"display_name": "Drew", "composite_score": 0.39},
                        ],
                    },
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    guess = analysis["turns"][0]["guesses"][0]
    assert guess["selector_name"] == "Cherry"
    assert guess["selector_source"] == "selector_candidates"


def test_strong_same_turn_selector_overrides_weak_candidate_pick() -> None:
    analysis = {
        "roster": [
            {"display_name": "Drew", "team_color": "blue", "role": "operative"},
            {"display_name": "Cherry", "team_color": "blue", "role": "operative"},
            {"display_name": "luke", "team_color": "blue", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "guesses": [
                    {
                        "word": "PENTAGON",
                        "timestamp_sec": 10.0,
                        "selector_name": "Cherry",
                        "selector_source": "selector_crop",
                        "selector_confidence": 0.8953,
                    },
                    {
                        "word": "PAGE",
                        "timestamp_sec": 12.0,
                        "selector_candidates": [
                            {"display_name": "Drew", "composite_score": 0.2785},
                            {"display_name": "Cherry", "composite_score": 0.2546},
                            {"display_name": "luke", "composite_score": 0.2496},
                        ],
                    },
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    guess = analysis["turns"][0]["guesses"][1]
    assert guess["selector_name"] == "Cherry"
    assert guess["selector_source"] == "turn_continuity"


def test_turn_candidate_top_backfills_clear_unresolved_guess_for_three_operative_team() -> None:
    analysis = {
        "roster": [
            {"display_name": "ty", "team_color": "blue", "role": "operative"},
            {"display_name": "Drew", "team_color": "blue", "role": "operative"},
            {"display_name": "arker", "team_color": "blue", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "guesses": [
                    {"word": "LEPRECHAUN", "timestamp_sec": 10.0, "selector_name": "ty", "selector_source": "selector_crop", "selector_confidence": 0.9},
                    {
                        "word": "EUROPE",
                        "timestamp_sec": 12.0,
                        "selector_candidates": [
                            {"display_name": "ty", "composite_score": 0.3268},
                            {"display_name": "arker", "composite_score": 0.3228},
                            {"display_name": "Drew", "composite_score": 0.2699},
                        ],
                    },
                    {"word": "PIT", "timestamp_sec": 15.0, "selector_name": "Drew", "selector_source": "selector_crop", "selector_confidence": 0.9},
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    guess = analysis["turns"][0]["guesses"][1]
    assert guess["selector_name"] == "ty"
    assert guess["selector_source"] == "turn_candidate_top"


def test_forward_only_continuity_keeps_earlier_blue_guess_from_inheriting_later_selector() -> None:
    analysis = {
        "roster": [
            {"display_name": "ty", "team_color": "blue", "role": "operative"},
            {"display_name": "Drew", "team_color": "blue", "role": "operative"},
            {"display_name": "arker", "team_color": "blue", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "blue",
                "guesses": [
                    {
                        "word": "GROOM",
                        "timestamp_sec": 10.0,
                        "selector_candidates": [
                            {
                                "display_name": "ty",
                                "pill_distance": 2.8366,
                                "template_score": 0.3829,
                                "avatar_hamming": 22,
                                "avatar_template_score": 0.4335,
                                "composite_score": 0.3282,
                            },
                            {
                                "display_name": "arker",
                                "pill_distance": 3.7207,
                                "template_score": 0.3829,
                                "avatar_hamming": 25,
                                "avatar_template_score": 0.4592,
                                "composite_score": 0.3242,
                            },
                            {
                                "display_name": "Drew",
                                "pill_distance": 79.1036,
                                "template_score": 0.3845,
                                "avatar_hamming": 22,
                                "avatar_template_score": 0.4467,
                                "composite_score": 0.2643,
                            },
                        ],
                    },
                    {
                        "word": "FEVER",
                        "timestamp_sec": 12.0,
                        "selector_name": "Drew",
                        "selector_source": "selector_crop",
                        "selector_confidence": 0.89,
                    },
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    guess = analysis["turns"][0]["guesses"][0]
    assert guess["selector_name"] == "ty"
    assert guess["selector_source"] == "turn_candidate_top"


def test_shared_alternative_backfills_pair_when_last_guess_is_resolved() -> None:
    analysis = {
        "roster": [
            {"display_name": "near", "team_color": "red", "role": "operative"},
            {"display_name": "revel", "team_color": "red", "role": "operative"},
            {"display_name": "shadowsn", "team_color": "red", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "red",
                "guesses": [
                    {
                        "word": "PURSE",
                        "timestamp_sec": 10.0,
                        "reference_selector_candidates": [
                            {"display_name": "near", "reference_template_score": 0.5427},
                            {"display_name": "shadowsn", "reference_template_score": 0.5388},
                            {"display_name": "revel", "reference_template_score": 0.5474},
                        ],
                        "selector_candidates": [
                            {"display_name": "near", "pill_distance": 1.25, "template_score": 0.4593, "composite_score": 0.3972},
                            {"display_name": "revel", "pill_distance": 1.0983, "template_score": 0.4634, "composite_score": 0.4003},
                            {"display_name": "shadowsn", "pill_distance": 1.4121, "template_score": 0.4555, "composite_score": 0.3657},
                        ],
                    },
                    {
                        "word": "GOLDILOCKS",
                        "timestamp_sec": 12.0,
                        "reference_selector_candidates": [
                            {"display_name": "near", "reference_template_score": 0.4943},
                            {"display_name": "revel", "reference_template_score": 0.4819},
                            {"display_name": "shadowsn", "reference_template_score": 0.4817},
                        ],
                        "selector_candidates": [
                            {"display_name": "near", "pill_distance": 1.3211, "template_score": 0.4577, "composite_score": 0.4122},
                            {"display_name": "revel", "pill_distance": 1.1741, "template_score": 0.4495, "composite_score": 0.4143},
                            {"display_name": "shadowsn", "pill_distance": 1.5828, "template_score": 0.4601, "composite_score": 0.3822},
                        ],
                    },
                    {
                        "word": "JEWELER",
                        "timestamp_sec": 15.0,
                        "selector_candidates": [
                            {"display_name": "revel", "pill_distance": 1.0335, "template_score": 0.4076, "composite_score": 0.391},
                            {"display_name": "near", "pill_distance": 1.2095, "template_score": 0.4075, "composite_score": 0.3909},
                            {"display_name": "shadowsn", "pill_distance": 1.3032, "template_score": 0.4083, "composite_score": 0.3599},
                        ],
                    },
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    first_guess, second_guess, third_guess = analysis["turns"][0]["guesses"]
    assert third_guess["selector_name"] == "revel"
    assert third_guess["selector_source"] in {"selector_candidates", "last_guess_candidate_top"}
    assert first_guess["selector_name"] == "near"
    assert first_guess["selector_source"] == "turn_shared_alternative"
    assert second_guess["selector_name"] == "near"
    assert second_guess["selector_source"] == "turn_shared_alternative"


def test_backward_continuity_uses_name_focused_candidates_for_last_resolved_guess() -> None:
    analysis = {
        "roster": [
            {"display_name": "near", "team_color": "red", "role": "operative"},
            {"display_name": "revel", "team_color": "red", "role": "operative"},
            {"display_name": "shadowsn", "team_color": "red", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "red",
                "guesses": [
                    {
                        "word": "SIGN",
                        "timestamp_sec": 10.0,
                        "reference_selector_candidates": [
                            {"display_name": "shadowsn", "reference_template_score": 0.4906},
                            {"display_name": "near", "reference_template_score": 0.4599},
                            {"display_name": "revel", "reference_template_score": 0.4404},
                        ],
                        "selector_candidates": [
                            {"display_name": "near", "pill_distance": 0.7405, "template_score": 0.4317, "composite_score": 0.4446},
                            {"display_name": "revel", "pill_distance": 0.7655, "template_score": 0.4335, "composite_score": 0.4315},
                            {"display_name": "shadowsn", "pill_distance": 0.796, "template_score": 0.4295, "composite_score": 0.4159},
                        ],
                    },
                    {
                        "word": "CODE",
                        "timestamp_sec": 12.0,
                        "reference_selector_candidates": [
                            {"display_name": "shadowsn", "reference_template_score": 0.4418},
                            {"display_name": "near", "reference_template_score": 0.4163},
                            {"display_name": "revel", "reference_template_score": 0.3998},
                        ],
                        "selector_candidates": [
                            {"display_name": "near", "pill_distance": 1.7803, "template_score": 0.3459, "composite_score": 0.3897},
                            {"display_name": "revel", "pill_distance": 1.6578, "template_score": 0.3451, "composite_score": 0.3719},
                            {"display_name": "shadowsn", "pill_distance": 2.0325, "template_score": 0.3591, "composite_score": 0.361},
                        ],
                    },
                    {
                        "word": "OPERA",
                        "timestamp_sec": 15.0,
                        "selector_candidates": [
                            {"display_name": "revel", "pill_distance": 0.7575, "template_score": 0.4023, "composite_score": 0.4169},
                            {"display_name": "near", "pill_distance": 0.934, "template_score": 0.3924, "composite_score": 0.4094},
                            {"display_name": "shadowsn", "pill_distance": 0.9949, "template_score": 0.4015, "composite_score": 0.3846},
                        ],
                    },
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    first_guess, second_guess, third_guess = analysis["turns"][0]["guesses"]
    assert third_guess["selector_name"] == "revel"
    assert first_guess["selector_name"] == "revel"
    assert second_guess["selector_name"] == "revel"


def test_middle_breakout_assigns_outer_pair_to_shared_alternative() -> None:
    analysis = {
        "roster": [
            {"display_name": "near", "team_color": "red", "role": "operative"},
            {"display_name": "revel", "team_color": "red", "role": "operative"},
            {"display_name": "shadowsn", "team_color": "red", "role": "operative"},
        ],
        "turns": [
            {
                "turn_index": 0,
                "team_color": "red",
                "guesses": [
                    {
                        "word": "WEB",
                        "timestamp_sec": 10.0,
                        "reference_selector_candidates": [
                            {"display_name": "shadowsn", "reference_template_score": 0.509},
                            {"display_name": "near", "reference_template_score": 0.4842},
                            {"display_name": "revel", "reference_template_score": 0.4715},
                        ],
                        "selector_candidates": [
                            {"display_name": "revel", "pill_distance": 1.0105, "template_score": 0.4436, "composite_score": 0.4029},
                            {"display_name": "near", "pill_distance": 1.1636, "template_score": 0.4392, "composite_score": 0.3896},
                            {"display_name": "shadowsn", "pill_distance": 1.299, "template_score": 0.423, "composite_score": 0.3621},
                        ],
                    },
                    {
                        "word": "ANGEL",
                        "timestamp_sec": 12.0,
                        "selector_candidates": [
                            {"display_name": "shadowsn", "pill_distance": 0.6905, "template_score": 0.398, "composite_score": 0.3942},
                            {"display_name": "revel", "pill_distance": 0.8472, "template_score": 0.3964, "composite_score": 0.3963},
                            {"display_name": "near", "pill_distance": 1.031, "template_score": 0.3916, "composite_score": 0.3783},
                        ],
                    },
                    {
                        "word": "GREENHOUSE",
                        "timestamp_sec": 15.0,
                        "reference_selector_candidates": [
                            {"display_name": "shadowsn", "reference_template_score": 0.4967},
                            {"display_name": "near", "reference_template_score": 0.4792},
                            {"display_name": "revel", "reference_template_score": 0.4709},
                        ],
                        "selector_candidates": [
                            {"display_name": "revel", "pill_distance": 2.2178, "template_score": 0.4327, "composite_score": 0.3427},
                            {"display_name": "near", "pill_distance": 2.3712, "template_score": 0.424, "composite_score": 0.3461},
                            {"display_name": "shadowsn", "pill_distance": 2.5017, "template_score": 0.4325, "composite_score": 0.3354},
                        ],
                    },
                ],
            }
        ],
    }

    resolve_selector_fields_inplace(analysis)

    first_guess, second_guess, third_guess = analysis["turns"][0]["guesses"]
    assert second_guess["selector_name"] == "shadowsn"
    assert first_guess["selector_name"] == "near"
    assert first_guess["selector_source"] == "turn_middle_breakout"
    assert third_guess["selector_name"] == "near"
    assert third_guess["selector_source"] == "turn_middle_breakout"
