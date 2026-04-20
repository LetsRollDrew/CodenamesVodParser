from __future__ import annotations

from app.parsers.gamelog import ClueEvent, GuessEvent
from app.core.models import CardColor, PlayerRole, PlayerRosterEntry, TeamColor
from app.parsers.banner import (
    _banner_signatures_equivalent,
    _build_banner_clue_event,
    _drop_visible_events_when_banner_conflicts,
    _focus_visible_events_on_latest_turn,
    _infer_active_guessing_team,
    _refine_visible_events_from_banner,
    _should_probe_visible_game_log,
    _suppress_unconfirmed_visible_clue,
)


def _roster() -> list[PlayerRosterEntry]:
    return [
        PlayerRosterEntry(
            player_name="drew",
            display_name="Drew",
            team_color=TeamColor.BLUE,
            role=PlayerRole.OPERATIVE,
        ),
        PlayerRosterEntry(
            player_name="cherry",
            display_name="Cherry",
            team_color=TeamColor.BLUE,
            role=PlayerRole.OPERATIVE,
        ),
        PlayerRosterEntry(
            player_name="ty",
            display_name="ty",
            team_color=TeamColor.BLUE,
            role=PlayerRole.SPYMASTER,
        ),
        PlayerRosterEntry(
            player_name="near",
            display_name="near",
            team_color=TeamColor.RED,
            role=PlayerRole.OPERATIVE,
        ),
        PlayerRosterEntry(
            player_name="shadowsn",
            display_name="shadowsn",
            team_color=TeamColor.RED,
            role=PlayerRole.SPYMASTER,
        ),
    ]


def test_refine_visible_events_from_banner_only_updates_latest_clue() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="shadowsn",
            clue_text="PROM",
            clue_count="4",
            timestamp_sec=100.0,
            sequence_index=0,
            confidence=0.7,
        ),
        GuessEvent(
            player_name="near",
            word="SUIT",
            card_color=CardColor.RED,
            timestamp_sec=101.0,
            sequence_index=1,
            confidence=0.8,
        ),
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="ty",
            clue_text="ANIMA",
            clue_count="2",
            timestamp_sec=140.0,
            sequence_index=2,
            confidence=0.6,
        ),
    ]

    refined = _refine_visible_events_from_banner(
        events,
        top_banner_text="WAIT FOR YOUR TURN, Drew Cherry ARE GUESSING",
        roster=_roster(),
        clue_banner_text="ANIMAL",
        clue_banner_count="2",
    )

    assert refined[0].clue_text == "PROM"
    assert refined[0].clue_count == "4"
    assert refined[2].clue_text == "ANIMAL"
    assert refined[2].clue_count == "2"


def test_focus_visible_events_on_latest_turn_drops_older_clues_and_guesses_once_latest_turn_has_guesses() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="shadowsn",
            clue_text="PROM",
            clue_count="4",
            timestamp_sec=100.0,
            sequence_index=0,
            confidence=0.7,
        ),
        GuessEvent(
            player_name="near",
            word="SUIT",
            card_color=CardColor.RED,
            timestamp_sec=101.0,
            sequence_index=1,
            confidence=0.8,
        ),
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="ty",
            clue_text="ANIMA",
            clue_count="2",
            timestamp_sec=140.0,
            sequence_index=2,
            confidence=0.6,
        ),
        GuessEvent(
            player_name="Drew",
            word="DRAGON",
            card_color=CardColor.BLUE,
            timestamp_sec=141.0,
            sequence_index=3,
            confidence=0.84,
        ),
    ]

    focused = _focus_visible_events_on_latest_turn(events)

    assert len(focused) == 2
    assert isinstance(focused[0], ClueEvent)
    assert focused[0].clue_text == "ANIMA"
    assert isinstance(focused[1], GuessEvent)
    assert focused[1].word == "DRAGON"


def test_refine_visible_events_from_banner_updates_latest_clue_even_with_guess_rows_following() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="ty",
            clue_text="ANIMA",
            clue_count="2",
            timestamp_sec=241.0,
            sequence_index=0,
            confidence=0.7,
        ),
        GuessEvent(
            player_name="Drew",
            word="DRAGON",
            card_color=CardColor.BLUE,
            timestamp_sec=243.0,
            sequence_index=1,
            confidence=0.8,
        ),
    ]

    refined = _refine_visible_events_from_banner(
        events,
        top_banner_text="WAIT FOR YOUR TURN, Drew Cherry ARE GUESSING",
        roster=_roster(),
        clue_banner_text="ANIMAL",
        clue_banner_count="2",
    )

    assert isinstance(refined[0], ClueEvent)
    assert refined[0].clue_text == "ANIMAL"
    assert refined[0].clue_count == "2"


def test_suppress_unconfirmed_visible_clue_drops_weak_clue_only_slice() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="ty",
            clue_text="GE",
            clue_count="3",
            timestamp_sec=425.0,
            sequence_index=0,
            confidence=0.11,
        )
    ]

    suppressed = _suppress_unconfirmed_visible_clue(
        events,
        clue_banner_text="LEATHER",
        clue_banner_count="2",
    )

    assert suppressed == []


def test_suppress_unconfirmed_visible_clue_drops_same_count_wrong_word_slice() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="ty",
            clue_text="OE",
            clue_count="2",
            timestamp_sec=435.0,
            sequence_index=0,
            confidence=0.02,
        )
    ]

    suppressed = _suppress_unconfirmed_visible_clue(
        events,
        clue_banner_text="LEATHER",
        clue_banner_count="2",
    )

    assert suppressed == []


def test_suppress_unconfirmed_visible_clue_keeps_banner_confirmed_slice() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="shadowsn",
            clue_text="LEATH",
            clue_count="2",
            timestamp_sec=447.0,
            sequence_index=0,
            confidence=0.18,
        )
    ]

    kept = _suppress_unconfirmed_visible_clue(
        events,
        clue_banner_text="LEATHER",
        clue_banner_count="2",
    )

    assert len(kept) == 1
    assert isinstance(kept[0], ClueEvent)


def test_drop_visible_events_when_banner_conflicts_without_matching_clue() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="ty",
            clue_text="IRELAND",
            clue_count="2",
            timestamp_sec=3728.0,
            sequence_index=0,
            confidence=1.0,
        ),
        GuessEvent(
            player_name="unknown",
            word="LEPRECHAUN",
            card_color=CardColor.BLUE,
            timestamp_sec=3728.0,
            sequence_index=1,
            confidence=0.84,
        ),
    ]

    dropped = _drop_visible_events_when_banner_conflicts(
        events,
        clue_banner_text="BATH",
        clue_banner_count="2",
    )

    assert dropped == []


def test_banner_signatures_equivalent_ignores_glide_count_flicker() -> None:
    assert _banner_signatures_equivalent(("ANIMAL", None), ("ANIMAL", "2"))
    assert _banner_signatures_equivalent(("LEATH", "2"), ("LEATHER", None))


def test_infer_active_guessing_team_requires_more_than_short_name_substring_noise() -> None:
    assert _infer_active_guessing_team("WAIT FOR YOUR TURN, ARE GUESSING TEVD", _roster()) is None


def test_build_banner_clue_event_does_not_flip_same_persistent_banner_to_opposite_team() -> None:
    history = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="Cherry",
            clue_text="WOMEN",
            clue_count="3",
            timestamp_sec=60.0,
            sequence_index=0,
            confidence=0.9,
        )
    ]

    banner_event = _build_banner_clue_event(
        clue_text="WOMEN",
        clue_count="1",
        top_banner_text="WAIT FOR YOUR TURN, ARE GUESSING IED",
        roster=_roster(),
        history=history,
        left_counter=8,
        right_counter=9,
        timestamp_sec=62.0,
    )

    assert banner_event is None


def test_build_banner_clue_event_rejects_zero_count_banner_noise() -> None:
    banner_event = _build_banner_clue_event(
        clue_text="MZN",
        clue_count="0",
        top_banner_text="WAIT FOR YOUR TURN, near ARE GUESSING",
        roster=_roster(),
        history=[],
        left_counter=8,
        right_counter=9,
        timestamp_sec=83.0,
    )

    assert banner_event is None


def test_build_banner_clue_event_rejects_roster_name_like_clue() -> None:
    banner_event = _build_banner_clue_event(
        clue_text="Cherry",
        clue_count="2",
        top_banner_text="WAIT FOR YOUR TURN, near ARE GUESSING",
        roster=_roster(),
        history=[],
        left_counter=8,
        right_counter=9,
        timestamp_sec=83.0,
    )

    assert banner_event is None


def test_should_probe_visible_game_log_rechecks_quiet_banner_when_no_visible_clue() -> None:
    assert _should_probe_visible_game_log(
        log_changed=False,
        quiet_log_frame_count=2,
        clue_banner_text="ANIMAL",
        clue_banner_count="2",
        previous_visible_events=[],
    )


def test_should_probe_visible_game_log_skips_when_visible_clue_matches_banner() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="ty",
            clue_text="ANIMAL",
            clue_count="2",
            timestamp_sec=241.0,
            sequence_index=0,
            confidence=0.7,
        ),
        GuessEvent(
            player_name="Drew",
            word="DRAGON",
            card_color=CardColor.BLUE,
            timestamp_sec=243.0,
            sequence_index=1,
            confidence=0.8,
        ),
    ]

    assert _should_probe_visible_game_log(
        log_changed=False,
        quiet_log_frame_count=2,
        clue_banner_text="ANIMAL",
        clue_banner_count="2",
        previous_visible_events=events,
    )


def test_focus_visible_events_on_latest_turn_keeps_only_latest_clue_row_once_newer_clue_exists() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="Cherry",
            clue_text="WOMEN",
            clue_count="3",
            timestamp_sec=10.0,
            sequence_index=0,
            confidence=0.7,
        ),
        GuessEvent(
            player_name="near",
            word="PURSE",
            card_color=CardColor.RED,
            timestamp_sec=12.0,
            sequence_index=1,
            confidence=0.85,
        ),
        GuessEvent(
            player_name="near",
            word="GOLDILOCKS",
            card_color=CardColor.RED,
            timestamp_sec=14.0,
            sequence_index=2,
            confidence=0.85,
        ),
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="luke",
            clue_text="IRELAND",
            clue_count="2",
            timestamp_sec=20.0,
            sequence_index=3,
            confidence=0.8,
        ),
    ]

    focused = _focus_visible_events_on_latest_turn(events)

    assert len(focused) == 1
    assert isinstance(focused[0], ClueEvent)
    assert focused[0].clue_text == "IRELAND"


def test_focus_visible_events_on_latest_turn_drops_older_clue_when_no_intervening_guesses() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="Cherry",
            clue_text="RUISE",
            clue_count="6",
            timestamp_sec=18.0,
            sequence_index=0,
            confidence=0.32,
        ),
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="luke",
            clue_text="IRELAND",
            clue_count="2",
            timestamp_sec=20.0,
            sequence_index=1,
            confidence=0.84,
        ),
    ]

    focused = _focus_visible_events_on_latest_turn(events)

    assert len(focused) == 1
    assert isinstance(focused[0], ClueEvent)
    assert focused[0].clue_text == "IRELAND"


def test_refine_visible_events_from_banner_keeps_stable_log_count_over_noisy_banner_count() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.BLUE,
            spymaster_name="ty",
            clue_text="ANIMA",
            clue_count="2",
            timestamp_sec=241.0,
            sequence_index=0,
            confidence=0.7,
        )
    ]

    refined = _refine_visible_events_from_banner(
        events,
        top_banner_text="WAIT FOR YOUR TURN, Drew Cherry ARE GUESSING",
        roster=_roster(),
        clue_banner_text="ANIMAL",
        clue_banner_count="7",
    )

    assert refined[0].clue_text == "ANIMAL"
    assert refined[0].clue_count == "2"


def test_refine_visible_events_from_banner_does_not_rewrite_older_clue_when_guess_rows_follow() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="shadowsn",
            clue_text="PROM",
            clue_count="4",
            timestamp_sec=241.0,
            sequence_index=0,
            confidence=0.7,
        ),
        GuessEvent(
            player_name="unknown",
            word="COLLAR",
            card_color=CardColor.RED,
            timestamp_sec=241.0,
            sequence_index=1,
            confidence=0.8,
        ),
    ]

    refined = _refine_visible_events_from_banner(
        events,
        top_banner_text="",
        roster=_roster(),
        clue_banner_text="ANIMAL",
        clue_banner_count="2",
    )

    assert refined[0].clue_text == "PROM"
    assert refined[0].clue_count == "4"


def test_refine_visible_events_from_banner_does_not_replace_unrelated_clue_text() -> None:
    events = [
        ClueEvent(
            team_color=TeamColor.RED,
            spymaster_name="shadowsn",
            clue_text="PROM",
            clue_count="4",
            timestamp_sec=243.0,
            sequence_index=0,
            confidence=0.7,
        )
    ]

    refined = _refine_visible_events_from_banner(
        events,
        top_banner_text="",
        roster=_roster(),
        clue_banner_text="ANIMAL",
        clue_banner_count="2",
    )

    assert refined[0].clue_text == "PROM"
    assert refined[0].clue_count == "4"
