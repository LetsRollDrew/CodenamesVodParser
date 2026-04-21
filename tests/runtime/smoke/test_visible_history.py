from __future__ import annotations

from app.parsers.gamelog import ClueEvent, GuessEvent
from app.runtime.smoke.cycle.history_merge import _append_new_visible_events
from app.runtime.smoke.cycle.visible_log import _commit_visible_log_events
from app.runtime.smoke.runtime_models import SmokeRuntimeState

from app.core.models import CardColor, TeamColor


def _guess(word: str, *, timestamp_sec: float, confidence: float = 0.6, player_name: str = "unknown") -> GuessEvent:
    return GuessEvent(
        player_name=player_name,
        word=word,
        card_color=CardColor.RED,
        timestamp_sec=timestamp_sec,
        sequence_index=0,
        confidence=confidence,
    )


def _clue(
    clue_text: str,
    *,
    timestamp_sec: float,
    confidence: float = 0.6,
    clue_count: str = "4",
    team_color: TeamColor = TeamColor.RED,
    spymaster_name: str = "shadowsn",
) -> ClueEvent:
    return ClueEvent(
        team_color=team_color,
        spymaster_name=spymaster_name,
        clue_text=clue_text,
        clue_count=clue_count,
        timestamp_sec=timestamp_sec,
        sequence_index=0,
        confidence=confidence,
    )


def test_append_new_visible_events_keeps_previous_visible_on_empty_parse() -> None:
    suit = _guess("SUIT", timestamp_sec=129.0)

    history, previous_visible, new_events = _append_new_visible_events([], [], [suit])
    assert new_events == [suit]

    history, previous_visible, new_events = _append_new_visible_events(history, previous_visible, [])
    assert new_events == []
    assert previous_visible == [suit]

    history, previous_visible, new_events = _append_new_visible_events(
        history,
        previous_visible,
        [_guess("SUIT", timestamp_sec=131.0, confidence=0.7)],
    )
    assert new_events == []
    assert len(history) == 1


def test_append_new_visible_events_replaces_duplicate_guess_with_better_candidate() -> None:
    initial = _guess("CROWN", timestamp_sec=181.0, confidence=0.52)
    improved = _guess("CROWN", timestamp_sec=195.0, confidence=0.62, player_name="near")

    history, previous_visible, _ = _append_new_visible_events([], [], [initial])
    history, previous_visible, new_events = _append_new_visible_events(history, previous_visible, [improved])

    assert new_events == []
    assert len(history) == 1
    assert history[0].player_name == "near"
    assert history[0].confidence == 0.62
    assert history[0].timestamp_sec == 181.0


def test_append_new_visible_events_replaces_persistent_clue_variant_in_same_team_span() -> None:
    initial = _clue("PROM", timestamp_sec=93.0, confidence=0.62)
    improved = _clue("PROME", timestamp_sec=145.0, confidence=0.66)

    history = [initial]
    previous_visible = [initial]
    history, previous_visible, new_events = _append_new_visible_events(history, previous_visible, [improved])

    assert new_events == []
    assert len(history) == 1
    assert isinstance(history[0], ClueEvent)
    assert history[0].clue_text == "PROME"
    assert history[0].timestamp_sec == 93.0
    assert previous_visible == [improved]


def test_append_new_visible_events_backfills_stale_residue_guess_into_earlier_turn() -> None:
    prom = _clue("PROM", timestamp_sec=93.0, confidence=0.95)
    suit = _guess("SUIT", timestamp_sec=141.0)
    collar = _guess("COLLAR", timestamp_sec=183.0)
    animal = _clue(
        "ANIMAL",
        timestamp_sec=241.0,
        confidence=0.95,
        clue_count="2",
        team_color=TeamColor.BLUE,
        spymaster_name="ty",
    )
    history = [prom, suit, collar, animal]

    history, previous_visible, new_events = _append_new_visible_events(
        history,
        [],
        [prom, suit, collar, _guess("CROWN", timestamp_sec=247.0, player_name="near")],
    )

    assert len(new_events) == 1
    assert isinstance(new_events[0], GuessEvent)
    assert new_events[0].word == "CROWN"
    assert isinstance(history[3], GuessEvent)
    assert history[3].word == "CROWN"
    assert isinstance(history[4], ClueEvent)
    assert history[4].clue_text == "ANIMAL"
    assert previous_visible[0] == prom


def test_append_new_visible_events_matches_long_running_clue_within_extended_window() -> None:
    initial = _clue("PROM", timestamp_sec=93.0, confidence=0.95)
    later = _clue("PROM", timestamp_sec=221.0, confidence=0.82)

    history = [initial]
    previous_visible = [initial]
    history, previous_visible, new_events = _append_new_visible_events(history, previous_visible, [later])

    assert new_events == []
    assert len(history) == 1
    assert isinstance(history[0], ClueEvent)
    assert history[0].timestamp_sec == 93.0


def test_append_new_visible_events_keeps_best_clue_text_when_later_fragment_has_better_count() -> None:
    first = _clue(
        "CONSTITUTIONS",
        timestamp_sec=869.0,
        confidence=0.56,
        clue_count="6",
        team_color=TeamColor.BLUE,
        spymaster_name="ty",
    )
    later = _clue(
        "ITUTIO",
        timestamp_sec=899.0,
        confidence=0.29,
        clue_count="infinity",
        team_color=TeamColor.BLUE,
        spymaster_name="ty",
    )

    history = [first]
    previous_visible = [first]
    history, previous_visible, new_events = _append_new_visible_events(history, previous_visible, [later])

    assert new_events == []
    assert len(history) == 1
    assert isinstance(history[0], ClueEvent)
    assert history[0].clue_text == "CONSTITUTIONS"
    assert history[0].clue_count == "infinity"


def test_append_new_visible_events_keeps_stable_small_count_over_later_infinity_noise() -> None:
    first = _clue("PROM", timestamp_sec=93.0, confidence=1.0, clue_count="4")
    later = _clue("PROM", timestamp_sec=221.0, confidence=0.57, clue_count="infinity")

    history = [first]
    previous_visible = [first]
    history, previous_visible, new_events = _append_new_visible_events(history, previous_visible, [later])

    assert new_events == []
    assert len(history) == 1
    assert isinstance(history[0], ClueEvent)
    assert history[0].clue_text == "PROM"
    assert history[0].clue_count == "4"


def test_append_new_visible_events_prefers_non_neutral_duplicate_guess_color() -> None:
    neutral = GuessEvent(
        player_name="unknown",
        word="BACON",
        card_color=CardColor.NEUTRAL,
        timestamp_sec=295.0,
        sequence_index=0,
        confidence=0.88,
    )
    corrected = GuessEvent(
        player_name="Drew",
        word="BACON",
        card_color=CardColor.BLUE,
        timestamp_sec=421.0,
        sequence_index=0,
        confidence=0.86,
    )

    history, previous_visible, _ = _append_new_visible_events([], [], [neutral])
    history, previous_visible, new_events = _append_new_visible_events(history, previous_visible, [corrected])

    assert new_events == []
    assert len(history) == 1
    assert isinstance(history[0], GuessEvent)
    assert history[0].card_color == CardColor.BLUE
    assert history[0].player_name == "Drew"


def test_append_new_visible_events_allows_repeated_same_word_after_long_gap() -> None:
    history, previous_visible, _ = _append_new_visible_events(
        [],
        [],
        [_clue("RENOVATION", timestamp_sec=10.0, clue_count="2")],
    )
    history, previous_visible, _ = _append_new_visible_events(
        history,
        previous_visible,
        [_guess("SAW", timestamp_sec=30.0, player_name="arker")],
    )
    history, previous_visible, _ = _append_new_visible_events(history, previous_visible, [])
    history, previous_visible, new_events = _append_new_visible_events(
        history,
        previous_visible,
        [_guess("SAW", timestamp_sec=55.0, player_name="arker")],
    )

    assert len(new_events) == 1
    saw_guesses = [event for event in history if isinstance(event, GuessEvent)]
    assert len(saw_guesses) == 2


def test_append_new_visible_events_does_not_backfill_old_clue_residue_far_later() -> None:
    transit = _clue(
        "TRANSIT",
        timestamp_sec=100.0,
        confidence=0.95,
        clue_count="3",
        team_color=TeamColor.BLUE,
        spymaster_name="Drew",
    )
    renovation = _clue("RENOVATION", timestamp_sec=220.0, confidence=0.95, clue_count="2")
    history = [transit, renovation]

    history, previous_visible, new_events = _append_new_visible_events(
        history,
        [],
        [
            _clue(
                "TRANSIT",
                timestamp_sec=300.0,
                confidence=0.8,
                clue_count="3",
                team_color=TeamColor.BLUE,
                spymaster_name="Drew",
            ),
            _guess("SAW", timestamp_sec=300.0, player_name="arker"),
        ],
    )

    assert len(new_events) == 1
    assert isinstance(history[-1], GuessEvent)
    assert history[-1].word == "SAW"


def test_append_new_visible_events_does_not_retrack_old_visible_clue_with_recent_guess_rows() -> None:
    ireland = _clue(
        "IRELAND",
        timestamp_sec=3420.0,
        confidence=1.0,
        clue_count="2",
        team_color=TeamColor.BLUE,
        spymaster_name="luke",
    )
    leprechaun = GuessEvent(
        player_name="unknown",
        word="LEPRECHAUN",
        card_color=CardColor.BLUE,
        timestamp_sec=3456.0,
        sequence_index=0,
        confidence=0.84,
    )
    europe = GuessEvent(
        player_name="unknown",
        word="EUROPE",
        card_color=CardColor.BLUE,
        timestamp_sec=3457.0,
        sequence_index=1,
        confidence=0.82,
    )
    language = _clue("LANGUAGE", timestamp_sec=3500.0, confidence=0.9, clue_count="2")
    history = [ireland, leprechaun, europe, language]

    history, _, new_events = _append_new_visible_events(
        history,
        [],
        [
            _clue(
                "IRELAND",
                timestamp_sec=3728.0,
                confidence=1.0,
                clue_count="2",
                team_color=TeamColor.BLUE,
                spymaster_name="luke",
            ),
            GuessEvent(
                player_name="unknown",
                word="LEPRECHAUN",
                card_color=CardColor.BLUE,
                timestamp_sec=3728.0,
                sequence_index=2,
                confidence=0.84,
            ),
            GuessEvent(
                player_name="unknown",
                word="EUROPE",
                card_color=CardColor.BLUE,
                timestamp_sec=3728.0,
                sequence_index=3,
                confidence=0.82,
            ),
        ],
    )

    assert new_events == []
    ireland_clues = [
        event
        for event in history
        if isinstance(event, ClueEvent) and event.clue_text == "IRELAND"
    ]
    assert len(ireland_clues) == 1


def test_append_new_visible_events_trims_historical_prefix_before_new_turn_suffix() -> None:
    language = _clue(
        "LANGUAGE",
        timestamp_sec=3500.0,
        confidence=0.95,
        clue_count="2",
        team_color=TeamColor.RED,
        spymaster_name="Cherry",
    )
    sign = GuessEvent(
        player_name="unknown",
        word="SIGN",
        card_color=CardColor.RED,
        timestamp_sec=3520.0,
        sequence_index=0,
        confidence=0.84,
    )
    code = GuessEvent(
        player_name="unknown",
        word="CODE",
        card_color=CardColor.RED,
        timestamp_sec=3535.0,
        sequence_index=1,
        confidence=0.82,
    )
    opera = GuessEvent(
        player_name="unknown",
        word="OPERA",
        card_color=CardColor.BLUE,
        timestamp_sec=3560.0,
        sequence_index=2,
        confidence=0.8,
    )
    history = [language, sign, code, opera]

    bath = _clue(
        "BATH",
        timestamp_sec=3716.0,
        confidence=0.95,
        clue_count="2",
        team_color=TeamColor.BLUE,
        spymaster_name="luke",
    )
    groom = GuessEvent(
        player_name="unknown",
        word="GROOM",
        card_color=CardColor.BLUE,
        timestamp_sec=3716.0,
        sequence_index=0,
        confidence=0.84,
    )

    history, _, new_events = _append_new_visible_events(
        history,
        [],
        [language, sign, code, opera, bath, groom],
    )

    assert [type(event).__name__ for event in new_events] == ["GuessEvent"]
    assert isinstance(new_events[0], GuessEvent)
    assert new_events[0].word == "GROOM"
    bath_clues = [
        event
        for event in history
        if isinstance(event, ClueEvent) and event.clue_text == "BATH"
    ]
    assert bath_clues == []


def test_commit_visible_log_events_suppresses_recently_seen_rows_with_same_clue_context() -> None:
    state = SmokeRuntimeState()
    visible_slice = [
        _clue("LANGUAGE", timestamp_sec=100.0, clue_count="2"),
        _guess("SIGN", timestamp_sec=101.0, player_name="Drew"),
    ]

    first_new_events = _commit_visible_log_events(state, visible_slice, timestamp_sec=100.0)
    second_new_events = _commit_visible_log_events(state, visible_slice, timestamp_sec=104.0)

    assert len(first_new_events) == 1
    assert second_new_events == []
    assert len(state.history) == 1


def test_commit_visible_log_events_allows_same_guess_under_new_clue_context() -> None:
    state = SmokeRuntimeState()
    first_slice = [
        _clue("LANGUAGE", timestamp_sec=100.0, clue_count="2"),
        _guess("SIGN", timestamp_sec=101.0, player_name="Drew"),
    ]
    second_slice = [
        _clue("BATH", timestamp_sec=160.0, clue_count="2", team_color=TeamColor.BLUE, spymaster_name="ty"),
        GuessEvent(
            player_name="Drew",
            word="SIGN",
            card_color=CardColor.BLUE,
            timestamp_sec=161.0,
            sequence_index=0,
            confidence=0.72,
        ),
    ]

    _commit_visible_log_events(state, first_slice, timestamp_sec=100.0)
    new_events = _commit_visible_log_events(state, second_slice, timestamp_sec=160.0)

    assert [type(event).__name__ for event in new_events] == ["GuessEvent"]
    assert isinstance(new_events[0], GuessEvent)
    assert new_events[0].word == "SIGN"


def test_new_clue_can_only_enter_history_via_pending_promotion() -> None:
    state = SmokeRuntimeState()

    new_events = _commit_visible_log_events(
        state,
        [
            _clue("GAMELOGWOMEN", timestamp_sec=240.5, clue_count="3"),
            _guess("GOLDILOCK", timestamp_sec=240.5),
        ],
        timestamp_sec=240.5,
    )

    assert all(not isinstance(event, ClueEvent) for event in new_events)
    assert all(not isinstance(event, ClueEvent) for event in state.history)


def test_gamelogwomen_contaminated_slice_does_not_commit_even_with_guesses() -> None:
    state = SmokeRuntimeState(
        history=[
            _clue("IRELAND", timestamp_sec=239.0, clue_count="2", team_color=TeamColor.BLUE, spymaster_name="luke")
        ]
    )

    new_events = _commit_visible_log_events(
        state,
        [
            _clue("GAMELOGWOMEN", timestamp_sec=240.5, clue_count="3"),
            _guess("GOLDILOCK", timestamp_sec=240.5),
            _guess("JEWELER", timestamp_sec=240.5),
        ],
        timestamp_sec=240.5,
    )

    assert all(not isinstance(event, ClueEvent) for event in new_events)
    committed_clues = [event for event in state.history if isinstance(event, ClueEvent)]
    assert [event.clue_text for event in committed_clues] == ["IRELAND"]
