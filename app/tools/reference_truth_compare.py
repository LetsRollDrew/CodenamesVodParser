"""Compare parsed game outputs against manual reference-truth fixtures"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.vision.ocr.preprocessing import collapse_whitespace


def _normalize_name(value: Any) -> str:
    return collapse_whitespace(str(value or "")).casefold()


def _normalize_word(value: Any) -> str:
    return collapse_whitespace(str(value or "")).upper()


def _normalize_count(value: Any) -> str:
    normalized = collapse_whitespace(str(value or "")).casefold()
    if normalized in {"∞", "inf", "infinity"}:
        return "infinity"
    return normalized


def _normalize_result(value: Any) -> str:
    normalized = collapse_whitespace(str(value or "")).casefold()
    if normalized in {"wrong", "enemy"}:
        return "enemy"
    if normalized in {"neutral", "neutral card"}:
        return "neutral"
    return normalized


def _truth_games(payload: dict[str, Any]) -> dict[str, Any]:
    games = payload.get("games")
    if isinstance(games, dict):
        return games
    return {}


def _parsed_games(payload: dict[str, Any]) -> dict[str, Any]:
    if "games" in payload and isinstance(payload["games"], dict):
        return payload["games"]
    if isinstance(payload.get("turns"), list) and "winner_team" in payload:
        game_key = str(payload.get("game_key") or "game")
        return {game_key: payload}
    return {
        key: value
        for key, value in payload.items()
        if isinstance(value, dict) and key.casefold().startswith("game")
    }


def _truth_guess_events(turn: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in turn.get("events", [])
        if isinstance(event, dict) and event.get("event_type") == "guess"
    ]


def _truth_turn_end_markers(turn: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in turn.get("events", [])
        if isinstance(event, dict) and event.get("event_type") == "turn_end_marker"
    ]


def _parsed_guesses(turn: dict[str, Any]) -> list[dict[str, Any]]:
    guesses = turn.get("guesses", [])
    return [guess for guess in guesses if isinstance(guess, dict)]


def compare_truth_to_parsed(
    *,
    truth_payload: dict[str, Any],
    parsed_payload: dict[str, Any],
) -> dict[str, Any]:
    truth_games = _truth_games(truth_payload)
    parsed_games = _parsed_games(parsed_payload)
    report_games: dict[str, Any] = {}
    total_mismatches = 0

    for game_key, truth_game in truth_games.items():
        parsed_game = parsed_games.get(game_key)
        mismatches: list[dict[str, Any]] = []
        advisory: list[dict[str, Any]] = []
        if parsed_game is None:
            report_games[game_key] = {
                "status": "missing_game",
                "mismatches": [{"kind": "missing_game", "expected_game": game_key}],
                "advisory": advisory,
            }
            total_mismatches += 1
            continue

        if _normalize_name(parsed_game.get("winner_team")) != _normalize_name(truth_game.get("winner_team")):
            mismatches.append(
                {
                    "kind": "winner_team",
                    "expected": truth_game.get("winner_team"),
                    "actual": parsed_game.get("winner_team"),
                }
            )

        truth_turns = truth_game.get("turns", [])
        parsed_turns = parsed_game.get("turns", [])
        if len(parsed_turns) != len(truth_turns):
            mismatches.append(
                {
                    "kind": "turn_count",
                    "expected": len(truth_turns),
                    "actual": len(parsed_turns),
                }
            )

        turn_count = max(len(truth_turns), len(parsed_turns))
        for turn_index in range(turn_count):
            truth_turn = truth_turns[turn_index] if turn_index < len(truth_turns) else None
            parsed_turn = parsed_turns[turn_index] if turn_index < len(parsed_turns) else None
            if truth_turn is None:
                mismatches.append(
                    {
                        "kind": "unexpected_turn",
                        "turn_index": turn_index,
                        "actual": parsed_turn,
                    }
                )
                continue
            if parsed_turn is None:
                mismatches.append(
                    {
                        "kind": "missing_turn",
                        "turn_index": turn_index,
                        "expected": truth_turn,
                    }
                )
                continue

            for field_name, normalizer in (
                ("team_color", _normalize_name),
                ("spymaster_name", _normalize_name),
                ("clue_text", _normalize_word),
                ("clue_count", _normalize_count),
            ):
                expected = truth_turn.get(field_name)
                actual = parsed_turn.get(field_name)
                if normalizer(actual) != normalizer(expected):
                    mismatches.append(
                        {
                            "kind": "turn_field",
                            "turn_index": turn_index,
                            "field": field_name,
                            "expected": expected,
                            "actual": actual,
                        }
                    )

            truth_guesses = _truth_guess_events(truth_turn)
            parsed_guesses = _parsed_guesses(parsed_turn)
            if len(parsed_guesses) != len(truth_guesses):
                mismatches.append(
                    {
                        "kind": "guess_count",
                        "turn_index": turn_index,
                        "expected": len(truth_guesses),
                        "actual": len(parsed_guesses),
                    }
                )

            guess_count = max(len(truth_guesses), len(parsed_guesses))
            for guess_index in range(guess_count):
                truth_guess = truth_guesses[guess_index] if guess_index < len(truth_guesses) else None
                parsed_guess = parsed_guesses[guess_index] if guess_index < len(parsed_guesses) else None
                if truth_guess is None:
                    mismatches.append(
                        {
                            "kind": "unexpected_guess",
                            "turn_index": turn_index,
                            "guess_index": guess_index,
                            "actual": parsed_guess,
                        }
                    )
                    continue
                if parsed_guess is None:
                    mismatches.append(
                        {
                            "kind": "missing_guess",
                            "turn_index": turn_index,
                            "guess_index": guess_index,
                            "expected": truth_guess,
                        }
                    )
                    continue

                comparisons = (
                    ("word", _normalize_word),
                    ("card_color", _normalize_name),
                    ("result", _normalize_result),
                    ("selector_name", _normalize_name),
                )
                for field_name, normalizer in comparisons:
                    expected = truth_guess.get(field_name)
                    if expected in {None, ""}:
                        continue
                    actual = parsed_guess.get(field_name)
                    if normalizer(actual) != normalizer(expected):
                        mismatches.append(
                            {
                                "kind": "guess_field",
                                "turn_index": turn_index,
                                "guess_index": guess_index,
                                "field": field_name,
                                "expected": expected,
                                "actual": actual,
                            }
                        )

            truth_markers = _truth_turn_end_markers(truth_turn)
            if truth_markers:
                advisory.append(
                    {
                        "kind": "turn_end_markers_present_in_truth",
                        "turn_index": turn_index,
                        "count": len(truth_markers),
                    }
                )

        total_mismatches += len(mismatches)
        report_games[game_key] = {
            "status": "ok" if not mismatches else "mismatch",
            "mismatches": mismatches,
            "advisory": advisory,
        }

    return {
        "truth_source": truth_payload.get("source"),
        "total_mismatches": total_mismatches,
        "games": report_games,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Truth Comparison Report",
        "",
        f"Truth source: `{report.get('truth_source', 'unknown')}`",
        f"Total mismatches: `{report.get('total_mismatches', 0)}`",
        "",
    ]
    for game_key, game_report in report.get("games", {}).items():
        lines.append(f"## {game_key}")
        lines.append("")
        lines.append(f"Status: `{game_report.get('status', 'unknown')}`")
        mismatches = game_report.get("mismatches", [])
        advisory = game_report.get("advisory", [])
        if not mismatches:
            lines.append("- No clue/guess mismatches.")
        else:
            for mismatch in mismatches:
                kind = mismatch.get("kind")
                if kind == "winner_team":
                    lines.append(
                        f"- winner team: expected `{mismatch.get('expected')}`, actual `{mismatch.get('actual')}`"
                    )
                    continue
                if kind == "turn_count":
                    lines.append(
                        f"- turn count: expected `{mismatch.get('expected')}`, actual `{mismatch.get('actual')}`"
                    )
                    continue
                if kind == "turn_field":
                    lines.append(
                        f"- turn {mismatch.get('turn_index') + 1} {mismatch.get('field')}: expected `{mismatch.get('expected')}`, actual `{mismatch.get('actual')}`"
                    )
                    continue
                if kind == "guess_count":
                    lines.append(
                        f"- turn {mismatch.get('turn_index') + 1} guess count: expected `{mismatch.get('expected')}`, actual `{mismatch.get('actual')}`"
                    )
                    continue
                if kind == "guess_field":
                    lines.append(
                        f"- turn {mismatch.get('turn_index') + 1} guess {mismatch.get('guess_index') + 1} {mismatch.get('field')}: expected `{mismatch.get('expected')}`, actual `{mismatch.get('actual')}`"
                    )
                    continue
                if kind == "missing_guess":
                    lines.append(
                        f"- turn {mismatch.get('turn_index') + 1} missing guess {mismatch.get('guess_index') + 1}: expected `{mismatch.get('expected', {}).get('word')}`"
                    )
                    continue
                if kind == "unexpected_guess":
                    lines.append(
                        f"- turn {mismatch.get('turn_index') + 1} unexpected guess {mismatch.get('guess_index') + 1}: actual `{mismatch.get('actual', {}).get('word')}`"
                    )
                    continue
                if kind == "missing_turn":
                    lines.append(
                        f"- missing turn {mismatch.get('turn_index') + 1}"
                    )
                    continue
                if kind == "unexpected_turn":
                    lines.append(
                        f"- unexpected turn {mismatch.get('turn_index') + 1}"
                    )
                    continue
                if kind == "missing_game":
                    lines.append(f"- missing parsed game `{mismatch.get('expected_game')}`")
                    continue
                lines.append(f"- {json.dumps(mismatch, ensure_ascii=True)}")
        if advisory:
            lines.append("- Advisory:")
            for item in advisory:
                if item.get("kind") == "turn_end_markers_present_in_truth":
                    lines.append(
                        f"- truth includes `{item.get('count')}` green-check turn-end marker(s) in turn `{item.get('turn_index') + 1}`"
                    )
                else:
                    lines.append(f"- {json.dumps(item, ensure_ascii=True)}")
        lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-json", required=True)
    parser.add_argument("--parsed-json", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    truth_payload = json.loads(Path(args.truth_json).read_text(encoding="utf-8"))
    parsed_payload = json.loads(Path(args.parsed_json).read_text(encoding="utf-8"))
    report = compare_truth_to_parsed(
        truth_payload=truth_payload,
        parsed_payload=parsed_payload,
    )
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    markdown = render_markdown_report(report)
    if args.output_md:
        Path(args.output_md).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
