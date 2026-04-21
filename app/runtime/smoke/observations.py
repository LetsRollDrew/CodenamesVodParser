"""Roster and board observation helpers for smoke parsing"""

from __future__ import annotations

from difflib import SequenceMatcher

from app.core.models import BoardCell, BoardState, PlayerRosterEntry
from app.vision.ocr.preprocessing import collapse_whitespace


def _roster_quality(roster: list[PlayerRosterEntry]) -> int:
    if not roster:
        return 0

    names = [player.display_name.strip() for player in roster if player.display_name.strip()]
    distinct_names = {name.casefold() for name in names}
    alnum_weight = sum(sum(1 for character in name if character.isalnum()) for name in names)
    length_weight = sum(min(len(name), 12) for name in names)
    return (len(names) * 20) + (len(distinct_names) * 10) + alnum_weight + length_weight


def _roster_is_usable(roster: list[PlayerRosterEntry]) -> bool:
    if not roster:
        return False
    names = [player.display_name.strip() for player in roster if player.display_name.strip()]
    informative_names = [
        name for name in names if len(name) > 1 and any(character.isalnum() for character in name)
    ]
    return len(informative_names) >= max(1, len(roster) // 2)


def _board_quality(board_state: object | None) -> int:
    if board_state is None:
        return 0
    score = 0
    for cell in getattr(board_state, "cells", []):
        word = str(getattr(cell, "word", "")).strip()
        if not word:
            continue
        score += 20
        if word.isalpha():
            score += 10
        if len(word) >= 3:
            score += 5
        score += min(int(float(getattr(cell, "confidence", 0.0)) * 10), 10)
    return score


def _board_is_usable(board_state: object | None) -> bool:
    if board_state is None:
        return False
    valid_words = sum(
        1
        for cell in getattr(board_state, "cells", [])
        if str(getattr(cell, "word", "")).strip().isalpha() and len(str(getattr(cell, "word", "")).strip()) >= 3
    )
    return valid_words >= 20


def _merge_rosters(
    current_roster: list[PlayerRosterEntry],
    candidate_roster: list[PlayerRosterEntry],
) -> list[PlayerRosterEntry]:
    if not current_roster:
        return list(candidate_roster)
    if not candidate_roster:
        return list(current_roster)

    remaining = list(candidate_roster)
    merged: list[PlayerRosterEntry] = []
    for current_entry in current_roster:
        best_index = None
        for index, candidate_entry in enumerate(remaining):
            if (
                candidate_entry.team_color == current_entry.team_color
                and candidate_entry.role == current_entry.role
                and candidate_entry.avatar_hash
                and candidate_entry.avatar_hash == current_entry.avatar_hash
            ):
                best_index = index
                break
        if best_index is None:
            for index, candidate_entry in enumerate(remaining):
                if candidate_entry.team_color == current_entry.team_color and candidate_entry.role == current_entry.role:
                    best_index = index
                    break
        if best_index is None:
            merged.append(current_entry)
            continue

        candidate_entry = remaining.pop(best_index)
        if _should_replace_roster_name(current_entry.display_name, candidate_entry.display_name):
            merged.append(candidate_entry)
        else:
            merged.append(current_entry)

    merged.extend(remaining)
    return merged


def _observe_roster(
    observations: dict[tuple[str, str, str], dict[str, tuple[int, int, PlayerRosterEntry]]],
    roster: list[PlayerRosterEntry],
) -> None:
    for entry in roster:
        identity = _find_matching_roster_identity(observations, entry) or _roster_identity(entry)
        bucket = observations.setdefault(identity, {})
        name_key = collapse_whitespace(entry.display_name).casefold()
        count, best_score, best_entry = bucket.get(
            name_key,
            (0, _player_name_quality(entry.display_name), entry),
        )
        score = _player_name_quality(entry.display_name)
        exemplar = entry if score >= best_score else best_entry
        bucket[name_key] = (count + 1, max(best_score, score), exemplar)


def _materialize_roster_observations(
    observations: dict[tuple[str, str, str], dict[str, tuple[int, int, PlayerRosterEntry]]],
) -> list[PlayerRosterEntry]:
    grouped_candidates: dict[tuple[str, str], list[tuple[int, int, PlayerRosterEntry]]] = {}
    for identity, bucket in observations.items():
        best_name = max(
            bucket.items(),
            key=lambda item: (item[1][0], item[1][1], _player_name_quality(item[1][2].display_name)),
        )[1]
        count, best_score, best_entry = best_name
        grouped_candidates.setdefault(identity[:2], []).append((count, best_score, best_entry))

    materialized: list[PlayerRosterEntry] = []
    role_limits = {"operative": 3, "spymaster": 1}
    for (team_color, role), candidates in grouped_candidates.items():
        limit = role_limits.get(role, len(candidates))
        ranked = sorted(
            candidates,
            key=lambda item: (item[0], item[1], _player_name_quality(item[2].display_name)),
            reverse=True,
        )
        materialized.extend(entry for _, _, entry in ranked[:limit])

    return sorted(
        materialized,
        key=lambda entry: (
            0 if entry.team_color.value == "blue" else 1,
            0 if entry.role.value == "operative" else 1,
            entry.display_name.casefold(),
        ),
    )


def _roster_identity(entry: PlayerRosterEntry) -> tuple[str, str, str]:
    return (
        entry.team_color.value,
        entry.role.value,
        entry.avatar_hash or collapse_whitespace(entry.display_name).casefold(),
    )


def _find_matching_roster_identity(
    observations: dict[tuple[str, str, str], dict[str, tuple[int, int, PlayerRosterEntry]]],
    entry: PlayerRosterEntry,
) -> tuple[str, str, str] | None:
    best_identity: tuple[str, str, str] | None = None
    best_avatar_distance_ratio: float | None = None
    best_name_similarity = 0.0

    for identity, bucket in observations.items():
        if identity[0] != entry.team_color.value or identity[1] != entry.role.value:
            continue

        exemplar = max(bucket.values(), key=lambda item: (item[0], item[1]))[2]
        avatar_distance, avatar_distance_ratio = _avatar_hash_distance(entry.avatar_hash, exemplar.avatar_hash)
        if avatar_distance is not None and avatar_distance_ratio is not None and avatar_distance_ratio <= 0.22:
            if best_avatar_distance_ratio is None or avatar_distance_ratio < best_avatar_distance_ratio:
                best_avatar_distance_ratio = avatar_distance_ratio
                best_identity = identity
            continue

        similarity = SequenceMatcher(
            a=collapse_whitespace(entry.display_name).casefold(),
            b=collapse_whitespace(exemplar.display_name).casefold(),
        ).ratio()
        if similarity >= 0.84 and similarity > best_name_similarity:
            best_name_similarity = similarity
            best_identity = identity

    return best_identity


def _avatar_hash_distance(left: str | None, right: str | None) -> tuple[int | None, float | None]:
    if not left or not right:
        return None, None
    try:
        common_bits = min(len(left), len(right)) * 4
        if common_bits <= 0:
            return None, None
        distance = (int(left, 16) ^ int(right, 16)).bit_count()
        return distance, distance / common_bits
    except ValueError:
        return None, None


def _roster_signature(roster: list[PlayerRosterEntry]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            entry.team_color.value,
            entry.role.value,
            entry.display_name.casefold(),
        )
        for entry in roster
    )


def _player_name_quality(name: str) -> int:
    normalized = collapse_whitespace(name).strip()
    if not normalized:
        return 0
    score = 0
    score += min(len(normalized), 12) * 5
    if normalized.isalpha():
        score += 15
    if normalized.islower():
        score += 8 if len(normalized) <= 3 else 5
    elif normalized.istitle():
        score += 4
    if any(character.isdigit() for character in normalized):
        score -= 10
    if len(normalized) == 1:
        score -= 10
    return score


def _should_replace_roster_name(current_name: str, candidate_name: str) -> bool:
    current_score = _player_name_quality(current_name)
    candidate_score = _player_name_quality(candidate_name)
    if candidate_score <= current_score:
        return False
    if current_score < 40:
        return True

    similarity = SequenceMatcher(
        a=collapse_whitespace(current_name).casefold(),
        b=collapse_whitespace(candidate_name).casefold(),
    ).ratio()
    if similarity >= 0.75:
        return candidate_score >= current_score + 10
    return candidate_score >= current_score + 6


def _merge_board_states(
    current_state: BoardState | None,
    candidate_state: BoardState | None,
) -> BoardState | None:
    if current_state is None:
        return candidate_state
    if candidate_state is None:
        return current_state

    best_cells: dict[tuple[int, int], BoardCell] = {
        (cell.row, cell.col): cell for cell in current_state.cells
    }
    for cell in candidate_state.cells:
        key = (cell.row, cell.col)
        existing = best_cells.get(key)
        if existing is None or _board_cell_quality(cell) > _board_cell_quality(existing):
            best_cells[key] = cell
    return BoardState(cells=[best_cells[key] for key in sorted(best_cells)])


def _observe_board_state(
    observations: dict[tuple[int, int], dict[str, tuple[int, int, BoardCell]]],
    board_state: BoardState,
) -> None:
    for cell in board_state.cells:
        key = (cell.row, cell.col)
        bucket = observations.setdefault(key, {})
        word_key = str(cell.word).strip().upper()
        count, best_score, best_cell = bucket.get(
            word_key,
            (0, _board_cell_quality(cell), cell),
        )
        score = _board_cell_quality(cell)
        exemplar = cell if score >= best_score else best_cell
        bucket[word_key] = (count + 1, max(best_score, score), exemplar)


def _materialize_board_observations(
    observations: dict[tuple[int, int], dict[str, tuple[int, int, BoardCell]]],
) -> BoardState | None:
    if not observations:
        return None
    cells: list[BoardCell] = []
    for key in sorted(observations):
        bucket = observations[key]
        ranked_cells = sorted(
            bucket.items(),
            key=lambda item: (
                _board_cell_observation_priority(item[1][2], observation_count=item[1][0]),
                item[1][0],
                item[1][1],
                len(item[1][2].word),
            ),
            reverse=True,
        )
        best_cell = ranked_cells[0][1][2]
        cells.append(best_cell)
    return BoardState(cells=cells)


def _board_signature(board_state: BoardState | None) -> tuple[str, ...]:
    if board_state is None:
        return ()
    return tuple(cell.word for cell in sorted(board_state.cells, key=lambda item: (item.row, item.col)))


def _board_cell_quality(cell: BoardCell) -> int:
    word = str(cell.word).strip().upper()
    if not word:
        return 0
    score = 0
    score += int(float(cell.confidence) * 100)
    if word.isalpha():
        score += 35
    if len(word) >= 3:
        score += 15
    if 3 <= len(word) <= 10:
        score += 10
    if any(character.isdigit() for character in word):
        score -= 15
    if len(word) <= 2:
        score -= 10
    vowels = sum(1 for character in word if character in "AEIOUY")
    vowel_ratio = vowels / len(word)
    if vowels == 0:
        score -= 20
    elif 0.20 <= vowel_ratio <= 0.65:
        score += 10
    else:
        score -= 5
    duplicate_runs = sum(1 for index in range(len(word) - 1) if word[index] == word[index + 1])
    score -= min(duplicate_runs * 5, 15)
    consonant_run = 0
    max_consonant_run = 0
    for character in word:
        if character.isalpha() and character not in "AEIOUY":
            consonant_run += 1
            max_consonant_run = max(max_consonant_run, consonant_run)
        else:
            consonant_run = 0
    if max_consonant_run >= 4:
        score -= 20
    elif max_consonant_run == 3:
        score -= 8
    if getattr(cell, "corrected_by_default_dictionary", False):
        score -= 12
        score += int(float(getattr(cell, "dictionary_similarity", 0.0)) * 8)
        score += int(float(getattr(cell, "dictionary_runner_up_gap", 0.0)) * 50)
    return score


def _board_cell_observation_priority(cell: BoardCell, *, observation_count: int) -> int:
    score = _board_cell_quality(cell)
    if getattr(cell, "corrected_by_default_dictionary", False) and observation_count < 2:
        score -= 40
    return score
