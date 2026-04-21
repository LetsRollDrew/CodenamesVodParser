import numpy as np

from app.parsers.selectors.attribution import _reference_identities_from_slots, _select_representative_crop


def test_select_representative_crop_prefers_sharpest_frame() -> None:
    blurry = np.full((24, 24, 3), 128, dtype=np.uint8)
    sharp = blurry.copy()
    sharp[:, ::2] = 255

    selected = _select_representative_crop([blurry, sharp], [0.99, 0.8])

    assert np.array_equal(selected, sharp)


def test_reference_identities_from_slots_uses_roster_order() -> None:
    section = np.zeros((60, 90, 3), dtype=np.uint8)
    section[:, :30] = (255, 0, 0)
    section[:, 30:60] = (0, 255, 0)
    section[:, 60:] = (0, 0, 255)

    identities = _reference_identities_from_slots(
        section=section,
        operative_names=["Drew", "luke", "Cherry"],
        team_color="blue",
        role="operative",
    )

    assert [identity.display_name for identity in identities] == ["Drew", "luke", "Cherry"]
    assert all(identity.avatar_hash for identity in identities)
    assert all(identity.avatar_image.size > 0 for identity in identities)
