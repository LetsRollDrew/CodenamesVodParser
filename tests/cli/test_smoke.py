from app.cli.smoke import parse_time_offset


def test_parse_time_offset_accepts_hms_and_seconds() -> None:
    assert parse_time_offset("394") == 394.0
    assert parse_time_offset("00:06:34") == 394.0
