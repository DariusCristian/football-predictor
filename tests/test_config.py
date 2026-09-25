from footy.config import API_BASE, COMPETITION


def test_api_config():
    assert API_BASE.startswith("https://")
    assert COMPETITION == "PL"