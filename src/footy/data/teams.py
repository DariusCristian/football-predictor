"""Team name normalization across data sources.

Both sources map into a canonical name. Unknown names raise, rather than
being silently dropped — a missing team would corrupt the model invisibly.
"""

# OpenFootball (historical JSON) -> canonical
OPENFOOTBALL_TO_CANONICAL = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton Hove",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich Town",
    "Leeds United FC": "Leeds United",
    "Leicester City FC": "Leicester City",
    "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton Town",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Norwich City FC": "Norwich City",
    "Nottingham Forest FC": "Nottingham",
    "Sheffield United FC": "Sheffield United",
    "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "Watford FC": "Watford",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}

# football-data.org shortName -> canonical (identity for current teams,
# kept explicit so a renamed shortName fails loudly instead of silently)
API_TO_CANONICAL = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton Hove": "Brighton Hove",
    "Chelsea": "Chelsea",
    "Coventry City": "Coventry City",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull City": "Hull City",
    "Ipswich Town": "Ipswich Town",
    "Leeds United": "Leeds United",
    "Liverpool": "Liverpool",
    "Man City": "Man City",
    "Man United": "Man United",
    "Newcastle": "Newcastle",
    "Nottingham": "Nottingham",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
}

CANONICAL_TEAMS = set(OPENFOOTBALL_TO_CANONICAL.values()) | set(API_TO_CANONICAL.values())


class UnknownTeamError(KeyError):
    pass


def _normalize(name: str, mapping: dict[str, str], source: str) -> str:
    try:
        return mapping[name]
    except KeyError:
        raise UnknownTeamError(
            f"Unknown {source} team name: {name!r}. "
            f"Add it to {source.upper()}_TO_CANONICAL in footy/data/teams.py"
        ) from None


def from_openfootball(name: str) -> str:
    return _normalize(name, OPENFOOTBALL_TO_CANONICAL, "openfootball")


def from_api(name: str) -> str:
    return _normalize(name, API_TO_CANONICAL, "api")