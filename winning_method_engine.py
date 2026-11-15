"""
The Winning Method - value engine (API-Football + Poisson + 10,000 simulations)

Usage:
    pip install requests numpy
    export API_FOOTBALL_KEY="your_key"          # from dashboard.api-football.com
    python winning_method_engine.py daily        # today's top value picks
    python winning_method_engine.py daily 2026-11-21
    python winning_method_engine.py check 1234567  # pre-match check (lineups, injuries) ~30 min before kickoff
    python winning_method_engine.py demo         # test run with fake data, no key needed
"""

import csv
import json
import math
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import requests

# ---------------- settings ----------------
API_KEY = os.getenv("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"

LEAGUES = {          # league_id: name  (add/remove as you like)
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
    383: "Ligat Ha'Al",
}
SEASON = 2026          # season start year (2026 = 2026/27)
BOOKMAKER_ID = 8       # 8 = Bet365
MIN_ODDS = 1.8
MIN_EDGE = 0.05        # at least 5% expected value
MAX_PICKS = 3
LAST_N_MATCHES = 10    # form window per team
SIMULATIONS = 10_000
MAX_GOALS = 10

DAILY_REQUEST_BUDGET = int(os.getenv("DAILY_REQUEST_BUDGET", "90"))  # free plan = 100/day, keep a margin
_requests_made = 0

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# ---------------- API layer (with daily cache to save requests) ----------------
def api_get(endpoint, params):
    key = endpoint.strip("/").replace("/", "_") + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    cache_file = CACHE_DIR / f"{date.today()}_{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    global _requests_made
    if not API_KEY:
        sys.exit("Missing API_FOOTBALL_KEY environment variable.")
    if _requests_made >= DAILY_REQUEST_BUDGET:
        raise RuntimeError("daily request budget reached")
    _requests_made += 1
    r = requests.get(BASE_URL + endpoint, headers={"x-apisports-key": API_KEY}, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        raise RuntimeError(f"API error on {endpoint}: {data['errors']}")
    cache_file.write_text(json.dumps(data["response"]))
    time.sleep(0.4)  # stay under rate limit
    return data["response"]


def get_fixtures(day):
    fixtures = []
    for league_id in LEAGUES:
        fixtures += api_get("/fixtures", {"date": day, "league": league_id, "season": SEASON})
    return [f for f in fixtures if f["fixture"]["status"]["short"] == "NS"]


def get_team_history(team_id):
    return api_get("/fixtures", {"team": team_id, "last": LAST_N_MATCHES, "status": "FT"})


def get_odds(fixture_id):
    resp = api_get("/odds", {"fixture": fixture_id, "bookmaker": BOOKMAKER_ID})
    if not resp:
        return {}
    odds = {}
    for bet in resp[0]["bookmakers"][0]["bets"]:
        name = bet["name"]
        for v in bet["values"]:
            if name == "Match Winner":
                odds[{"Home": "1", "Draw": "X", "Away": "2"}[v["value"]]] = float(v["odd"])
            elif name == "Goals Over/Under" and v["value"] in ("Over 2.5", "Under 2.5"):
                odds[v["value"]] = float(v["odd"])
            elif name == "Both Teams Score":
                odds["BTTS " + v["value"]] = float(v["odd"])
    return odds


# ---------------- model ----------------
def team_rates(history, team_id, venue):
    """Weighted avg goals scored/conceded, recent matches weigh more, venue matches x1.5."""
    scored = conceded = weights = 0.0
    for i, m in enumerate(sorted(history, key=lambda x: x["fixture"]["timestamp"], reverse=True)):
        is_home = m["teams"]["home"]["id"] == team_id
        gf = m["goals"]["home"] if is_home else m["goals"]["away"]
        ga = m["goals"]["away"] if is_home else m["goals"]["home"]
        if gf is None or ga is None:
            continue
        w = 0.9 ** i
        if (venue == "home") == is_home:
            w *= 1.5
        scored += gf * w
        conceded += ga * w
        weights += w
    if weights == 0:
        return 1.3, 1.3
    return scored / weights, conceded / weights


def expected_goals(home_hist, away_hist, home_id, away_id):
    h_for, h_against = team_rates(home_hist, home_id, "home")
    a_for, a_against = team_rates(away_hist, away_id, "away")
    lam_home = max(0.2, (h_for + a_against) / 2 * 1.05)  # small home advantage
    lam_away = max(0.2, (a_for + h_against) / 2 * 0.95)
    return lam_home, lam_away


def poisson_probs(lam_h, lam_a):
    ph = [math.exp(-lam_h) * lam_h ** k / math.factorial(k) for k in range(MAX_GOALS + 1)]
    pa = [math.exp(-lam_a) * lam_a ** k / math.factorial(k) for k in range(MAX_GOALS + 1)]
    m = np.outer(ph, pa)
    i, j = np.indices(m.shape)
    return {
        "1": m[i > j].sum(), "X": m[i == j].sum(), "2": m[i < j].sum(),
        "Over 2.5": m[i + j > 2].sum(), "Under 2.5": m[i + j <= 2].sum(),
        "BTTS Yes": m[(i > 0) & (j > 0)].sum(), "BTTS No": m[(i == 0) | (j == 0)].sum(),
    }


def simulate(lam_h, lam_a, n=SIMULATIONS, seed=None):
    rng = np.random.default_rng(seed)
    h = rng.poisson(lam_h, n)
    a = rng.poisson(lam_a, n)
    return {
        "1": np.mean(h > a), "X": np.mean(h == a), "2": np.mean(h < a),
        "Over 2.5": np.mean(h + a > 2), "Under 2.5": np.mean(h + a <= 2),
        "BTTS Yes": np.mean((h > 0) & (a > 0)), "BTTS No": np.mean((h == 0) | (a == 0)),
    }


def find_value(match_name, kickoff, fixture_id, lam_h, lam_a, odds):
    exact = poisson_probs(lam_h, lam_a)
    sim = simulate(lam_h, lam_a)
    bets = []
    for market, odd in odds.items():
        if market not in exact or odd < MIN_ODDS:
            continue
        p = (exact[market] + sim[market]) / 2
        edge = p * odd - 1
        if edge >= MIN_EDGE:
            bets.append({
                "fixture_id": fixture_id, "match": match_name, "kickoff": kickoff,
                "market": market, "odds": odd, "model_prob": round(p, 3),
                "fair_odds": round(1 / p, 2), "edge": round(edge, 3),
                "xg_home": round(lam_h, 2), "xg_away": round(lam_a, 2),
            })
    return bets


# ---------------- commands ----------------
def run_daily(day):
    fixtures = get_fixtures(day)
    print(f"{len(fixtures)} upcoming matches on {day}")
    all_bets = []
    for f in fixtures:
        h, a = f["teams"]["home"], f["teams"]["away"]
        try:
            odds = get_odds(f["fixture"]["id"])
            if not odds:
                continue
            lam_h, lam_a = expected_goals(get_team_history(h["id"]), get_team_history(a["id"]), h["id"], a["id"])
            all_bets += find_value(f"{h['name']} - {a['name']}", f["fixture"]["date"], f["fixture"]["id"], lam_h, lam_a, odds)
        except Exception as e:
            print(f"  skipped {h['name']} - {a['name']}: {e}")
    report(all_bets, day)


def report(all_bets, day):
    best = {}
    for b in sorted(all_bets, key=lambda x: x["edge"], reverse=True):
        best.setdefault(b["match"], b)  # one bet per match
    picks = list(best.values())[:MAX_PICKS]

    if not picks:
        print("No value bets today. Skipping is also a decision.")
        return
    print(f"(API requests used this run: {_requests_made})")
    print(f"\nTop {len(picks)} picks:")
    for p in picks:
        print(f"  {p['match']} | {p['market']} @ {p['odds']} | model {p['model_prob']:.0%} "
              f"(fair {p['fair_odds']}) | edge {p['edge']:+.1%} | id {p['fixture_id']}")

    out_dir = Path("picks")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"picks_{day}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=picks[0].keys())
        w.writeheader()
        w.writerows(picks)
    print(f"\nSaved to {out}")


def run_check(fixture_id):
    """Pre-match check: lineups, injuries, referee. Run ~30 min before kickoff."""
    fx = api_get("/fixtures", {"id": fixture_id})[0]
    print(f"{fx['teams']['home']['name']} - {fx['teams']['away']['name']}")
    print(f"Referee: {fx['fixture'].get('referee') or 'not announced'}")
    print(f"Venue: {fx['fixture']['venue']['name']}")

    lineups = api_get("/fixtures/lineups", {"fixture": fixture_id})
    if not lineups:
        print("Lineups not published yet - try again closer to kickoff.")
    for team in lineups:
        print(f"\n{team['team']['name']} ({team['formation']}) - coach {team['coach']['name']}")
        print("  " + ", ".join(p["player"]["name"] for p in team["startXI"]))

    injuries = api_get("/injuries", {"fixture": fixture_id})
    if injuries:
        print("\nMissing players:")
        for inj in injuries:
            print(f"  {inj['team']['name']}: {inj['player']['name']} ({inj['player']['reason']})")


def run_demo():
    """Fake matches to test the model without an API key."""
    demo = [
        ("Maccabi Haifa - Hapoel Be'er Sheva", 1.9, 1.1, {"1": 2.10, "X": 3.40, "2": 3.30, "Over 2.5": 1.85, "Under 2.5": 1.95}),
        ("Arsenal - Chelsea", 1.6, 1.3, {"1": 1.95, "X": 3.60, "2": 3.90, "Over 2.5": 1.80, "Under 2.5": 2.00, "BTTS Yes": 1.80}),
        ("Getafe - Valencia", 0.9, 0.8, {"1": 2.40, "X": 2.90, "2": 3.30, "Over 2.5": 2.40, "Under 2.5": 1.55}),
    ]
    bets = []
    for i, (name, lh, la, odds) in enumerate(demo):
        bets += find_value(name, "demo", i, lh, la, odds)
    report(bets, "demo")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if cmd == "daily":
        run_daily(sys.argv[2] if len(sys.argv) > 2 else str(date.today()))
    elif cmd == "check":
        run_check(sys.argv[2])
    elif cmd == "demo":
        run_demo()
    else:
        print(__doc__)
