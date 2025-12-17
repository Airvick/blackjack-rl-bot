import json
import random
from pathlib import Path

from blackjack_engine import create_shoe, play_round, hand_value, can_split

# ===============================
# Config
# ===============================

BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "policy_out"

NUM_ROUNDS = 10000000       # you can increase this (e.g. 100000)
NUM_DECKS = 6            # should match your engine config
BASE_BET = 1.0


# ===============================
# Load learned policies
# ===============================

with open(POLICY_DIR / "normal_policy.json", "r") as f:
    NORMAL_POLICY = json.load(f)

with open(POLICY_DIR / "split_policy.json", "r") as f:
    SPLIT_POLICY = json.load(f)


# ===============================
# Helper: map state -> action using learned tables
# ===============================

def policy_from_tables(player_cards, dealer_up, ctx):
    """
    Map current state to an action using:
      - split_policy for valid splits
      - normal_policy for totals
      - simple fallback otherwise

    player_cards: list of ints (2-11, Ace=11)
    dealer_up: int (2-11)
    ctx: extra info from engine:
         {
            "player_total",
            "player_soft",
            "can_split",
            "can_double",
            "hand_index",
            "num_hands"
         }
    Returns: one of {"H", "S", "D", "P", "R"}
    """

    total = ctx["player_total"]
    soft = ctx["player_soft"]
    can_spl = ctx["can_split"]
    can_dbl = ctx["can_double"]

    # 1) Split logic: only if allowed and we have split policy info
    if can_spl:
        pair_rank = player_cards[0]
        sp_key = f"{int(pair_rank)}_{int(dealer_up)}"
        sp = SPLIT_POLICY.get(sp_key)
        if sp and sp["action"] == "P":
            return "P"

    # 2) Normal policy lookup for this (total, soft, dealer_up)
    norm_key = f"{int(total)}_{int(bool(soft))}_{int(dealer_up)}"
    np = NORMAL_POLICY.get(norm_key)
    if np:
        action = np["action"]
        # enforce some basic rule sanity with engine context
        if action == "D" and not can_dbl:
            # if cannot double here, convert to hit
            return "H"
        if action == "P" and not can_spl:
            # if cannot split, convert to hit
            return "H"
        return action

    # 3) Fallback:
    # if no entry in tables, use a simple heuristic
    if total >= 17:
        return "S"
    if total <= 8:
        return "H"
    if 9 <= total <= 11 and can_dbl:
        return "D"
    return "H"


# ===============================
# Simulation
# ===============================

def main():
    random.seed(42)

    shoe = create_shoe(NUM_DECKS)

    total_profit = 0.0
    rounds_played = 0

    for i in range(NUM_ROUNDS):
        # reshuffle if shoe is low
        if len(shoe) < 52:
            shoe = create_shoe(NUM_DECKS)

        round_state = play_round(
            shoe=shoe,
            policy=policy_from_tables,
            base_bet=BASE_BET,
            log=False,
        )

        # sum results from all hands (splits)
        round_profit = 0.0
        for hand in round_state.player_hands:
            if hand.result is None:
                continue
            round_profit += hand.result

        total_profit += round_profit
        rounds_played += 1

    avg_per_round = total_profit / rounds_played if rounds_played > 0 else 0.0

    print("=== Simulation summary ===")
    print(f"Rounds played: {rounds_played}")
    print(f"Total profit: {total_profit:.4f} units")
    print(f"Average profit per round: {avg_per_round:.6f} units")
    print(f"Average profit per 100 rounds: {avg_per_round * 100:.4f} units")


if __name__ == "__main__":
    main()
