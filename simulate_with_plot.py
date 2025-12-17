import json
import random
import matplotlib.pyplot as plt
from pathlib import Path

from blackjack_engine import create_shoe, play_round

# ===============================
# Config
# ===============================

BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "policy_out"

NUM_ROUNDS = 5000   # for quick plotting; you can increase to 10000+
NUM_DECKS = 6
BASE_BET = 1.0

# ===============================
# Load policies
# ===============================

with open(POLICY_DIR / "normal_policy.json", "r") as f:
    NORMAL_POLICY = json.load(f)

with open(POLICY_DIR / "split_policy.json", "r") as f:
    SPLIT_POLICY = json.load(f)


# ===============================
# Policy function
# ===============================

def policy_from_tables(player_cards, dealer_up, ctx):
    total = ctx["player_total"]
    soft = ctx["player_soft"]
    can_spl = ctx["can_split"]
    can_dbl = ctx["can_double"]

    # split logic
    if can_spl:
        pair_rank = player_cards[0]
        sp_key = f"{int(pair_rank)}_{int(dealer_up)}"
        sp = SPLIT_POLICY.get(sp_key)
        if sp and sp["action"] == "P":
            return "P"

    # normal policy
    norm_key = f"{int(total)}_{int(bool(soft))}_{int(dealer_up)}"
    np = NORMAL_POLICY.get(norm_key)
    if np:
        action = np["action"]
        if action == "D" and not can_dbl:
            return "H"
        if action == "P" and not can_spl:
            return "H"
        return action

    # fallback
    if total >= 17:
        return "S"
    if total <= 8:
        return "H"
    if 9 <= total <= 11 and can_dbl:
        return "D"
    return "H"


# ===============================
# Simulation + Plotting
# ===============================

def main():
    random.seed(42)
    shoe = create_shoe(NUM_DECKS)

    total_profit = 0.0
    profits = []

    for i in range(NUM_ROUNDS):
        if len(shoe) < 52:
            shoe = create_shoe(NUM_DECKS)

        round_state = play_round(shoe, policy_from_tables, base_bet=BASE_BET)

        round_profit = 0.0
        for hand in round_state.player_hands:
            if hand.result is not None:
                round_profit += hand.result

        total_profit += round_profit
        profits.append(total_profit)

    # ---- Plot ----
    plt.figure(figsize=(10, 5))
    plt.plot(profits, color="gold", linewidth=2)
    plt.title(f"Blackjack Bot Profit Curve ({NUM_ROUNDS} Rounds)")
    plt.xlabel("Rounds Played")
    plt.ylabel("Cumulative Profit (units)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("profit_curve.png")
    plt.show()

    print("Saved plot as profit_curve.png")


if __name__ == "__main__":
    main()
