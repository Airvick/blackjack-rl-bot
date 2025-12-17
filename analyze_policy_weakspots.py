import json
import random
from collections import defaultdict
from pathlib import Path

from blackjack_engine import create_shoe, play_round, hand_value, can_split

# ===============================
# Config
# ===============================

BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "policy_out"

NUM_ROUNDS = 50000     # increase for more stable stats
NUM_DECKS = 6
BASE_BET = 1.0

MIN_VISITS = 50        # only report states seen at least this many times
TOP_K = 40             # how many worst states to print


# ===============================
# Load learned policies
# ===============================

with open(POLICY_DIR / "normal_policy.json", "r") as f:
    NORMAL_POLICY = json.load(f)

with open(POLICY_DIR / "split_policy.json", "r") as f:
    SPLIT_POLICY = json.load(f)


# ===============================
# Policy function (same as simulate)
# ===============================

def policy_from_tables(player_cards, dealer_up, ctx):
    total = ctx["player_total"]
    soft = ctx["player_soft"]
    can_spl = ctx["can_split"]
    can_dbl = ctx["can_double"]

    # 1) split: only if allowed and table suggests P
    if can_spl:
        pair_rank = player_cards[0]
        sp_key = f"{int(pair_rank)}_{int(dealer_up)}"
        sp = SPLIT_POLICY.get(sp_key)
        if sp and sp["action"] == "P":
            return "P"

    # 2) normal: use learned best action
    norm_key = f"{int(total)}_{int(bool(soft))}_{int(dealer_up)}"
    np = NORMAL_POLICY.get(norm_key)
    if np:
        action = np["action"]
        if action == "D" and not can_dbl:
            return "H"
        if action == "P" and not can_spl:
            return "H"
        return action

    # 3) fallback heuristic
    if total >= 17:
        return "S"
    if total <= 8:
        return "H"
    if 9 <= total <= 11 and can_dbl:
        return "D"
    return "H"


# ===============================
# Weak spot analysis
# ===============================

def main():
    random.seed(123)
    shoe = create_shoe(NUM_DECKS)

    # stats[(total, soft, dealer_up, action)] = {"sum": ..., "count": ...}
    stats = defaultdict(lambda: {"sum": 0.0, "count": 0})

    for i in range(NUM_ROUNDS):
        if len(shoe) < 52:
            shoe = create_shoe(NUM_DECKS)

        round_state = play_round(
            shoe=shoe,
            policy=policy_from_tables,
            base_bet=BASE_BET,
            log=True,  # important: we want step-by-step actions
        )

        # For each hand, we collect its final result,
        # and assign that result to every decision made on that hand.
        for hand_index, hand in enumerate(round_state.player_hands):
            if hand.result is None:
                continue

            hand_profit = hand.result  # relative to base bet

            # find all logged steps for this hand_index
            for step in round_state.logs.get("steps", []):
                if step["hand_index"] != hand_index:
                    continue

                total = step["ctx"]["player_total"]
                soft = bool(step["ctx"]["player_soft"])
                dealer_up = int(step["dealer_up"])
                action = step["action"]

                key = (total, soft, dealer_up, action)
                stats[key]["sum"] += hand_profit
                stats[key]["count"] += 1

    # build list for analysis
    rows = []
    for (total, soft, dealer_up, action), agg in stats.items():
        c = agg["count"]
        if c < MIN_VISITS:
            continue
        avg = agg["sum"] / c
        rows.append({
            "total": total,
            "soft": soft,
            "dealer_up": dealer_up,
            "action": action,
            "count": c,
            "avg_profit": avg,
        })

    # sort by avg_profit ascending: worst spots first
    rows.sort(key=lambda r: r["avg_profit"])

    print("=== Potential weak spots (min visits = {}, worst first) ===".format(MIN_VISITS))
    for r in rows[:TOP_K]:
        hand_type = "soft" if r["soft"] else "hard"
        print(
            f"Total {r['total']:>2} ({hand_type}) vs dealer {r['dealer_up']:>2} "
            f"-> action {r['action']} | visits={r['count']:>4} | avg_profit={r['avg_profit']:.4f}"
        )


if __name__ == "__main__":
    main()
