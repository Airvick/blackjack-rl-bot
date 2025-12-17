import json
import ast
import csv
from pathlib import Path

# ===============================
# Paths & policy loading
# ===============================

BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "policy_out"
LOG_PATH = BASE_DIR / "live_sessions_log.csv"

with open(POLICY_DIR / "normal_policy.json", "r") as f:
    NORMAL_POLICY = json.load(f)

with open(POLICY_DIR / "split_policy.json", "r") as f:
    SPLIT_POLICY = json.load(f)


# ===============================
# Smart fallback
# ===============================

def smart_fallback(player_value, player_soft):
    """
    If there is no exact (player_value, soft, dealer_up) match,
    choose the action with highest expected_reward among all states
    with the same (player_value, player_soft).
    """
    prefix = f"{int(player_value)}_{int(bool(player_soft))}_"
    best_action = None
    best_er = -1e9

    for key, val in NORMAL_POLICY.items():
        if key.startswith(prefix):
            er = float(val["expected_reward"])
            if er > best_er:
                best_er = er
                best_action = val["action"]

    if best_action is not None:
        return best_action, best_er

    # last resort heuristic
    fallback_action = "H" if player_value < 12 else "S"
    return fallback_action, 0.0


# ===============================
# Hand parsing & features
# ===============================

def parse_hand_input(hand_str):
    """
    Parse user input like:
      '10,6'
      'A,7'
      '[10, 6]'
    into a list of numeric values (Ace = 11).
    """
    hand_str = hand_str.strip()

    # allow list-style input
    if hand_str.startswith("["):
        hand = ast.literal_eval(hand_str)
    else:
        parts = [p.strip() for p in hand_str.split(",")]
        hand = []
        for p in parts:
            if p.upper() == "A":
                hand.append(11)
            else:
                hand.append(int(p))
    return hand


def hand_features(hand):
    """
    Compute:
      total value (with Ace adjustment),
      is_soft,
      is_pair,
      pair_rank
    """
    if isinstance(hand, str):
        hand = ast.literal_eval(hand)

    if not isinstance(hand, (list, tuple)) or len(hand) == 0:
        return None, None, False, None

    values = []
    ace_count = 0

    for c in hand:
        v = int(c)
        if v == 11:
            ace_count += 1
            values.append(11)
        else:
            values.append(v)

    total = sum(values)

    # adjust Aces from 11 to 1 while busting
    aces_as_11 = ace_count
    while total > 21 and aces_as_11 > 0:
        total -= 10
        aces_as_11 -= 1

    is_soft = aces_as_11 > 0 and total <= 21
    is_pair = len(values) == 2 and values[0] == values[1]
    pair_rank = values[0] if is_pair else None

    return total, is_soft, is_pair, pair_rank


# ===============================
# Logging
# ===============================

def log_decision(hand_in, dealer_in, rec, result=None):
    """
    Log:
      - input hand & dealer
      - derived state
      - recommended action
      - source
      - expected_reward
      - (optional) result: 'w','l','p'
    """
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "hand_input",
                "dealer_up_input",
                "player_value",
                "player_soft",
                "is_pair",
                "pair_rank",
                "dealer_up",
                "action",
                "expected_reward",
                "source",
                "result",
            ])
        writer.writerow([
            hand_in,
            dealer_in,
            rec["player_value"],
            rec["player_soft"],
            rec["is_pair"],
            rec["pair_rank"],
            rec["dealer_up"],
            rec["action"],
            rec["expected_reward"],
            rec["source"],
            result,
        ])



# ===============================
# Recommender
# ===============================

def recommend_action(initial_hand_str, dealer_up_str):
    # parse input
    hand = parse_hand_input(initial_hand_str)

    try:
        dealer_up = int(dealer_up_str)
    except ValueError:
        raise ValueError("Dealer upcard must be an integer between 2 and 11 (11 = Ace).")

    player_value, player_soft, is_pair, pair_rank = hand_features(hand)

    if player_value is None:
        raise ValueError("Invalid hand.")

    # 1) split policy (only for real pairs)
    if is_pair and pair_rank is not None:
        sp_key = f"{int(pair_rank)}_{dealer_up}"
        sp = SPLIT_POLICY.get(sp_key)
        if sp:
            return {
                "player_value": int(player_value),
                "player_soft": bool(player_soft),
                "is_pair": True,
                "pair_rank": int(pair_rank),
                "dealer_up": dealer_up,
                "action": sp["action"],
                "expected_reward": float(sp["expected_reward"]),
                "source": "split_policy",
            }

    # 2) normal policy lookup
    norm_key = f"{int(player_value)}_{int(bool(player_soft))}_{dealer_up}"
    np = NORMAL_POLICY.get(norm_key)
    if np:
        return {
            "player_value": int(player_value),
            "player_soft": bool(player_soft),
            "is_pair": bool(is_pair),
            "pair_rank": int(pair_rank) if is_pair else None,
            "dealer_up": dealer_up,
            "action": np["action"],
            "expected_reward": float(np["expected_reward"]),
            "source": "normal_policy",
        }

    # 3) smart fallback by (total, soft)
    fb_action, fb_er = smart_fallback(player_value, player_soft)
    return {
        "player_value": int(player_value),
        "player_soft": bool(player_soft),
        "is_pair": bool(is_pair),
        "pair_rank": int(pair_rank) if is_pair else None,
        "dealer_up": dealer_up,
        "action": fb_action,
        "expected_reward": float(fb_er),
        "source": "smart_fallback",
        "note": "State not found; used best action for this total/softness.",
    }


# ===============================
# CLI
# ===============================

def cli():
    print("=== Blackjack Policy CLI (trained on full dataset) ===")
    print("Enter your hand like: 8,8  or  A,7  or  10,6")
    print("Dealer upcard: 2-11 (where 11 = Ace)")
    print("Type 'q' to quit.")

    while True:
        hand_in = input("\nYour hand (or 'q' to quit): ").strip()
        if hand_in.lower() in ("q", "quit", "exit"):
            break

        dealer_in = input("Dealer upcard (2-11, or 'q' to quit): ").strip()
        if dealer_in.lower() in ("q", "quit", "exit"):
            break

        try:
            rec = recommend_action(hand_in, dealer_in)
        except Exception as e:
            print(f"Error: {e}")
            continue

        state_desc = (
            f"value={rec['player_value']} "
            f"{'(soft)' if rec['player_soft'] else '(hard)'} "
            f"{'(pair ' + str(rec['pair_rank']) + ')' if rec['is_pair'] else ''}"
        )

        print(f"State: {state_desc}, dealer_up={rec['dealer_up']}")
        print(
            f"→ Recommended action: {rec['action']} "
            f"(ER={rec['expected_reward']:.4f}, source={rec['source']})"
        )

        # Megkérdezzük: vége van-e a leosztásnak?
        finished = input("Is this hand finished now? (y = yes, n = no): ").strip().lower()

        result_in = None
        if finished == "y":
            r = input("Result? (w=win, l=lose, p=push, Enter=skip): ").strip().lower()
            if r in ("w", "l", "p"):
                result_in = r

        # Logolunk (ha még megy tovább a leosztás, result=None marad)
        log_decision(hand_in, dealer_in, rec, result_in)




if __name__ == "__main__":
    cli()
