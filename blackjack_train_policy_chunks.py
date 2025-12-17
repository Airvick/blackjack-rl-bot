import pandas as pd
import os
import ast
import json
from pathlib import Path

# ===============================
# Config
# ===============================

DATASET_PATH = os.path.expanduser("~/blackjack_bot/blackjack_simulator.csv")
CHUNK_SIZE = 1_000_000
VALID_ACTIONS = {"H", "S", "D", "P", "R"}


# ===============================
# Helper functions
# ===============================

def hand_features(hand):
    if isinstance(hand, str):
        try:
            hand = ast.literal_eval(hand)
        except (SyntaxError, ValueError):
            return None, None, False, None

    if not isinstance(hand, (list, tuple)) or len(hand) == 0:
        return None, None, False, None

    values = []
    ace_count = 0

    for c in hand:
        try:
            v = int(c)
        except (TypeError, ValueError):
            return None, None, False, None

        if v == 11:
            ace_count += 1
            values.append(11)
        else:
            values.append(v)

    total = sum(values)

    aces_as_11 = ace_count
    while total > 21 and aces_as_11 > 0:
        total -= 10
        aces_as_11 -= 1

    player_soft = aces_as_11 > 0 and total <= 21

    is_pair = len(values) == 2 and values[0] == values[1]
    pair_rank = values[0] if is_pair else None

    return total, player_soft, is_pair, pair_rank


def extract_first_action(actions):
    if isinstance(actions, str):
        try:
            actions = ast.literal_eval(actions)
        except (SyntaxError, ValueError):
            return None

    if not actions or not isinstance(actions, (list, tuple)):
        return None

    first_hand = actions[0]
    if not first_hand or not isinstance(first_hand, (list, tuple)):
        return None

    first_action = str(first_hand[0]).upper().strip()
    return first_action if first_action in VALID_ACTIONS else None


def aggregate_chunk(chunk):
    # features
    (
        chunk["player_value"],
        chunk["player_soft"],
        chunk["is_pair"],
        chunk["pair_rank"],
    ) = zip(*chunk["initial_hand"].apply(hand_features))

    # first action
    chunk["final_action"] = chunk["actions_taken"].apply(extract_first_action)

    # reward
    chunk["reward"] = chunk["win"]

    # keep only valid actions
    valid = chunk[chunk["final_action"].isin(VALID_ACTIONS)].copy()

    # masks
    normal_mask = (~valid["is_pair"]) | (
        (valid["is_pair"]) & (valid["final_action"] != "P")
    )
    pair_mask = (valid["is_pair"]) & (valid["final_action"] == "P")

    # aggregate normal: sum + count
    normal = (
        valid[normal_mask]
        .groupby(["player_value", "player_soft", "dealer_up", "final_action"])["reward"]
        .agg(["sum", "count"])
    )

    # aggregate split/pair: sum + count
    pair = (
        valid[pair_mask]
        .groupby(["pair_rank", "dealer_up", "final_action"])["reward"]
        .agg(["sum", "count"])
    )

    return normal, pair


def merge_agg(global_df, new_df):
    if global_df is None:
        return new_df.copy()
    merged = global_df.add(new_df, fill_value=0)
    merged["sum"] = merged["sum"].astype(float)
    merged["count"] = merged["count"].astype(int)
    return merged


# ===============================
# Main
# ===============================

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"File not found at: {DATASET_PATH}")

    normal_global = None
    pair_global = None

    for i, chunk in enumerate(pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE)):
        n, p = aggregate_chunk(chunk)
        normal_global = merge_agg(normal_global, n)
        pair_global = merge_agg(pair_global, p)
        print(f"Processed chunk {i+1}")

    # compute averages
    normal_global["avg"] = normal_global["sum"] / normal_global["count"]
    pair_global["avg"] = pair_global["sum"] / pair_global["count"]

    # build greedy normal policy
    normal_policy = {}
    for (pv, ps, du), sub in normal_global.groupby(
        level=["player_value", "player_soft", "dealer_up"]
    ):
        best_idx = sub["avg"].idxmax()
        best_action = best_idx[-1]  # final_action
        best_reward = float(sub.loc[best_idx]["avg"])
        key = f"{int(pv)}_{int(ps)}_{int(du)}"
        normal_policy[key] = {
            "action": best_action,
            "expected_reward": best_reward,
        }

    # build greedy split policy
    split_policy = {}
    for (pr, du), sub in pair_global.groupby(
        level=["pair_rank", "dealer_up"]
    ):
        best_idx = sub["avg"].idxmax()
        best_action = best_idx[-1]
        best_reward = float(sub.loc[best_idx]["avg"])
        key = f"{int(pr)}_{int(du)}"
        split_policy[key] = {
            "action": best_action,
            "expected_reward": best_reward,
        }

    outdir = Path("policy_out")
    outdir.mkdir(exist_ok=True)

    with open(outdir / "normal_policy.json", "w") as f:
        json.dump(normal_policy, f)

    with open(outdir / "split_policy.json", "w") as f:
        json.dump(split_policy, f)

    print("Saved policies to policy_out/normal_policy.json and split_policy.json")


if __name__ == "__main__":
    main()
