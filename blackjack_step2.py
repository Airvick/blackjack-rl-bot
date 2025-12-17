import pandas as pd
import os
import ast

# ===============================
# Config & dataset loading
# ===============================

DATASET_PATH = os.path.expanduser("~/blackjack_bot/blackjack_simulator.csv")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"File not found at: {DATASET_PATH}")

print(f"Loaded file: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

print("\nColumns:")
print(df.columns.tolist())

# Use a sample for prototyping (full dataset will be handled later via chunked script / Slurm)
sample = df.head(100000).copy()

# ===============================
# Hand feature extraction
# ===============================

def hand_features(hand):
    """
    Parse initial_hand into:
      - player_value: total value after Ace adjustment
      - player_soft: True if at least one Ace is counted as 11 without bust
      - is_pair: True if exactly 2 cards of same rank
      - pair_rank: numeric value of the pair (e.g. 8 for [8, 8])
    """
    # Allow string representation like "[10, 11]"
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

        if v == 11:  # Ace encoded as 11
            ace_count += 1
            values.append(11)
        else:
            values.append(v)

    total = sum(values)

    # Adjust Aces from 11 to 1 while busting
    aces_as_11 = ace_count
    while total > 21 and aces_as_11 > 0:
        total -= 10
        aces_as_11 -= 1

    # Soft if at least one Ace still counted as 11 and not bust
    player_soft = aces_as_11 > 0 and total <= 21

    # Pair only if 2 cards & same rank
    is_pair = len(values) == 2 and values[0] == values[1]
    pair_rank = values[0] if is_pair else None

    return total, player_soft, is_pair, pair_rank

# Apply features
(
    sample["player_value"],
    sample["player_soft"],
    sample["is_pair"],
    sample["pair_rank"],
) = zip(*sample["initial_hand"].apply(hand_features))

# ===============================
# Action extraction (first decision)
# ===============================

VALID_ACTIONS = {"H", "S", "D", "P", "R"}  # Hit, Stand, Double, Split, Surrender

def extract_first_action(actions):
    """
    Extract FIRST action from FIRST hand.
    actions_taken examples:
      [['S']]
      [['H', 'S']]
      [['P', 'H', 'S'], ['H', 'S']]
    Returns: one of VALID_ACTIONS or None.
    """
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

sample["final_action"] = sample["actions_taken"].apply(extract_first_action)

# ===============================
# Reward
# ===============================

sample["reward"] = sample["win"]

# ===============================
# Diagnostics
# ===============================

print("\nSample preview:")
print(
    sample[
        [
            "initial_hand",
            "player_value",
            "player_soft",
            "is_pair",
            "pair_rank",
            "actions_taken",
            "final_action",
            "reward",
        ]
    ].head(20)
)

print("\nUnique final_action values:")
print(sample["final_action"].value_counts(dropna=False))

print("\nAverage reward per final_action:")
print(sample.groupby("final_action")["reward"].mean())

# ===============================
# Build normal + split policies
# ===============================

valid_sample = sample[sample["final_action"].isin(VALID_ACTIONS)].copy()

# Normal policy: all non-pairs, plus pairs where action != P
normal_mask = (~valid_sample["is_pair"]) | (
    (valid_sample["is_pair"]) & (valid_sample["final_action"] != "P")
)

# Split policy: only true pairs with split as first action
pair_mask = (valid_sample["is_pair"]) & (valid_sample["final_action"] == "P")

# --- Normal policy ---
normal_grouped = (
    valid_sample[normal_mask]
    .groupby(["player_value", "player_soft", "dealer_up", "final_action"])["reward"]
    .mean()
    .reset_index()
)

normal_policy = (
    normal_grouped.sort_values("reward", ascending=False)
    .groupby(["player_value", "player_soft", "dealer_up"])
    .first()
    .reset_index()
)

print("\nNormal policy table sample (top 20 rows):")
print(normal_policy.head(20))

# --- Split policy ---
pair_grouped = (
    valid_sample[pair_mask]
    .groupby(["pair_rank", "dealer_up", "final_action"])["reward"]
    .mean()
    .reset_index()
)

split_policy = (
    pair_grouped.sort_values("reward", ascending=False)
    .groupby(["pair_rank", "dealer_up"])
    .first()
    .reset_index()
)

print("\nSplit policy table sample (top 20 rows):")
print(split_policy.head(20))

# ===============================
# Build lookup dicts
# ===============================

normal_policy_dict = {}
for _, row in normal_policy.iterrows():
    key = (int(row["player_value"]), bool(row["player_soft"]), int(row["dealer_up"]))
    normal_policy_dict[key] = (row["final_action"], float(row["reward"]))

split_policy_dict = {}
for _, row in split_policy.iterrows():
    key = (int(row["pair_rank"]), int(row["dealer_up"]))
    split_policy_dict[key] = (row["final_action"], float(row["reward"]))

# ===============================
# Recommender
# ===============================

def recommend_action(initial_hand, dealer_up):
    """
    Recommend an action using:
      - split_policy_dict for true pairs (if available)
      - normal_policy_dict for totals
      - simple fallback otherwise
    """
    if isinstance(initial_hand, str):
        try:
            initial_hand = ast.literal_eval(initial_hand)
        except (SyntaxError, ValueError):
            raise ValueError(f"Invalid hand format: {initial_hand}")

    player_value, player_soft, is_pair, pair_rank = hand_features(initial_hand)

    if player_value is None:
        raise ValueError(f"Invalid initial_hand: {initial_hand}")

    du = int(dealer_up)

    # 1) Use split policy if this is a pair and we have data
    if is_pair and pair_rank is not None:
        sp_key = (int(pair_rank), du)
        if sp_key in split_policy_dict:
            action, er = split_policy_dict[sp_key]
            return {
                "player_value": int(player_value),
                "player_soft": bool(player_soft),
                "is_pair": True,
                "pair_rank": int(pair_rank),
                "dealer_up": du,
                "action": action,
                "expected_reward": float(er),
                "source": "split_policy",
            }

    # 2) Use normal policy for this total/softness/dealer
    n_key = (int(player_value), bool(player_soft), du)
    if n_key in normal_policy_dict:
        action, er = normal_policy_dict[n_key]
        return {
            "player_value": int(player_value),
            "player_soft": bool(player_soft),
            "is_pair": bool(is_pair),
            "pair_rank": int(pair_rank) if is_pair else None,
            "dealer_up": du,
            "action": action,
            "expected_reward": float(er),
            "source": "normal_policy",
        }

    # 3) Fallback: naive rule if state not covered in sample
    fallback_action = "H" if player_value < 12 else "S"
    return {
        "player_value": int(player_value),
        "player_soft": bool(player_soft),
        "is_pair": bool(is_pair),
        "pair_rank": int(pair_rank) if is_pair else None,
        "dealer_up": du,
        "action": fallback_action,
        "expected_reward": 0.0,
        "source": "fallback",
        "note": "No exact match in learned policies (sample-based).",
    }

# ===============================
# Quick tests
# ===============================

print("\nRecommendation tests:")
test_cases = [
    ([10, 11], 10),  # blackjack vs 10
    ([10, 6], 10),   # 16 vs 10
    ([8, 8], 6),     # 8,8 vs 6 (split candidate)
]

for hand, up in test_cases:
    print(f"Hand={hand}, dealer_up={up} -> {recommend_action(hand, up)}")
