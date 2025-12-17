import pandas as pd
import os
import ast
import matplotlib.pyplot as plt



# 1️⃣ ataset path
DATASET_PATH = os.path.expanduser("~/blackjack_bot/blackjack_simulator.csv")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"File not found at: {DATASET_PATH}")

print(f"Loaded file: {DATASET_PATH}")

# 2️⃣ oad dataset
df = pd.read_csv(DATASET_PATH)

# 3️⃣ isplay column names
print("\nColumn names:")
print(df.columns.tolist())

# 4️⃣ isplay first 5 rows
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
print("\nFirst 5 rows:")
print(df.head(5))

# 5️⃣ alculate player_value and player_soft from initial_hand

def hand_value_and_soft(hand):
    """
    Input: [10, 11] or string "[10, 11]"
    Output: (hand_value, soft_hand_boolean)
    """
    if isinstance(hand, str):
        hand = ast.literal_eval(hand)

    total = 0
    aces = 0

    for card in hand:
        if card == 11:  # 11 = Ace
            aces += 1
            total += 11
        else:
            total += int(card)

    soft = False

    # If total > 21 and there are aces, convert 11 -> 1
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
        soft = True

    # If there's still an Ace counted as 11 and total <= 21, it's a soft hand
    if aces > 0 and total <= 21:
        soft = True

    return total, soft


# 6️⃣reate new columns
# Work only on a sample to test the logic
sample = df.head(1000).copy()
sample["player_value"], sample["player_soft"] = zip(*sample["initial_hand"].apply(hand_value_and_soft))

print("\nCalculated values on sample (initial_hand → player_value, player_soft):")
print(sample[["initial_hand", "player_value", "player_soft"]].head(20))

def extract_last_action(actions):
    """
    Extract the final action from actions_taken.
    actions example formats:
    - [['S']]
    - [['H', 'S']]
    - [['H', 'H', 'S']]
    - [['P', 'H', 'S'], ['H', 'S']]  (multiple hands after split)
    For now, we take the last action of the FIRST hand.
    """
    if isinstance(actions, str):
        actions = ast.literal_eval(actions)

    if not actions:
        return None

    # actions is a list of action-lists per hand
    first_hand_actions = actions[0]
    if not first_hand_actions:
        return None

    return first_hand_actions[-1]

sample["final_action"] = sample["actions_taken"].apply(extract_last_action)

print("\nSample with final_action:")
print(sample[["initial_hand", "actions_taken", "final_action"]].head(20))

print("\nUnique final_action values in sample:")
print(sample["final_action"].value_counts())
# Map win as reward for the sample
sample["reward"] = sample["win"]

print("\nAverage reward per final_action (on sample):")
print(sample.groupby("final_action")["reward"].mean())


# Group average rewards by action
avg_reward = sample.groupby("final_action")["reward"].mean().sort_values(ascending=False)
# Plot the results
plt.figure(figsize=(7, 4))
avg_reward.plot(kind="bar", color="steelblue")
plt.title("Average Reward per Action")
plt.xlabel("Final Action")
plt.ylabel("Average Reward")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# Save the plot to a file 
output_path = os.path.join(os.path.expanduser("~/blackjack_bot"), "reward_per_action.png")
plt.savefig(output_path)
print(f"\nChart saved to: {output_path}")
