import random
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Dict, Any, Optional


# ===============================
# Config
# ===============================

DEFAULT_NUM_DECKS = 6
BLACKJACK_PAYOUT = 1.5
DEALER_HITS_SOFT_17 = False  # S17 table: dealer stands on all 17
ALLOW_DOUBLE_AFTER_SPLIT = True
MAX_SPLITS = 3  # up to 4 hands total


# ===============================
# Card & shoe utilities
# ===============================

def create_shoe(num_decks: int = DEFAULT_NUM_DECKS) -> List[int]:
    """
    Create a shoe as a list of card values.
    Card encoding:
      2-10 = pip values
      10 repeated for J, Q, K
      11 = Ace
    """
    one_deck = (
        [2, 3, 4, 5, 6, 7, 8, 9] +
        [10, 10, 10, 10] +  # 10, J, Q, K
        [11]  # Ace
    )

    # adjust counts per real deck:
    # 4 of each pip (2-9), 16 of value 10, 4 of Ace
    deck = []
    for _ in range(4):
        deck.extend([2, 3, 4, 5, 6, 7, 8, 9])
        deck.extend([10] * 4)
        deck.append(11)

    shoe = deck * num_decks
    random.shuffle(shoe)
    return shoe


def draw_card(shoe: List[int]) -> int:
    if not shoe:
        raise RuntimeError("Shoe is empty. Recreate or reshuffle before drawing.")
    return shoe.pop()


# ===============================
# Hand evaluation
# ===============================

def hand_value(hand: List[int]) -> Tuple[int, bool]:
    """
    Returns (total, is_soft).
    Ace encoded as 11. Adjusts to avoid bust if possible.
    """
    total = sum(hand)
    aces = hand.count(11)

    # downgrade Aces from 11 to 1 while busting
    aces_as_11 = aces
    while total > 21 and aces_as_11 > 0:
        total -= 10
        aces_as_11 -= 1

    is_soft = aces_as_11 > 0 and total <= 21
    return total, is_soft


def is_blackjack(hand: List[int]) -> bool:
    """
    True if hand is a natural blackjack: exactly 2 cards: Ace + 10-value.
    """
    if len(hand) != 2:
        return False
    total, _ = hand_value(hand)
    return total == 21


def can_split(hand: List[int]) -> bool:
    """
    True if exactly 2 cards and same value (pair).
    """
    return len(hand) == 2 and hand[0] == hand[1]


# ===============================
# Data structures
# ===============================

@dataclass
class HandState:
    cards: List[int]
    bet: float = 1.0
    is_finished: bool = False
    is_busted: bool = False
    is_blackjack: bool = False
    is_split_ace: bool = False  # special rule: after A,A split
    result: Optional[float] = None  # final profit for this hand relative to bet
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundState:
    shoe: List[int]
    player_hands: List[HandState]
    dealer_cards: List[int]
    initial_blackjack: bool = False
    logs: Dict[str, Any] = field(default_factory=dict)


# ===============================
# Dealer play
# ===============================

def play_dealer(shoe: List[int], dealer_cards: List[int]) -> List[int]:
    """
    Play dealer hand according to S17/H17 rules.
    """
    while True:
        total, is_soft = hand_value(dealer_cards)

        if total > 21:
            break

        # dealer stands on all 17 if S17
        if not DEALER_HITS_SOFT_17:
            if total >= 17:
                break
        else:
            # H17: hit soft 17, stand on hard 17+
            if total > 17:
                break
            if total == 17 and not is_soft:
                break

        # otherwise hit
        dealer_cards.append(draw_card(shoe))

    return dealer_cards


# ===============================
# Round logic
# ===============================

def apply_action_hit(shoe: List[int], hand: HandState) -> None:
    hand.cards.append(draw_card(shoe))
    total, _ = hand_value(hand.cards)
    if total > 21:
        hand.is_busted = True
        hand.is_finished = True


def apply_action_stand(hand: HandState) -> None:
    hand.is_finished = True


def apply_action_double(shoe: List[int], hand: HandState) -> None:
    """
    Double: allowed only on first two cards (standard).
    One more card, then stand.
    """
    if len(hand.cards) != 2:
        # treat as hit if invalid double
        apply_action_hit(shoe, hand)
        return

    hand.bet *= 2
    hand.cards.append(draw_card(shoe))
    total, _ = hand_value(hand.cards)
    if total > 21:
        hand.is_busted = True
    hand.is_finished = True


def apply_action_split(shoe: List[int], round_state: RoundState, hand_index: int) -> None:
    """
    Split one hand into two.
    Only if:
      - pair
      - max splits not exceeded
    """
    hand = round_state.player_hands[hand_index]
    if not can_split(hand.cards):
        return

    # count existing split groups by meta flag
    existing_splits = sum(1 for h in round_state.player_hands if h.meta.get("from_split", False))
    if existing_splits >= MAX_SPLITS:
        return

    card = hand.cards[0]

    # new hand 1
    hand.cards = [card, draw_card(shoe)]
    hand.is_blackjack = is_blackjack(hand.cards)
    hand.meta["from_split"] = True

    # new hand 2
    new_hand_cards = [card, draw_card(shoe)]
    new_hand = HandState(
        cards=new_hand_cards,
        bet=hand.bet,
        is_finished=False,
        is_busted=False,
        is_blackjack=is_blackjack(new_hand_cards),
        meta={"from_split": True},
    )

    # A,A split rule: often only 1 card per hand, no further hit
    if card == 11:
        hand.is_split_ace = True
        new_hand.is_split_ace = True
        hand.is_finished = True
        new_hand.is_finished = True

    round_state.player_hands.insert(hand_index + 1, new_hand)


def settle_hand_vs_dealer(hand: HandState, dealer_cards: List[int]) -> float:
    """
    Compute final profit for one hand:
      +1  win
      -1  lose
      0   push
      +1.5 blackjack (if natural)
    """
    if hand.result is not None:
        return hand.result  # already settled

    if hand.is_busted:
        hand.result = -hand.bet
        return hand.result

    player_total, _ = hand_value(hand.cards)
    dealer_total, _ = hand_value(dealer_cards)

    # dealer bust
    if dealer_total > 21:
        hand.result = hand.bet
        return hand.result

    # player blackjack
    if hand.is_blackjack and not (len(dealer_cards) == 2 and dealer_total == 21):
        hand.result = hand.bet * BLACKJACK_PAYOUT
        return hand.result

    # dealer blackjack vs non-blackjack already handled outside usually

    if dealer_total > player_total:
        hand.result = -hand.bet
    elif dealer_total < player_total:
        hand.result = hand.bet
    else:
        hand.result = 0.0

    return hand.result


# ===============================
# Main round simulation API
# ===============================

PolicyFunc = Callable[[List[int], int, Dict[str, Any]], str]


def play_round(
    shoe: List[int],
    policy: PolicyFunc,
    base_bet: float = 1.0,
    log: bool = False,
) -> RoundState:
    """
    Play one full round of blackjack for a single player (with possible splits),
    using the given policy function.

    policy(player_cards, dealer_upcard, context) -> action in {"H","S","D","P","R"}

    Returns:
      RoundState with full details (hands, dealer cards, results).
    """

    # initial deal
    player_cards = [draw_card(shoe), draw_card(shoe)]
    dealer_cards = [draw_card(shoe), draw_card(shoe)]  # dealer_cards[0] = upcard

    player_hand = HandState(
        cards=player_cards[:],
        bet=base_bet,
        is_blackjack=is_blackjack(player_cards),
    )

    state = RoundState(
        shoe=shoe,
        player_hands=[player_hand],
        dealer_cards=dealer_cards[:],
        initial_blackjack=player_hand.is_blackjack,
        logs={"steps": []} if log else {},
    )

    dealer_up = dealer_cards[0]

    # Check for immediate blackjack outcomes
    dealer_has_bj = is_blackjack(dealer_cards)
    if player_hand.is_blackjack or dealer_has_bj:
        # no further actions, settle directly
        if player_hand.is_blackjack and not dealer_has_bj:
            player_hand.result = base_bet * BLACKJACK_PAYOUT
        elif dealer_has_bj and not player_hand.is_blackjack:
            player_hand.result = -base_bet
        else:
            player_hand.result = 0.0  # both BJ -> push

        return state

    # Player actions (possibly multiple hands due to splits)
    hand_index = 0
    while hand_index < len(state.player_hands):
        hand = state.player_hands[hand_index]

        # if already finished (e.g. split Aces)
        if hand.is_finished:
            hand_index += 1
            continue

        # Loop for this hand until stand/bust/double/etc.
        while not hand.is_finished:
            total, is_soft = hand_value(hand.cards)

            # if busted already
            if total > 21:
                hand.is_busted = True
                hand.is_finished = True
                break

            # build context for policy
            ctx = {
                "player_total": total,
                "player_soft": is_soft,
                "can_split": can_split(hand.cards),
                "can_double": len(hand.cards) == 2,
                "hand_index": hand_index,
                "num_hands": len(state.player_hands),
            }

            action = policy(hand.cards[:], dealer_up, ctx).upper()

            if log:
                state.logs["steps"].append({
                    "hand_index": hand_index,
                    "cards": hand.cards[:],
                    "dealer_up": dealer_up,
                    "action": action,
                    "ctx": ctx,
                })

            # Apply action
            if action == "H":
                # cannot hit split aces in standard rules
                if hand.is_split_ace:
                    hand.is_finished = True
                else:
                    apply_action_hit(shoe, hand)

            elif action == "S":
                apply_action_stand(hand)

            elif action == "D":
                if len(hand.cards) == 2:
                    apply_action_double(shoe, hand)
                else:
                    # treat invalid double as hit
                    apply_action_hit(shoe, hand)

            elif action == "P":
                if can_split(hand.cards):
                    apply_action_split(shoe, state, hand_index)
                    # after split, do not auto-advance hand_index yet
                    break
                else:
                    # invalid split -> treat as hit
                    apply_action_hit(shoe, hand)

            elif action == "R":
                # surrender not enabled in this engine by default
                # treat as stand or ignore; here: treat as stand
                apply_action_stand(hand)

            else:
                # unknown action -> stand as safe fallback
                apply_action_stand(hand)

        hand_index += 1

    # Dealer plays if at least one hand not busted
    if any(not h.is_busted for h in state.player_hands):
        play_dealer(shoe, state.dealer_cards)

    # Settle all hands
    for hand in state.player_hands:
        settle_hand_vs_dealer(hand, state.dealer_cards)

    return state
