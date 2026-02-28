#!/usr/bin/env python3
import sys
import socket
import random
from collections import defaultdict
from dataclasses import dataclass

# ─── Tuning ───────────────────────────────────────────────────────────────────
SIMULATIONS    = 1000   # ↑ upgrade 1: more rollouts = better accuracy
DENIAL_WEIGHT  = 0.6    # how much to penalise best opponent's score (0=ignore, 1=full)
RANDOM_SEED    = None   # set an int for reproducible games

# ─── Card catalogue ───────────────────────────────────────────────────────────
MAKI_CARDS   = {"Maki Roll 1", "Maki Roll 2", "Maki Roll 3"}
MAKI_ICONS   = {"Maki Roll 1": 1, "Maki Roll 2": 2, "Maki Roll 3": 3}
NIGIRI_CARDS = {"Squid Nigiri", "Salmon Nigiri", "Egg Nigiri"}
NIGIRI_PTS   = {"Squid Nigiri": 3, "Salmon Nigiri": 2, "Egg Nigiri": 1}
SET_CARDS    = {"Sashimi", "Tempura", "Dumpling"}

DECK: dict[str, int] = {
    "Tempura":       14,
    "Sashimi":       14,
    "Dumpling":      14,
    "Maki Roll 1":    6,
    "Maki Roll 2":   12,
    "Maki Roll 3":    8,
    "Salmon Nigiri": 10,
    "Squid Nigiri":   5,
    "Egg Nigiri":     5,
    "Pudding":       10,
    "Wasabi":         6,
    "Chopsticks":     4,
}


# ─── Strategy profiles ────────────────────────────────────────────────────────

@dataclass
class StrategyProfile:
    name:              str
    maki_weight:       float
    set_weight:        float
    dumpling_weight:   float
    nigiri_weight:     float
    wasabi_weight:     float
    pudding_weight:    float
    chopsticks_weight: float
    denial_mode:       bool   # apply relative scoring vs best opponent
    maki_commit:       bool   # skip Maki Roll 1 entirely
    avoid_slow_sets:   bool   # skip sets when hand is too small to complete


STRATEGIES: dict[int, StrategyProfile] = {
    2: StrategyProfile(
        name="2p: Denial + Pudding Control",
        maki_weight=1.4, set_weight=0.8, dumpling_weight=1.1,
        nigiri_weight=1.1, wasabi_weight=1.3, pudding_weight=2.0,
        chopsticks_weight=0.6,
        denial_mode=True, maki_commit=True, avoid_slow_sets=True,
    ),
    3: StrategyProfile(
        name="3p: Balanced Opportunist",
        maki_weight=1.1, set_weight=1.1, dumpling_weight=1.1,
        nigiri_weight=1.0, wasabi_weight=1.2, pudding_weight=1.3,
        chopsticks_weight=0.9,
        denial_mode=False, maki_commit=False, avoid_slow_sets=False,
    ),
    4: StrategyProfile(
        name="4p: Dumplings + Guaranteed Points",
        maki_weight=0.7, set_weight=0.9, dumpling_weight=1.5,
        nigiri_weight=1.1, wasabi_weight=1.2, pudding_weight=1.4,
        chopsticks_weight=1.0,
        denial_mode=False, maki_commit=False, avoid_slow_sets=False,
    ),
    5: StrategyProfile(
        name="5p: Nigiri Sprint",
        maki_weight=0.5, set_weight=0.4, dumpling_weight=1.3,
        nigiri_weight=1.4, wasabi_weight=1.6, pudding_weight=1.5,
        chopsticks_weight=0.3,
        denial_mode=False, maki_commit=True, avoid_slow_sets=True,
    ),
}
STRATEGIES[1] = STRATEGIES[2]
STRATEGIES[6] = STRATEGIES[5]


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_table(played: list[str]) -> int:
    """Score one player's table for the round (no maki, no pudding end-game)."""
    counts = defaultdict(int)
    wasabi = 0
    score  = 0
    for card in played:
        if card == "Wasabi":
            wasabi += 1
        elif card in NIGIRI_CARDS:
            pts = NIGIRI_PTS[card]
            if wasabi > 0:
                pts   *= 3
                wasabi -= 1
            score += pts
        else:
            counts[card] += 1
    score += (counts["Tempura"]  // 2) * 5
    score += (counts["Sashimi"]  // 3) * 10
    score += [0, 1, 3, 6, 10, 15][min(counts["Dumpling"], 5)]
    return score


def score_maki(all_played: list[list[str]]) -> list[int]:
    """Maki bonus: 6 pts to most, 3 pts to second most (if >1 player)."""
    totals  = [sum(MAKI_ICONS.get(c, 0) for c in p) for p in all_played]
    bonuses = [0] * len(all_played)
    if not any(totals):
        return bonuses
    ranked  = sorted(set(totals), reverse=True)
    first_w = [i for i, t in enumerate(totals) if t == ranked[0]]
    for i in first_w:
        bonuses[i] += 6 // len(first_w)
    if len(first_w) == 1 and len(ranked) > 1 and ranked[1] > 0:
        second_w = [i for i, t in enumerate(totals) if t == ranked[1]]
        for i in second_w:
            bonuses[i] += 3 // len(second_w)
    return bonuses


def score_pudding(counts: list[int]) -> list[int]:
    """End-game pudding: +6 most, −6 fewest (2+ players)."""
    bonuses = [0] * len(counts)
    if len(counts) < 2:
        return bonuses
    hi = max(counts); lo = min(counts)
    for i, p in enumerate(counts):
        if p == hi: bonuses[i] +=  6 // sum(1 for x in counts if x == hi)
        if p == lo: bonuses[i] += -6 // sum(1 for x in counts if x == lo)
    return bonuses


# ─── Upgrade 4: opponent needs model ─────────────────────────────────────────

def opponent_needs(table: list[str]) -> dict[str, float]:
    """
    Given what an opponent has on their table, return a weight map
    of cards they are likely to want next.  Used in heuristic rollouts
    so simulated opponents behave realistically rather than randomly.
    """
    needs: dict[str, float] = {}
    n = table.count

    # Sashimi: desperately needs the last piece(s)
    sash_mod = n("Sashimi") % 3
    if sash_mod == 2:   needs["Sashimi"] = 10.0   # one away from 10pts
    elif sash_mod == 1: needs["Sashimi"] = 4.0

    # Tempura: needs second of pair
    if n("Tempura") % 2 == 1: needs["Tempura"] = 5.0

    # Dumplings: always want more (accelerating returns)
    d = min(n("Dumpling"), 4)
    needs["Dumpling"] = [0, 1, 3, 6, 10, 15][d + 1] - [0, 1, 3, 6, 10, 15][d] + 0.5

    # Wasabi: want to cash it in
    unmatched_wasabi = n("Wasabi") - n("Squid Nigiri") - n("Salmon Nigiri") - n("Egg Nigiri")
    if unmatched_wasabi > 0:
        needs["Squid Nigiri"]  = 9.0
        needs["Salmon Nigiri"] = 6.0
        needs["Egg Nigiri"]    = 3.0
    else:
        needs["Wasabi"]        = 4.0
        needs["Squid Nigiri"]  = 3.0
        needs["Salmon Nigiri"] = 2.0
        needs["Egg Nigiri"]    = 1.0

    # Maki: always mildly interesting
    needs["Maki Roll 3"] = 3.0
    needs["Maki Roll 2"] = 2.0
    needs["Maki Roll 1"] = 0.5

    # Pudding: mild value
    needs["Pudding"] = 1.5

    return needs


def _heuristic_choice(
    hand:    list[str],
    table:   list[str],
    profile: StrategyProfile,
    rng:     random.Random,
) -> str:
    """
    Upgrade 3: pick a card using heuristic weights rather than pure random.
    Used for BOTH our simulated future turns AND opponent simulated turns.
    """
    needs   = opponent_needs(table)
    weights = []

    for card in hand:
        base = needs.get(card, 1.0)

        # Apply profile multipliers
        if card in MAKI_CARDS:
            w = base * profile.maki_weight
            if profile.maki_commit and card == "Maki Roll 1":
                w *= 0.1
        elif card in ("Sashimi", "Tempura"):
            w = base * profile.set_weight
            if profile.avoid_slow_sets and table.count(card) == 0:
                w *= 0.2
        elif card == "Dumpling":
            w = base * profile.dumpling_weight
        elif card in NIGIRI_CARDS:
            w = base * profile.nigiri_weight
        elif card == "Wasabi":
            w = base * profile.wasabi_weight
        elif card == "Pudding":
            w = base * profile.pudding_weight
        elif card == "Chopsticks":
            w = base * profile.chopsticks_weight
        else:
            w = base

        weights.append(max(w, 0.01))

    total = sum(weights)
    r     = rng.random() * total
    for card, w in zip(hand, weights):
        r -= w
        if r <= 0:
            return card
    return hand[-1]


# ─── Simulation engine ────────────────────────────────────────────────────────

def simulate_round(
    hands:       list[list[str]],   # index 0 = us (already has candidate played)
    tables:      list[list[str]],   # index 0 = our table after candidate
    puddings:    list[int],
    num_players: int,
    profile:     StrategyProfile,
    rng:         random.Random,
    opp_profiles: list[StrategyProfile],  # upgrade 4: per-opponent profiles
) -> float:
    """
    Simulate the rest of the round.
    - Our future picks: heuristic-guided (upgrade 3)
    - Opponent picks:   heuristic-guided with their inferred needs (upgrades 3+4)
    - Scoring:          relative (our score − best_opponent × DENIAL_WEIGHT) (upgrade 2)
    """
    hands  = [h[:] for h in hands]
    tables = [t[:] for t in tables]

    while any(h for h in hands):
        for p in range(num_players):
            if not hands[p]:
                continue
            prof = profile if p == 0 else opp_profiles[p - 1]
            card = _heuristic_choice(hands[p], tables[p], prof, rng)
            hands[p].remove(card)
            tables[p].append(card)

        # Pass hands left
        hands = [hands[(p + 1) % num_players] for p in range(num_players)]

    # Score everyone
    maki_bonus = score_maki(tables)
    round_scores = [score_table(tables[p]) + maki_bonus[p] for p in range(num_players)]

    pud_counts = [puddings[p] + tables[p].count("Pudding") for p in range(num_players)]
    pud_bonus  = score_pudding(pud_counts)

    our_total  = round_scores[0] + pud_bonus[0] * 0.33

    # ── Upgrade 2: relative scoring ──────────────────────────────────────────
    if num_players > 1:
        best_opp = max(round_scores[p] + pud_bonus[p] * 0.33
                       for p in range(1, num_players))
        our_total -= best_opp * DENIAL_WEIGHT

    return our_total


# ─── Bot ──────────────────────────────────────────────────────────────────────

class SushiGoBot:

    def __init__(self, host: str, port: int, game_id: str, name: str) -> None:
        self.host    = host
        self.port    = port
        self.game_id = game_id
        self.name    = name
        self.rng     = random.Random(RANDOM_SEED)

        # Game state
        self.num_players: int             = 2
        self.round:       int             = 0
        self.our_table:   list[str]       = []
        self.puddings:    list[int]       = []

        # Upgrade 4+5: per-opponent state
        self.opp_tables:  list[list[str]] = []   # what each opponent has played
        self.seen_cards:  list[str]       = []   # all cards seen this round

        self._profile: StrategyProfile    = STRATEGIES[2]

    # ── Main strategy entry point ─────────────────────────────────────────────

    def choose_card(self, hand: list[str]) -> int:
        """
        Choose which card to play.

        This is where you implement your AI strategy!
        Uses all 5 competitive upgrades:
          1. 1000 simulations per candidate
          2. Relative scoring (maximise lead, not just score)
          3. Heuristic rollouts (opponents play smart)
          4. Opponent modelling (track what they need)
          5. Rotation tracking (realistic deck estimation)

        Args:
            hand: List of card names in your current hand

        Returns:
            Index of the card to play (0-based)
        """
        if len(hand) == 1:
            return 0

        # Auto-detect player count and load profile (upgrade 4)
        num_players      = max(2, len(self.opp_tables) + 1)
        self.num_players = num_players
        self._profile    = STRATEGIES.get(num_players, STRATEGIES[3])

        # Build per-opponent profiles for heuristic rollouts (upgrade 4)
        # Opponents are assumed to play the same player-count strategy as us
        opp_profiles = [self._profile] * (num_players - 1)

        print(f"\n  ── {self._profile.name} ──")
        print(f"     Players={num_players}  Round={self.round+1}"
              f"  Table={self.our_table}")
        print(f"     Hand={hand}")

        # Upgrade 5: realistic deck pool
        deck_est = self._estimate_remaining_deck(hand)

        best_idx, best_ev = 0, -999.0
        skipped_all = True   # safety: if all cards skipped, fall back

        for cand_idx, cand_card in enumerate(hand):
            if self._should_skip(cand_card, hand):
                print(f"    {cand_card:<24} SKIPPED")
                continue
            skipped_all = False

            our_table_after = self.our_table + [cand_card]
            remaining       = [c for i, c in enumerate(hand) if i != cand_idx]
            total           = 0.0

            for _ in range(SIMULATIONS):
                # Deal opponent hands from realistic remaining pool (upgrade 5)
                opp_hands  = self._deal_opponent_hands(deck_est, remaining, num_players)
                all_hands  = [remaining[:]] + opp_hands
                all_tables = [our_table_after[:]] + [t[:] for t in self.opp_tables]
                while len(all_tables) < num_players:
                    all_tables.append([])

                total += simulate_round(
                    hands        = all_hands,
                    tables       = all_tables,
                    puddings     = self._get_pudding_counts(num_players),
                    num_players  = num_players,
                    profile      = self._profile,
                    rng          = self.rng,
                    opp_profiles = opp_profiles,
                )

            ev = total / SIMULATIONS
            print(f"    {cand_card:<24} EV={ev:+.2f}")

            if ev > best_ev:
                best_ev  = ev
                best_idx = cand_idx

        # Safety fallback: if all candidates were skipped (shouldn't happen)
        if skipped_all:
            best_idx = 0

        print(f"  → PLAY '{hand[best_idx]}' (EV={best_ev:+.2f})\n")
        return best_idx

    def _should_skip(self, card: str, hand: list[str]) -> bool:
        """Hard overrides — skip before simulation when strategically obvious."""
        p = self._profile
        n = self.our_table.count

        if p.avoid_slow_sets:
            if card == "Sashimi" and n("Sashimi") == 0:
                return True
            if card == "Tempura" and n("Tempura") == 0 and len(hand) <= 4:
                return True

        if p.maki_commit and card == "Maki Roll 1":
            return True

        # Never stack unmatched wasabi
        unmatched = n("Wasabi") - n("Squid Nigiri") - n("Salmon Nigiri") - n("Egg Nigiri")
        if card == "Wasabi" and unmatched > 0:
            return True

        return False

    # ── State management ──────────────────────────────────────────────────────

    def _estimate_remaining_deck(self, our_hand: list[str]) -> list[str]:
        """
        Upgrade 5: subtract seen cards AND our hand from the full deck.
        This gives the most accurate possible pool for opponent hand sampling.
        """
        remaining = dict(DECK)
        for card in self.seen_cards:
            remaining[card] = max(0, remaining.get(card, 0) - 1)
        for card in our_hand:
            remaining[card] = max(0, remaining.get(card, 0) - 1)
        return [card for card, cnt in remaining.items() for _ in range(cnt)]

    def _deal_opponent_hands(
        self, deck_est: list[str], our_remaining: list[str], num_players: int
    ) -> list[list[str]]:
        pool = deck_est[:]
        self.rng.shuffle(pool)
        size  = len(our_remaining)
        hands = []
        for _ in range(num_players - 1):
            hands.append(pool[:size])
            pool = pool[size:]
        return hands

    def _get_pudding_counts(self, num_players: int) -> list[int]:
        counts = list(self.puddings)
        while len(counts) < num_players:
            counts.append(0)
        return counts[:num_players]

    def _record_play(self, card: str) -> None:
        self.our_table.append(card)
        self.seen_cards.append(card)

    def _record_opponent_play(self, opp_idx: int, card: str) -> None:
        """Upgrade 4: track each opponent's table for opponent modelling."""
        while len(self.opp_tables) <= opp_idx:
            self.opp_tables.append([])
        self.opp_tables[opp_idx].append(card)
        self.seen_cards.append(card)

    def _reset_round(self) -> None:
        if not self.puddings:
            self.puddings = [0] * max(2, len(self.opp_tables) + 1)
        self.puddings[0] += self.our_table.count("Pudding")
        for i, t in enumerate(self.opp_tables):
            if i + 1 < len(self.puddings):
                self.puddings[i + 1] += t.count("Pudding")
        self.round     += 1
        self.our_table  = []
        self.opp_tables = []
        self.seen_cards = []

    # ── Protocol ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        print(f"Connecting to {self.host}:{self.port} …")
        with socket.create_connection((self.host, self.port)) as sock:
            f = sock.makefile("rw", encoding="utf-8")

            def send(msg: str) -> None:
                print(f"→ {msg}")
                f.write(msg + "\n")
                f.flush()

            send(f"JOIN {self.game_id} {self.name}")

            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                print(f"← {line}")

                if line.startswith("WELCOME"):
                    send("READY")

                elif line == "OK":
                    pass

                elif line.startswith("HAND"):
                    hand = self._parse_hand(line)
                    idx  = self.choose_card(hand)
                    self._record_play(hand[idx])
                    send(f"PLAY {idx}")

                elif line.startswith("PLAYED"):
                    # PLAYED player_id:Card Name player_id:Card Name ...
                    self._handle_played(line)

                elif line.startswith("ROUND_END"):
                    print(f"── Round {self.round + 1} ended ──")
                    self._reset_round()

                elif line.startswith("GAME_END"):
                    print("🎉 Game over!")
                    break

        print("Disconnected.")

    def _handle_played(self, line: str) -> None:
        """
        Parse PLAYED message and record opponent cards.
        PLAYED format: player_id:CardName player_id:Multi Word Card ...
        We record all entries as opponent plays; our own card is already
        recorded via _record_play so we skip the last entry in our table.
        """
        tokens  = line.split()[1:]
        entries = []
        current_id   = None
        current_name: list[str] = []

        for tok in tokens:
            if ":" in tok and tok.split(":")[0].isdigit():
                if current_id is not None:
                    entries.append((current_id, " ".join(current_name)))
                current_id   = tok.split(":")[0]
                current_name = [tok.split(":", 1)[1]]
            elif current_id is not None:
                current_name.append(tok)
        if current_id is not None:
            entries.append((current_id, " ".join(current_name)))

        # Assign opponents by slot order, skipping our own last played card
        our_last = self.our_table[-1] if self.our_table else None
        opp_idx  = 0
        for pid, card in entries:
            if card == our_last:
                our_last = None   # skip our own entry once
                continue
            self._record_opponent_play(opp_idx, card)
            opp_idx += 1

    @staticmethod
    def _parse_hand(line: str) -> list[str]:
        """Parse 'HAND 0:Tempura 1:Salmon Nigiri 2:Pudding' → list of names."""
        tokens  = line.split()[1:]
        hand: list[str]       = []
        current: list[str]    = []
        for tok in tokens:
            if tok and tok[0].isdigit() and ":" in tok:
                if current:
                    hand.append(" ".join(current))
                    current = []
                current.append(tok.split(":", 1)[1])
            elif current:
                current.append(tok)
        if current:
            hand.append(" ".join(current))
        return hand


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    if len(args) >= 4 and args[1].isdigit():
        host, port_str, game_id, *name_parts = args
    elif len(args) >= 2:
        host, port_str = "localhost", "7878"
        game_id, *name_parts = args
    else:
        print("Usage: python sushi_go_client.py [host port] <game_id> <n>")
        sys.exit(1)

    SushiGoBot(
        host    = host,
        port    = int(port_str),
        game_id = game_id,
        name    = " ".join(name_parts),
    ).run()


if __name__ == "__main__":
    main()
