"""Synthetic transaction generator for fraud-monitoring demonstrations."""
from __future__ import annotations

import random
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HISTORICAL_DATA_PATH = DATA_DIR / "historical_transactions.csv"

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "electronics", "clothing", "travel", "hotel",
    "fuel", "online_services", "jewelry", "gaming", "pharmacy", "other",
]

LOCATIONS = [
    "Addis_Ababa", "Bahir_Dar", "Hawassa", "Dire_Dawa", "Mekelle", "Gondar",
    "Jimma", "Adama", "Nairobi", "Dubai", "London", "New_York",
]

HOME_LOCATIONS = LOCATIONS[:8]

MERCHANT_AMOUNT_MULTIPLIERS = {
    "grocery": 0.7, "restaurant": 0.5, "electronics": 2.0,
    "clothing": 0.9, "travel": 2.5, "hotel": 2.0, "fuel": 0.7,
    "online_services": 0.8, "jewelry": 4.0, "gaming": 0.8,
    "pharmacy": 0.6, "other": 1.0,
}

SCENARIOS = {
    "normal",
    "high_value",
    "location_mismatch",
    "high_velocity",
    "combined",
}


class TransactionGenerator:
    """Generate synthetic transactions with reproducible risk patterns."""

    def __init__(self, seed: int | None = 42, n_accounts: int = 1000) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.account_ids = [f"ACC_{i:05d}" for i in range(1, n_accounts + 1)]
        self.account_locations = {
            account_id: self.rng.choice(HOME_LOCATIONS)
            for account_id in self.account_ids
        }
        self.transaction_history: dict[str, deque[datetime]] = defaultdict(
            lambda: deque(maxlen=200)
        )
        self.last_live_account: str | None = None
        self.live_burst_remaining = 0

    @staticmethod
    def _transaction_id() -> str:
        return f"TXN_{uuid.uuid4().hex[:16].upper()}"

    def _choose_live_account(self, scenario: str | None = None) -> str:
        """Reuse accounts for burst scenarios so velocity becomes meaningful."""
        burst_scenarios = {"high_velocity", "combined"}
        if scenario in burst_scenarios and self.last_live_account:
            return self.last_live_account

        if self.last_live_account and self.live_burst_remaining > 0:
            self.live_burst_remaining -= 1
            return self.last_live_account

        account = self.rng.choice(self.account_ids)
        self.last_live_account = account
        if self.rng.random() < 0.12:
            self.live_burst_remaining = self.rng.randint(3, 7)
        return account

    def _amount(self, merchant_category: str, suspicious: bool = False) -> float:
        amount = self.np_rng.lognormal(mean=np.log(55), sigma=0.95)
        amount *= MERCHANT_AMOUNT_MULTIPLIERS[merchant_category]
        if suspicious and self.rng.random() < 0.65:
            amount *= self.rng.uniform(20, 70)
        elif self.rng.random() < 0.008:
            amount *= self.rng.uniform(12, 35)
        return round(float(np.clip(amount, 2.0, 50000.0)), 2)

    def _velocity(self, account_id: str, timestamp: datetime) -> int:
        history = self.transaction_history[account_id]
        cutoff = timestamp - timedelta(hours=1)
        while history and history[0] <= cutoff:
            history.popleft()
        velocity = len(history)
        history.append(timestamp)
        return velocity

    def _location_mismatch(self, account_id: str, suspicious: bool = False) -> int:
        if suspicious and self.rng.random() < 0.9:
            return 1
        if self.rng.random() < 0.035:
            return int(self.rng.choice(LOCATIONS) != self.account_locations[account_id])
        return 0

    @staticmethod
    def _fraud_probability(
        amount: float,
        location_mismatch: int,
        device_velocity: int,
        merchant_category: str,
    ) -> float:
        """Business-inspired synthetic fraud propensity used only for labels."""
        probability = 0.0025
        if amount > 5000:
            probability += 0.48
        elif amount > 2000:
            probability += 0.035
        if device_velocity > 5:
            probability += 0.16
        elif device_velocity > 3:
            probability += 0.055
        if location_mismatch and device_velocity > 3:
            probability += 0.42
        elif location_mismatch:
            probability += 0.025
        if merchant_category in {"jewelry", "gaming"}:
            probability += 0.008
        return float(np.clip(probability, 0.0001, 0.97))

    def generate_transaction(
        self,
        timestamp: datetime | None = None,
        force_fraud: bool | None = None,
        suspicious_scenario: bool | None = None,
        scenario: str | None = None,
    ) -> dict[str, Any]:
        """Generate one JSON-compatible transaction payload.

        Parameters
        ----------
        scenario:
            Optional deterministic demo scenario: ``normal``, ``high_value``,
            ``location_mismatch``, ``high_velocity``, or ``combined``.
            Leaving it as ``None`` preserves the realistic random stream.
        """
        timestamp = timestamp or datetime.now(timezone.utc)

        if scenario == "random":
            scenario = None
        if scenario is not None and scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {sorted(SCENARIOS)}")

        if scenario is not None:
            suspicious_scenario = scenario != "normal"
        elif suspicious_scenario is None:
            suspicious_scenario = self.rng.random() < 0.012

        account_id = self._choose_live_account(scenario)
        merchant_category = self.rng.choice(MERCHANT_CATEGORIES)

        if scenario is None and suspicious_scenario:
            scenario = self.rng.choice(["high_value", "combined"])

        amount = self._amount(merchant_category, bool(suspicious_scenario))
        location_mismatch = self._location_mismatch(account_id, bool(suspicious_scenario))
        device_velocity = self._velocity(account_id, timestamp)

        if scenario == "normal":
            # Keep the controlled normal scenario genuinely low-risk.
            amount = round(self.rng.uniform(10, 300), 2)
            location_mismatch = 0
            device_velocity = min(device_velocity, 2)
        elif scenario == "high_value":
            # Keep the controlled scenario focused on the amount signal.
            merchant_category = "travel"
            amount = round(self.rng.uniform(12000, 15000), 2)
            location_mismatch = 0
            device_velocity = min(device_velocity, 2)
        elif scenario == "location_mismatch":
            amount = round(self.rng.uniform(20, 800), 2)
            location_mismatch = 1
            device_velocity = min(device_velocity, 2)
        elif scenario == "high_velocity":
            amount = round(self.rng.uniform(20, 800), 2)
            device_velocity = max(device_velocity, self.rng.randint(5, 8))
        elif scenario == "combined":
            # Use deliberately strong values so the portfolio demo reliably
            # crosses the same model decision threshold as a real stream.
            merchant_category = "travel"
            amount = 15000.00
            location_mismatch = 1
            device_velocity = max(device_velocity, self.rng.randint(5, 8))

        probability = self._fraud_probability(
            amount, location_mismatch, device_velocity, merchant_category
        )
        is_fraud = int(
            force_fraud
            if force_fraud is not None
            else self.np_rng.random() < probability
        )

        return {
            "transaction_id": self._transaction_id(),
            "timestamp": timestamp.isoformat(),
            "account_id": account_id,
            "amount": amount,
            "merchant_category": merchant_category,
            "location_mismatch": int(location_mismatch),
            "device_velocity": int(device_velocity),
            "is_fraud": is_fraud,
        }

    def generate_historical_data(self, n_rows: int = 10_000) -> pd.DataFrame:
        """Generate a highly imbalanced training sample with ~0.7% fraud labels."""
        if n_rows < 100:
            raise ValueError("n_rows must be at least 100")

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=180)
        timestamps = sorted(
            start + timedelta(seconds=self.rng.randint(0, 180 * 24 * 3600))
            for _ in range(n_rows)
        )

        rows: list[dict[str, Any]] = []
        for ts in timestamps:
            suspicious = self.rng.random() < 0.012
            rows.append(
                self.generate_transaction(
                    timestamp=ts,
                    suspicious_scenario=suspicious,
                )
            )

        df = pd.DataFrame(rows)

        # Construct a small, controlled positive class so the model sees each
        # intended fraud pattern. This keeps the synthetic prevalence near 0.7%
        # without letting one correlated pattern dominate every positive label.
        target_fraud = max(50, min(100, round(n_rows * 0.007)))
        df["is_fraud"] = 0
        selected: set[int] = set()

        pattern_specs = [
            (
                (df["amount"] > 5000)
                & (df["location_mismatch"] == 1)
                & (df["device_velocity"] > 3),
                30,
            ),
            (
                (df["amount"] > 5000)
                & (df["location_mismatch"] == 0)
                & (df["device_velocity"] <= 3),
                25,
            ),
            (
                (df["amount"] <= 5000)
                & (df["location_mismatch"] == 0)
                & (df["device_velocity"] > 3),
                10,
            ),
            (
                (df["amount"] <= 5000)
                & (df["location_mismatch"] == 1)
                & (df["device_velocity"] <= 3),
                5,
            ),
        ]

        for mask, quota in pattern_specs:
            candidates = df.index[mask].tolist()
            self.rng.shuffle(candidates)
            chosen = 0
            for index in candidates:
                if len(selected) >= target_fraud or chosen >= quota:
                    break
                selected.add(index)
                chosen += 1

        # Fill any unmet quota from the strongest remaining risk candidates.
        if len(selected) < target_fraud:
            candidates = [i for i in df.index if i not in selected]
            candidates.sort(
                key=lambda i: (
                    df.loc[i, "amount"] > 5000,
                    df.loc[i, "location_mismatch"] == 1
                    and df.loc[i, "device_velocity"] > 3,
                    df.loc[i, "device_velocity"],
                    df.loc[i, "amount"],
                ),
                reverse=True,
            )
            selected.update(candidates[: target_fraud - len(selected)])

        for index in selected:
            df.loc[index, "is_fraud"] = 1

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)


def generate_historical_dataset(
    n_rows: int = 10_000,
    output_path: Path = HISTORICAL_DATA_PATH,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = TransactionGenerator(seed=42).generate_historical_data(n_rows)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Fraud: {int(df['is_fraud'].sum()):,} ({df['is_fraud'].mean() * 100:.2f}%)")
    return df


if __name__ == "__main__":
    generate_historical_dataset()
