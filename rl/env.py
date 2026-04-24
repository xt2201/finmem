from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

ACTION_TO_POSITION = np.array([-1, 0, 1], dtype=np.int32)


@dataclass
class TradingEnvConfig:
    window: int = 10
    transaction_cost: float = 0.001


class TradingEnv:
    def __init__(
        self,
        prices: np.ndarray,
        dates: List,
        config: TradingEnvConfig,
    ) -> None:
        if len(prices) <= config.window + 2:
            raise ValueError("Not enough price history for the chosen window")
        self.prices = prices.astype(np.float32)
        self.dates = dates
        self.config = config
        self._features = self._build_features()
        self.t = 0
        self.position = 0

    @property
    def obs_size(self) -> int:
        return self._features.shape[1] + 1

    @property
    def action_size(self) -> int:
        return len(ACTION_TO_POSITION)

    def reset(self) -> np.ndarray:
        self.t = 0
        self.position = 0
        return self._get_obs()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        if action_idx < 0 or action_idx >= self.action_size:
            raise ValueError("Invalid action index")
        target_position = int(ACTION_TO_POSITION[action_idx])
        price_idx = self.t + self.config.window
        price = float(self.prices[price_idx])
        next_price = float(self.prices[price_idx + 1])

        log_return = float(np.log(next_price / price))
        turnover_cost = self.config.transaction_cost * abs(target_position - self.position)
        reward = target_position * log_return - turnover_cost

        self.position = target_position
        self.t += 1
        done = self.t >= len(self._features) - 1
        obs = self._get_obs()
        info = {
            "date": self.dates[price_idx],
            "position": self.position,
            "price": price,
            "next_price": next_price,
        }
        return obs, float(reward), done, info

    def _get_obs(self) -> np.ndarray:
        base = self._features[self.t]
        obs = np.concatenate([base, np.array([self.position], dtype=np.float32)])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_features(self) -> np.ndarray:
        window = self.config.window
        prices = self.prices
        features = []
        for idx in range(window, len(prices) - 1):
            p = prices[idx]
            p1 = prices[idx - 1]
            p3 = prices[idx - 3] if idx - 3 >= 0 else p1
            p5 = prices[idx - 5] if idx - 5 >= 0 else p1

            start_5 = max(0, idx - 5)
            start_10 = max(0, idx - 10)

            log_ret_1 = float(np.log(p / p1))
            log_ret_5 = float(np.log(p / p5))
            ma_5 = float(np.mean(prices[start_5 : idx + 1]))
            ma_10 = float(np.mean(prices[start_10 : idx + 1]))
            ma_5_ratio = float(p / ma_5 - 1.0)
            ma_10_ratio = float(p / ma_10 - 1.0)
            momentum_3 = float(p / p3 - 1.0)
            log_window_5 = np.log(prices[start_5 : idx + 1])
            vol_5 = float(np.std(np.diff(log_window_5))) if len(log_window_5) >= 2 else 0.0

            feature = np.asarray(
                [
                    log_ret_1,
                    log_ret_5,
                    ma_5_ratio,
                    ma_10_ratio,
                    momentum_3,
                    vol_5,
                ],
                dtype=np.float32,
            )
            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(feature.tolist())

        feature_array = np.asarray(features, dtype=np.float32)
        return np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
