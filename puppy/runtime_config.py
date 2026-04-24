import os
from datetime import date
from typing import Any, Dict, Optional


DEFAULT_TRADING_SYMBOL = "TSLA"
DEFAULT_MARKET_MODE = "US"


def _clean_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def resolve_path(cli_value: Optional[str], env_key: str, default_value: str) -> str:
    env_value = _clean_optional(os.environ.get(env_key))
    cli_clean = _clean_optional(cli_value)
    return env_value or cli_clean or default_value


def resolve_trading_symbol(
    config: Dict[str, Any],
    cli_symbol: Optional[str] = None,
    default_symbol: str = DEFAULT_TRADING_SYMBOL,
) -> str:
    general = config.get("general", {}) if isinstance(config, dict) else {}
    config_symbol = _clean_optional(general.get("trading_symbol"))
    env_symbol = _clean_optional(os.environ.get("FINMEM_TRADING_SYMBOL"))
    cli_clean = _clean_optional(cli_symbol)

    resolved = config_symbol or env_symbol or cli_clean or default_symbol
    return resolved.upper()


def normalize_market_mode(value: Optional[str]) -> str:
    clean = _clean_optional(value)
    if not clean:
        return DEFAULT_MARKET_MODE

    normalized = clean.strip().upper().replace("-", "_")
    if normalized in {"US", "USA", "U.S.", "U_S"}:
        return "US"
    if normalized in {"VN", "VNSE", "VIETNAM", "VIET_NAM"}:
        return "VN"
    raise ValueError(
        f"Unsupported market mode '{value}'. Supported values: US, VN."
    )


def resolve_market_mode(
    config: Optional[Dict[str, Any]] = None,
    cli_market_mode: Optional[str] = None,
    default_market_mode: str = DEFAULT_MARKET_MODE,
) -> str:
    general = config.get("general", {}) if isinstance(config, dict) else {}
    config_market = _clean_optional(general.get("market_mode") or general.get("market"))
    env_market = _clean_optional(
        os.environ.get("FINMEM_MARKET_MODE") or os.environ.get("FINMEM_MARKET")
    )
    cli_clean = _clean_optional(cli_market_mode)

    resolved = config_market or env_market or cli_clean or default_market_mode
    return normalize_market_mode(resolved)


def expand_symbol_template(text: str, symbol: str) -> str:
    if not isinstance(text, str):
        return text
    out = text
    out = out.replace("{trading_symbol}", symbol)
    out = out.replace("{symbol}", symbol)
    out = out.replace("${TRADING_SYMBOL}", symbol)
    return out


def validate_symbol_in_market_data(
    env_data_pkl: Dict[date, Dict[str, Any]],
    symbol: str,
    start_date: date,
    end_date: date,
) -> None:
    if not env_data_pkl:
        raise ValueError("Market data is empty.")
    if start_date not in env_data_pkl or end_date not in env_data_pkl:
        raise ValueError("start_date and end_date must be present in market data.")

    selected_dates = sorted(d for d in env_data_pkl.keys() if start_date <= d <= end_date)
    if not selected_dates:
        raise ValueError("No market data found in selected date range.")

    for cur_date in selected_dates:
        day_record = env_data_pkl.get(cur_date, {})
        price_map = day_record.get("price", {}) if isinstance(day_record, dict) else {}
        if symbol not in price_map:
            available = ", ".join(sorted(price_map.keys())) if isinstance(price_map, dict) else ""
            available = available or "<none>"
            raise ValueError(
                f"Resolved trading_symbol '{symbol}' not found in market data for {cur_date}. "
                f"Available symbols: {available}"
            )
