import os
import re
import json
import time
import pickle
import argparse
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Tuple, Optional

import httpx
import yfinance as yf
import pytz
from dateutil import parser
from dotenv import dotenv_values, load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


ALPACA_DEFAULT_NEWS_ENDPOINT = "https://data.alpaca.markets/v1beta1/news"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_NEWS_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_OPENROUTER_NEWS_FALLBACK_MODELS = (
    "qwen/qwen3.6-35b-a3b",
    "qwen/qwen3-32b",
)
SEC_QUERY_ENDPOINT = "https://api.sec-api.io"
SEC_EXTRACT_ENDPOINT = "https://api.sec-api.io/extractor"
DEFAULT_MARKET_MODE = "US"
SUPPORTED_MARKETS = {"US", "VN"}
DEFAULT_VNSTOCK_SOURCES = ("KBS",)
DEFAULT_VNSTOCK_NEWS_FETCH_LIMIT = 120
DEFAULT_VNSTOCK_NEWS_PAGE_SIZE = 20
DEFAULT_VNSTOCK_NEWS_MAX_PAGE = 20
MAX_VNSTOCK_NEWS_FETCH_LIMIT = 2000
MAX_KBS_NEWS_PAGE_SIZE = 20
DEFAULT_VN_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"


def _strip_wrapped_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_dotenv_compat() -> None:
    # Standard dotenv first.
    load_dotenv(override=False)

    # Fallback parser for files using spacing like: KEY = "value"
    env_candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.getcwd(), "..", ".env"),
    ]
    for env_path in env_candidates:
        if not os.path.exists(env_path):
            continue
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = _strip_wrapped_quotes(value.strip())
                if key and key not in os.environ:
                    os.environ[key] = value


def _resolve_market_mode(cli_market: Optional[str]) -> str:
    value = (
        cli_market
        or os.environ.get("FINMEM_MARKET_MODE")
        or os.environ.get("FINMEM_MARKET")
        or DEFAULT_MARKET_MODE
    )
    market = str(value).strip().upper().replace("-", "_")
    if market in {"US", "USA", "U.S.", "U_S"}:
        return "US"
    if market in {"VN", "VIETNAM", "VIET_NAM", "VNSE"}:
        return "VN"
    raise ValueError(
        f"Unsupported market '{value}'. Supported values: {sorted(SUPPORTED_MARKETS)}"
    )


def _resolve_bool_env(var_name: str, default: bool) -> bool:
    raw = (os.environ.get(var_name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _resolve_positive_int_env(
    var_name: str,
    default: int,
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> int:
    raw = (os.environ.get(var_name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < minimum:
        value = minimum
    if maximum is not None:
        value = min(value, maximum)
    return value


def _resolve_vnstock_sources() -> List[str]:
    raw = (os.environ.get("FINMEM_VNSTOCK_SOURCE") or "").strip()
    if raw and any(p.strip().upper() != "KBS" for p in raw.split(",") if p.strip()):
        print(
            "Warning: VN news pipeline supports KBS only; ignoring non-KBS FINMEM_VNSTOCK_SOURCE entries."
        )
    return list(DEFAULT_VNSTOCK_SOURCES)


def _resolve_vnstock_news_fetch_limit() -> int:
    return _resolve_positive_int_env(
        "FINMEM_VNSTOCK_NEWS_LIMIT",
        default=DEFAULT_VNSTOCK_NEWS_FETCH_LIMIT,
        minimum=1,
        maximum=MAX_VNSTOCK_NEWS_FETCH_LIMIT,
    )


def _resolve_vnstock_news_page_size() -> int:
    return _resolve_positive_int_env(
        "FINMEM_VNSTOCK_NEWS_PAGE_SIZE",
        default=DEFAULT_VNSTOCK_NEWS_PAGE_SIZE,
        minimum=1,
        maximum=MAX_KBS_NEWS_PAGE_SIZE,
    )


def _resolve_vnstock_news_max_page() -> int:
    return _resolve_positive_int_env(
        "FINMEM_VNSTOCK_NEWS_MAX_PAGE",
        default=DEFAULT_VNSTOCK_NEWS_MAX_PAGE,
        minimum=1,
        maximum=200,
    )


def _resolve_vn_news_align_window_days() -> int:
    raw = (os.environ.get("FINMEM_VN_NEWS_ALIGN_WINDOW_DAYS") or "").strip()
    if raw:
        try:
            value = int(raw)
            if value >= 0:
                return value
        except ValueError:
            pass
    return 3


def _normalize_text(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _build_news_headers() -> Dict[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET_KEY") or os.environ.get(
        "ALPACA_KEY_SECRET_KEY"
    )
    if not api_key or not api_secret:
        raise ValueError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET_KEY."
        )
    return {
        "Apca-Api-Key-Id": api_key,
        "Apca-Api-Secret-Key": api_secret,
    }


def _rotate_env_file_paths() -> List[str]:
    """Extra .env paths (comma-separated) with alternate Alpaca/SEC keys."""
    raw = (
        os.environ.get("FINMEM_ROTATE_ENV_FILES")
        or os.environ.get("FINMEM_ALPACA_ROTATE_ENV_FILES")
        or ""
    ).strip()
    out: List[str] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        path = p if os.path.isabs(p) else os.path.join(os.getcwd(), p)
        if os.path.isfile(path):
            out.append(path)
        else:
            print(f"Warning: FINMEM_ROTATE_ENV_FILES entry not found: {path}")
    return out


def _alpaca_headers_from_values(vals: Optional[Dict[str, Optional[str]]]) -> Optional[Dict[str, str]]:
    if not vals:
        return None
    api_key = vals.get("ALPACA_API_KEY") or vals.get("ALPACA_KEY")
    api_secret = vals.get("ALPACA_API_SECRET_KEY") or vals.get("ALPACA_KEY_SECRET_KEY")
    if not api_key or not api_secret:
        return None
    return {
        "Apca-Api-Key-Id": str(api_key).strip(),
        "Apca-Api-Secret-Key": str(api_secret).strip(),
    }


def _alpaca_news_headers_chain() -> List[Dict[str, str]]:
    """Primary env first, then Alpaca keys from FINMEM_ROTATE_ENV_FILES (deduped)."""
    primary = _build_news_headers()
    chain: List[Dict[str, str]] = [primary]
    seen = {primary["Apca-Api-Key-Id"]}
    for path in _rotate_env_file_paths():
        alt = _alpaca_headers_from_values(dotenv_values(path))
        if not alt:
            continue
        kid = alt["Apca-Api-Key-Id"]
        if kid in seen:
            continue
        seen.add(kid)
        chain.append(alt)
    return chain


def _get_sec_key() -> str:
    sec_key = os.environ.get("SEC_KEY")
    if not sec_key:
        raise ValueError("Missing SEC_KEY in environment. SEC filings are required.")
    return sec_key


def _sec_key_chain() -> List[str]:
    """Primary SEC_KEY from env, then SEC_KEY from each rotate env file (deduped)."""
    primary = _get_sec_key()
    chain: List[str] = [primary]
    seen = {primary}
    for path in _rotate_env_file_paths():
        vals = dotenv_values(path) or {}
        sk = vals.get("SEC_KEY")
        if not sk:
            continue
        sk = str(sk).strip()
        if sk and sk not in seen:
            seen.add(sk)
            chain.append(sk)
    return chain


def _is_api_quota_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return (
        "429" in s
        or "403" in s
        or "rate limit" in s
        or "too many requests" in s
        or "quota" in s
    )


def _key_rotate_sleep_seconds() -> float:
    try:
        return max(
            0.0,
            float((os.environ.get("FINMEM_KEY_ROTATE_SLEEP_SECONDS") or "2").strip()),
        )
    except ValueError:
        return 2.0


def _build_filing_maps_rotating(
    symbol: str,
    start_day: date,
    end_day: date,
    trading_days: List[date],
) -> Tuple[Dict[date, str], Dict[date, str]]:
    keys = _sec_key_chain()
    last_err: Optional[Exception] = None
    for ki, sec_key in enumerate(keys):
        try:
            return _build_filing_maps(
                symbol=symbol,
                start_day=start_day,
                end_day=end_day,
                trading_days=trading_days,
                sec_key=sec_key,
            )
        except ValueError as exc:
            last_err = exc
            if not _is_api_quota_error(exc) or ki + 1 >= len(keys):
                raise
            print(
                f"  SEC API rate/quota error; switching SEC key ({ki + 2}/{len(keys)})."
            )
            time.sleep(_key_rotate_sleep_seconds())
    if last_err:
        raise last_err
    raise ValueError("SEC filing download failed: no SEC keys configured")


def _download_prices_us(symbol: str, start_day: date, end_day: date) -> Dict[date, float]:
    df = yf.download(
        symbol,
        start=start_day.strftime("%Y-%m-%d"),
        end=(end_day + timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
    )
    if df.empty:
        raise ValueError("No price data downloaded from yfinance.")

    # yfinance returns either flat columns or MultiIndex columns depending on version.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        if ("Adj Close", symbol) in df.columns:
            price_series = df[("Adj Close", symbol)]
        elif ("Close", symbol) in df.columns:
            price_series = df[("Close", symbol)]
        else:
            raise ValueError("Unable to find adjusted/close price column in yfinance output")
    else:
        if "Adj Close" in df.columns:
            price_series = df["Adj Close"]
        elif "Close" in df.columns:
            price_series = df["Close"]
        else:
            raise ValueError("Unable to find adjusted/close price column in yfinance output")

    df = price_series.to_frame("price")
    df = df.reset_index()
    df["Date"] = df["Date"].dt.date
    prices: Dict[date, float] = {}
    for _, row in df.iterrows():
        prices[row["Date"]] = float(row["price"])
    return prices


def _download_prices_vn(
    symbol: str,
    start_day: date,
    end_day: date,
    sources: List[str],
) -> Dict[date, float]:
    try:
        from vnstock import Vnstock
    except ImportError as exc:
        raise ImportError(
            "vnstock is required for VN market mode. Install with: pip install vnstock"
        ) from exc

    last_error: Optional[Exception] = None
    df = None
    for source in sources:
        try:
            quote = Vnstock(show_log=False).stock(symbol=symbol, source=source).quote
            df = quote.history(
                start=start_day.strftime("%Y-%m-%d"),
                end=(end_day + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1D",
                show_log=False,
            )
            if df is not None and not df.empty:
                break
        except Exception as exc:  # pragma: no cover - network/provider dependent
            last_error = exc
            continue

    if df is None or df.empty:
        if last_error is not None:
            raise ValueError(
                f"No VN price data downloaded for {symbol} from sources={sources}. Last error: {last_error}"
            )
        raise ValueError(f"No VN price data downloaded for {symbol} from sources={sources}.")

    if "time" not in df.columns:
        raise ValueError("vnstock quote.history output missing 'time' column.")

    if "close" in df.columns:
        close_col = "close"
    elif "Close" in df.columns:
        close_col = "Close"
    else:
        raise ValueError("vnstock quote.history output missing close price column.")

    series = df[["time", close_col]].copy()
    series["time"] = series["time"].apply(
        lambda x: x.date() if isinstance(x, datetime) else date.fromisoformat(str(x)[:10])
    )
    series = series[(series["time"] >= start_day) & (series["time"] <= end_day)]

    prices: Dict[date, float] = {}
    for _, row in series.iterrows():
        prices[row["time"]] = float(row[close_col])

    if not prices:
        raise ValueError(
            f"No VN price data available for {symbol} in [{start_day}, {end_day}]"
        )
    return prices


def _download_prices(
    symbol: str,
    start_day: date,
    end_day: date,
    market_mode: str,
) -> Dict[date, float]:
    if market_mode == "US":
        return _download_prices_us(symbol=symbol, start_day=start_day, end_day=end_day)
    if market_mode == "VN":
        return _download_prices_vn(
            symbol=symbol,
            start_day=start_day,
            end_day=end_day,
            sources=_resolve_vnstock_sources(),
        )
    raise ValueError(f"Unsupported market mode: {market_mode}")


def _resolve_us_news_source() -> str:
    raw = (os.environ.get("FINMEM_US_NEWS_SOURCE") or "alpaca").strip().lower()
    if raw in {"alpaca", "openrouter", "auto"}:
        return raw
    raise ValueError(
        "FINMEM_US_NEWS_SOURCE must be one of: alpaca, openrouter, auto "
        f"(got {raw!r}). "
        "Note: auto uses Alpaca only (with FINMEM_ROTATE_ENV_FILES key rotation on 429); "
        "it does not fall back to OpenRouter."
    )


def _fetch_us_news_alpaca_for_day(
    client: httpx.Client,
    endpoint: str,
    headers: Dict[str, str],
    symbol: str,
    cur_day: date,
    max_news_per_day: int,
) -> List[str]:
    next_day = cur_day + timedelta(days=1)
    url = (
        f"{endpoint}?start={cur_day.strftime('%Y-%m-%d')}"
        f"&end={next_day.strftime('%Y-%m-%d')}"
        f"&limit=50&symbols={symbol}"
    )

    news_texts: List[str] = []
    seen = set()
    page_token: Optional[str] = None

    while True:
        request_url = (
            url
            if page_token is None
            else (
                f"{endpoint}?start={cur_day.strftime('%Y-%m-%d')}"
                f"&end={next_day.strftime('%Y-%m-%d')}"
                f"&limit=50&symbols={symbol}&page_token={page_token}"
            )
        )
        resp = client.get(request_url, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise ValueError(
                f"Alpaca news request failed: {resp.status_code} {resp.text}"
            )

        payload = resp.json()
        for item in payload.get("news", []):
            text = (
                item.get("summary")
                or item.get("headline")
                or item.get("content")
                or ""
            )
            text = _normalize_text(text)
            if not text or text in seen:
                continue
            seen.add(text)
            news_texts.append(text)
            if len(news_texts) >= max_news_per_day:
                return news_texts

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    return news_texts


def _openrouter_api_key() -> Optional[str]:
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")


_OPENROUTER_NEWS_RETRY_STATUSES = frozenset({429, 502, 503})


def _openrouter_news_model_chain() -> List[str]:
    """Primary model first, then fallbacks (deduped). Override fallbacks via env."""
    primary = (
        os.environ.get("FINMEM_OPENROUTER_MODEL") or DEFAULT_OPENROUTER_NEWS_MODEL
    ).strip()
    raw_fb = (os.environ.get("FINMEM_OPENROUTER_NEWS_MODEL_FALLBACKS") or "").strip()
    if raw_fb:
        fallbacks = [p.strip() for p in raw_fb.split(",") if p.strip()]
    else:
        fallbacks = list(DEFAULT_OPENROUTER_NEWS_FALLBACK_MODELS)
    seen: set = set()
    out: List[str] = []
    for m in [primary] + fallbacks:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _fetch_us_news_openrouter_for_day(
    client: httpx.Client,
    symbol: str,
    cur_day: date,
    max_news_per_day: int,
) -> List[str]:
    key = _openrouter_api_key()
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is required for OpenRouter US news. "
            "Set it in the environment or .env file."
        )
    models = _openrouter_news_model_chain()
    try:
        fb_sleep = float(
            (os.environ.get("FINMEM_OPENROUTER_MODEL_FALLBACK_SLEEP_SECONDS") or "3").strip()
        )
    except ValueError:
        fb_sleep = 3.0
    fb_sleep = max(0.0, fb_sleep)

    cap_raw = os.environ.get("FINMEM_OPENROUTER_MAX_SNIPPETS", "25").strip()
    try:
        snippet_cap = int(cap_raw)
    except ValueError:
        snippet_cap = 25
    snippet_cap = max(1, min(snippet_cap, 50))
    n = min(max_news_per_day, snippet_cap)
    user_prompt = (
        f'For the US stock ticker {symbol} on {cur_day.isoformat()}, '
        f"produce a JSON array of exactly {n} distinct short English strings. "
        f"Each string is one plausible financial news headline or one-sentence "
        f"market item for that calendar day (generic public-market context only; "
        f"do not claim non-public or insider facts). "
        f"Output nothing except the JSON array, no markdown."
    )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/pipiku915/FinMem-LLM-StockTrading",
        "X-Title": "FinMem data build",
    }
    last_fail: Optional[ValueError] = None
    for mi, model in enumerate(models):
        if mi > 0 and fb_sleep > 0:
            time.sleep(fb_sleep)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You reply with only valid JSON when the user asks for a JSON array.",
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.45,
            "max_tokens": min(4096, 80 * n + 200),
        }
        resp = client.post(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=120.0,
        )
        if resp.status_code == 200:
            break
        err = ValueError(
            f"OpenRouter request failed: {resp.status_code} {resp.text}"
        )
        last_fail = err
        retryable = resp.status_code in _OPENROUTER_NEWS_RETRY_STATUSES
        if retryable and mi + 1 < len(models):
            print(
                f"  OpenRouter model {model!r} returned {resp.status_code} on "
                f"{cur_day}; trying next model."
            )
            continue
        raise err
    else:
        if last_fail:
            raise last_fail
        raise ValueError("OpenRouter: no models configured")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"OpenRouter response missing message content: {data!r}") from exc
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```\s*$", "", content)
    try:
        arr = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenRouter returned non-JSON content: {content[:500]!r}") from exc
    if not isinstance(arr, list):
        raise ValueError("OpenRouter JSON must be an array of strings")
    news_texts: List[str] = []
    seen = set()
    for item in arr:
        text = _normalize_text(str(item))
        if not text or text in seen:
            continue
        seen.add(text)
        news_texts.append(text)
        if len(news_texts) >= max_news_per_day:
            break
    return news_texts


def _download_news(
    symbol: str,
    trading_days: List[date],
    endpoint: str,
    headers: Optional[Dict[str, str]],
    max_news_per_day: int,
    sleep_s: float,
) -> Dict[date, List[str]]:
    source = _resolve_us_news_source()
    news_by_day: Dict[date, List[str]] = {}
    alpaca_header_sets: Optional[List[Dict[str, str]]] = None
    if source in {"alpaca", "auto"}:
        alpaca_header_sets = _alpaca_news_headers_chain()
    with httpx.Client() as client:
        for i, cur_day in enumerate(trading_days, start=1):
            if source == "openrouter":
                texts = _fetch_us_news_openrouter_for_day(
                    client=client,
                    symbol=symbol,
                    cur_day=cur_day,
                    max_news_per_day=max_news_per_day,
                )
            elif source in {"alpaca", "auto"}:
                assert alpaca_header_sets is not None
                last_exc: Optional[ValueError] = None
                texts: List[str] = []
                for hi, hdr in enumerate(alpaca_header_sets):
                    try:
                        texts = _fetch_us_news_alpaca_for_day(
                            client=client,
                            endpoint=endpoint,
                            headers=hdr,
                            symbol=symbol,
                            cur_day=cur_day,
                            max_news_per_day=max_news_per_day,
                        )
                        last_exc = None
                        break
                    except ValueError as exc:
                        last_exc = exc
                        if not _is_api_quota_error(exc) or hi + 1 >= len(
                            alpaca_header_sets
                        ):
                            raise
                        print(
                            f"  Alpaca rate/quota on {cur_day}; "
                            f"switching Alpaca key ({hi + 2}/{len(alpaca_header_sets)})."
                        )
                        time.sleep(_key_rotate_sleep_seconds())
                if last_exc is not None:
                    raise last_exc
            else:  # pragma: no cover
                raise RuntimeError(f"Unexpected US news source: {source!r}")
            news_by_day[cur_day] = texts
            if i % 25 == 0:
                print(f"Fetched news for {i}/{len(trading_days)} trading days")
            if sleep_s > 0:
                time.sleep(sleep_s)
    return news_by_day


def _extract_vn_news_date(raw_value: object) -> Optional[date]:
    if raw_value is None:
        return None

    if isinstance(raw_value, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw_value) / 1000).date()
        except (OverflowError, ValueError):
            return None

    text = str(raw_value).strip()
    if not text:
        return None

    if text.isdigit():
        try:
            return datetime.fromtimestamp(float(text) / 1000).date()
        except (OverflowError, ValueError):
            return None

    try:
        return parser.parse(text).date()
    except (TypeError, ValueError):
        return None


def _extract_vn_news_text(record: Dict[str, object]) -> str:
    candidates = [
        record.get("news_short_content"),
        record.get("news_title"),
        record.get("news_sub_title"),
        record.get("news_full_content"),
        record.get("title"),
        record.get("head"),
        record.get("description"),
        record.get("content"),
    ]
    for value in candidates:
        if value:
            text = _normalize_text(str(value))
            if text:
                return text
    return ""


def _extract_vn_news_row_date(record: Dict[str, object]) -> Optional[date]:
    for field in (
        "public_date",
        "created_at",
        "publish_time",
        "published_at",
        "time",
        "date",
    ):
        value = _extract_vn_news_date(record.get(field))
        if value is not None:
            return value
    return None


def _align_vn_news_to_trading_day(
    raw_day: date,
    trading_days: List[date],
    align_window_days: int,
) -> Optional[date]:
    if not trading_days:
        return None

    first_day = trading_days[0]
    last_day = trading_days[-1]

    if raw_day > last_day:
        return None

    if raw_day < first_day and (first_day - raw_day).days > align_window_days:
        return None

    for trading_day in trading_days:
        if trading_day >= raw_day:
            if abs((trading_day - raw_day).days) > align_window_days:
                return None
            return trading_day
    return None


def _fetch_vnstock_news_kbs_rows(
    symbol: str,
    fetch_limit: int,
    page_size: int,
) -> Tuple[List[Dict[str, object]], int]:
    from vnstock import Company

    company = Company(symbol=symbol, source="KBS")
    all_rows: List[Dict[str, object]] = []
    seen_keys: set[Tuple[str, str]] = set()
    page = 1
    pages_fetched = 0
    max_page = _resolve_vnstock_news_max_page()
    consecutive_empty = 0

    while len(all_rows) < fetch_limit and page <= max_page:
        request_size = min(page_size, fetch_limit - len(all_rows))
        if request_size <= 0:
            break

        try:
            page_df = company.news(page=page, page_size=request_size, show_log=False)
        except TypeError:
            page_df = company.news(page, request_size, False)

        pages_fetched += 1

        if page_df is None or page_df.empty:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            page += 1
            continue

        page_rows = page_df.to_dict("records")
        if not page_rows:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            page += 1
            continue

        consecutive_empty = 0

        for row in page_rows:
            text = _extract_vn_news_text(row)
            raw_date = _extract_vn_news_row_date(row)
            if text or raw_date is not None:
                key = (raw_date.isoformat() if raw_date is not None else "", text)
            else:
                key = ("", _normalize_text(str(sorted(row.items()))))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_rows.append(row)
            if len(all_rows) >= fetch_limit:
                break

        page += 1

    return all_rows[:fetch_limit], pages_fetched


def _download_news_vn(
    symbol: str,
    trading_days: List[date],
    max_news_per_day: int,
) -> Dict[date, List[str]]:
    try:
        import vnstock  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vnstock is required for VN market mode. Install with: pip install vnstock"
        ) from exc

    fetch_limit = _resolve_vnstock_news_fetch_limit()
    page_size = _resolve_vnstock_news_page_size()
    align_window_days = _resolve_vn_news_align_window_days()
    try:
        raw_rows, pages_fetched = _fetch_vnstock_news_kbs_rows(
            symbol=symbol,
            fetch_limit=fetch_limit,
            page_size=page_size,
        )
    except Exception as exc:  # pragma: no cover - network/provider dependent
        print(
            f"Warning: VN news fetch failed for {symbol} using source ['KBS']: {exc}"
        )
        return {d: [] for d in trading_days}

    if not raw_rows:
        return {d: [] for d in trading_days}

    news_by_day: Dict[date, List[str]] = {d: [] for d in trading_days}
    seen_by_day: Dict[date, set[str]] = {d: set() for d in trading_days}
    seen_rows: set[Tuple[str, str]] = set()
    total_assigned = 0
    parsed_news_dates: List[date] = []
    drop_stats = {
        "missing_text": 0,
        "missing_date": 0,
        "duplicate_row": 0,
        "outside_alignment": 0,
        "day_cap": 0,
    }

    if max_news_per_day <= 0:
        return news_by_day

    for row in raw_rows:
        text = _extract_vn_news_text(row)
        if not text:
            drop_stats["missing_text"] += 1
            continue

        candidate_date = _extract_vn_news_row_date(row)
        if candidate_date is None:
            drop_stats["missing_date"] += 1
            continue
        parsed_news_dates.append(candidate_date)

        row_key = (candidate_date.isoformat(), text)
        if row_key in seen_rows:
            drop_stats["duplicate_row"] += 1
            continue
        seen_rows.add(row_key)

        target_day = _align_vn_news_to_trading_day(
            candidate_date,
            trading_days,
            align_window_days=align_window_days,
        )
        if target_day is None:
            drop_stats["outside_alignment"] += 1
            continue

        if text in seen_by_day[target_day]:
            drop_stats["duplicate_row"] += 1
            continue

        if len(news_by_day[target_day]) >= max_news_per_day:
            drop_stats["day_cap"] += 1
            continue

        seen_by_day[target_day].add(text)
        news_by_day[target_day].append(text)
        total_assigned += 1

    days_with_news = sum(1 for d in trading_days if news_by_day[d])
    if total_assigned == 0 and parsed_news_dates:
        print(
            "Warning: VN news rows were fetched but none mapped to trading days. "
            f"news_range={min(parsed_news_dates)}->{max(parsed_news_dates)}, "
            f"trading_range={trading_days[0]}->{trading_days[-1]}, "
            f"align_window_days={align_window_days}."
        )

    print(
        "VN news fetched: "
        f"source=KBS, pages={pages_fetched}, rows={len(raw_rows)}, unique_rows={len(seen_rows)}, "
        f"assigned={total_assigned}, trading_days_with_news={days_with_news}/{len(trading_days)}, "
        f"drops={drop_stats}"
    )

    return news_by_day


def _build_vi_to_en_translator() -> Optional[Callable[[str], str]]:
    if not _resolve_bool_env("FINMEM_VN_TRANSLATE_FOR_VADER", True):
        return None

    model_name = (os.environ.get("FINMEM_VN_TRANSLATION_MODEL") or DEFAULT_VN_TRANSLATION_MODEL).strip()
    local_only = _resolve_bool_env("FINMEM_VN_TRANSLATION_LOCAL_ONLY", True)
    max_length = _resolve_positive_int_env(
        "FINMEM_VN_TRANSLATION_MAX_LENGTH",
        default=256,
        minimum=32,
        maximum=1024,
    )

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_only,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            local_files_only=local_only,
        )
        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
        )
    except Exception as exc:  # pragma: no cover - model availability dependent
        mode = "local-only" if local_only else "local-or-remote"
        print(
            f"Warning: Could not initialize VI->EN translator ({model_name}, {mode}). "
            f"VADER will score original text. Error: {exc}"
        )
        return None

    cache: Dict[str, str] = {}
    warning_emitted = False

    def _translate(text: str) -> str:
        nonlocal warning_emitted
        normalized = _normalize_text(text)
        if not normalized:
            return ""

        cached = cache.get(normalized)
        if cached is not None:
            return cached

        try:
            result = translator(normalized, max_length=max_length, truncation=True)
            translated = ""
            if isinstance(result, list) and result:
                translated = _normalize_text(str(result[0].get("translation_text", "")))
            if not translated:
                translated = normalized
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            if not warning_emitted:
                print(
                    "Warning: Runtime VI->EN translation failed; falling back to original text. "
                    f"Error: {exc}"
                )
                warning_emitted = True
            translated = normalized

        cache[normalized] = translated
        return translated

    return _translate


def _append_vader_scores(
    news_list: List[str],
    analyzer: SentimentIntensityAnalyzer,
    translator: Optional[Callable[[str], str]] = None,
) -> List[str]:
    out = []
    for text in news_list:
        score_text = translator(text) if translator is not None else text
        if not score_text:
            score_text = text
        scores = analyzer.polarity_scores(score_text)
        out.append(
            f"{text} The positive score for this news is {scores['pos']}. "
            f"The neutral score for this news is {scores['neu']}. "
            f"The negative score for this news is {scores['neg']}."
        )
    return out


def _fetch_sec_index(
    client: httpx.Client,
    sec_key: str,
    symbol: str,
    form_type: str,
    start_day: date,
    end_day: date,
    page_size: int = 50,
) -> List[Dict[str, str]]:
    all_records: List[Dict[str, str]] = []
    from_idx = 0
    query_range = (
        f"ticker:{symbol} AND formType:\"{form_type}\" "
        f"AND filedAt:[{start_day.strftime('%Y-%m-%d')} TO {end_day.strftime('%Y-%m-%d')}]"
    )
    while True:
        payload = {
            "query": {"query_string": {"query": query_range}},
            "from": str(from_idx),
            "size": str(page_size),
            "sort": [{"filedAt": {"order": "desc"}}],
        }
        resp = client.post(
            f"{SEC_QUERY_ENDPOINT}?token={sec_key}",
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            raise ValueError(f"SEC query failed: {resp.status_code} {resp.text}")
        result = resp.json()
        filings = result.get("filings", [])
        if not filings:
            break

        for filing in filings:
            filed_at = filing.get("filedAt")
            document_url = None
            for doc in filing.get("documentFormatFiles", []):
                if doc.get("type") == form_type:
                    document_url = doc.get("documentUrl")
                    break
            if filed_at and document_url:
                all_records.append({"filedAt": filed_at, "documentUrl": document_url})

        from_idx += page_size

    return all_records


def _extract_sec_section(
    client: httpx.Client,
    sec_key: str,
    filing_url: str,
    section_item: str,
) -> Optional[str]:
    resp = client.get(
        f"{SEC_EXTRACT_ENDPOINT}?url={filing_url}&token={sec_key}&item={section_item}",
        timeout=90,
    )
    if resp.status_code != 200:
        return None
    txt = _normalize_text(resp.text)
    return txt if txt else None


def _align_to_trading_day(raw_day: date, trading_days: List[date]) -> Optional[date]:
    if raw_day in trading_days:
        return raw_day
    for d in trading_days:
        if d > raw_day:
            return d
    return None


def _build_filing_maps(
    symbol: str,
    start_day: date,
    end_day: date,
    trading_days: List[date],
    sec_key: str,
) -> Tuple[Dict[date, str], Dict[date, str]]:
    ten_k_item = "7"
    ten_q_item = "part1item2"
    us_eastern = pytz.timezone("US/Eastern")

    filing_k_map: Dict[date, str] = {}
    filing_q_map: Dict[date, str] = {}

    with httpx.Client() as client:
        ten_k_idx = _fetch_sec_index(
            client=client,
            sec_key=sec_key,
            symbol=symbol,
            form_type="10-K",
            start_day=start_day,
            end_day=end_day,
        )
        ten_q_idx = _fetch_sec_index(
            client=client,
            sec_key=sec_key,
            symbol=symbol,
            form_type="10-Q",
            start_day=start_day,
            end_day=end_day,
        )

        print(f"Found {len(ten_k_idx)} filings for 10-K and {len(ten_q_idx)} filings for 10-Q")

        for item in ten_k_idx:
            raw_ts = parser.parse(item["filedAt"])
            est_dt = raw_ts.astimezone(us_eastern).replace(tzinfo=None)
            target_day = _align_to_trading_day(est_dt.date(), trading_days)
            if not target_day:
                continue
            text = _extract_sec_section(
                client=client,
                sec_key=sec_key,
                filing_url=item["documentUrl"],
                section_item=ten_k_item,
            )
            if text:
                filing_k_map[target_day] = text

        for item in ten_q_idx:
            raw_ts = parser.parse(item["filedAt"])
            est_dt = raw_ts.astimezone(us_eastern).replace(tzinfo=None)
            target_day = _align_to_trading_day(est_dt.date(), trading_days)
            if not target_day:
                continue
            text = _extract_sec_section(
                client=client,
                sec_key=sec_key,
                filing_url=item["documentUrl"],
                section_item=ten_q_item,
            )
            if text:
                filing_q_map[target_day] = text

    return filing_k_map, filing_q_map


def build_market_input(
    symbol: str,
    start_day: date,
    end_day: date,
    output_path: str,
    market_mode: str = DEFAULT_MARKET_MODE,
    apply_vader: bool = True,
    max_news_per_day: int = 200,
    sleep_s: float = 0.0,
) -> None:
    _load_dotenv_compat()
    market_mode = _resolve_market_mode(market_mode)

    print(f"Step 1/5: Downloading price data (market={market_mode})")
    prices = _download_prices(
        symbol=symbol,
        start_day=start_day,
        end_day=end_day,
        market_mode=market_mode,
    )
    trading_days = sorted(prices.keys())
    print(f"Trading days: {len(trading_days)} | Range: {trading_days[0]} -> {trading_days[-1]}")

    if market_mode == "US":
        us_src = _resolve_us_news_source()
        if us_src == "openrouter":
            print("Step 2/5: US news via OpenRouter (LLM-generated snippets)")
            news_endpoint = ALPACA_DEFAULT_NEWS_ENDPOINT
        elif us_src == "auto":
            print(
                "Step 2/5: US news via Alpaca (switch Alpaca keys on rate limits; "
                "see FINMEM_ROTATE_ENV_FILES)"
            )
            news_endpoint = os.environ.get(
                "ALPACA_NEWS_ENDPOINT", ALPACA_DEFAULT_NEWS_ENDPOINT
            ).rstrip("/")
        else:
            print("Step 2/5: Downloading Alpaca news")
            if _rotate_env_file_paths():
                print(
                    "  Alternate Alpaca keys loaded from FINMEM_ROTATE_ENV_FILES "
                    "(used on 429 / quota errors)."
                )
            news_endpoint = os.environ.get(
                "ALPACA_NEWS_ENDPOINT", ALPACA_DEFAULT_NEWS_ENDPOINT
            ).rstrip("/")
        news_by_day = _download_news(
            symbol=symbol,
            trading_days=trading_days,
            endpoint=news_endpoint,
            headers=None,
            max_news_per_day=max_news_per_day,
            sleep_s=sleep_s,
        )
    else:
        print("Step 2/5: Downloading VN company news via vnstock")
        news_by_day = _download_news_vn(
            symbol=symbol,
            trading_days=trading_days,
            max_news_per_day=max_news_per_day,
        )

    if apply_vader:
        translator: Optional[Callable[[str], str]] = None
        if market_mode == "VN":
            print("Step 3/5: Translating VN news to English and appending VADER sentiment")
            translator = _build_vi_to_en_translator()
            if translator is None:
                print("Warning: VN translation unavailable/disabled; scoring original VN text with VADER.")
        else:
            print("Step 3/5: Appending VADER sentiment to news")

        analyzer = SentimentIntensityAnalyzer()
        for d in trading_days:
            news_by_day[d] = _append_vader_scores(
                news_by_day.get(d, []),
                analyzer,
                translator=translator,
            )
    else:
        print("Step 3/5: Skipped sentiment augmentation")

    if market_mode == "US":
        print("Step 4/5: Downloading SEC filings (10-K Item 7, 10-Q part1item2)")
        if len(_sec_key_chain()) > 1:
            print(
                "  Multiple SEC keys available (FINMEM_ROTATE_ENV_FILES); "
                "switching if the API returns rate/quota errors."
            )
        filing_k_map, filing_q_map = _build_filing_maps_rotating(
            symbol=symbol,
            start_day=start_day,
            end_day=end_day,
            trading_days=trading_days,
        )
    else:
        print("Step 4/5: Skipping SEC filings for VN market")
        filing_k_map, filing_q_map = {}, {}

    print("Step 5/5: Building runtime env_data dictionary")
    env_data: Dict[date, Dict[str, Dict[str, object]]] = {}
    for d in trading_days:
        env_data[d] = {
            "price": {symbol: prices[d]},
            "news": {symbol: news_by_day.get(d, [])},
            "filing_q": {symbol: filing_q_map[d]} if d in filing_q_map else {},
            "filing_k": {symbol: filing_k_map[d]} if d in filing_k_map else {},
        }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(env_data, f)

    with_news = sum(1 for d in trading_days if len(env_data[d]["news"][symbol]) > 0)
    with_k = sum(1 for d in trading_days if len(env_data[d]["filing_k"]) > 0)
    with_q = sum(1 for d in trading_days if len(env_data[d]["filing_q"]) > 0)
    print(f"Saved: {output_path}")
    print(f"Days with news: {with_news}/{len(trading_days)}")
    print(f"Days with 10-K: {with_k}/{len(trading_days)}")
    print(f"Days with 10-Q: {with_q}/{len(trading_days)}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Build FinMem paper-style market input data")
    arg_parser.add_argument(
        "--symbol",
        default=os.environ.get("FINMEM_TRADING_SYMBOL", "TSLA"),
        help="Ticker symbol (for example TSLA, AAPL)",
    )
    arg_parser.add_argument(
        "--market",
        default=os.environ.get("FINMEM_MARKET_MODE") or os.environ.get("FINMEM_MARKET") or DEFAULT_MARKET_MODE,
        help="Market mode: US or VN. Environment fallback: FINMEM_MARKET_MODE / FINMEM_MARKET",
    )
    arg_parser.add_argument(
        "--start",
        default=os.environ.get("FINMEM_BUILD_START", "2021-08-17"),
        help="Start date YYYY-MM-DD",
    )
    arg_parser.add_argument(
        "--end",
        default=os.environ.get("FINMEM_BUILD_END", "2023-04-10"),
        help="End date YYYY-MM-DD",
    )
    arg_parser.add_argument(
        "--output-path",
        default=None,
        help="Output pickle path. Default: data/03_model_input/<symbol>.pkl",
    )
    arg_parser.add_argument(
        "--max-news-per-day",
        type=int,
        default=int(os.environ.get("FINMEM_MAX_NEWS_PER_DAY", "200")),
    )
    arg_parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(os.environ.get("FINMEM_NEWS_SLEEP_SECONDS", "0.0")),
    )
    arg_parser.add_argument(
        "--disable-vader",
        action="store_true",
        help="Disable VADER sentiment augmentation",
    )
    args = arg_parser.parse_args()

    symbol = args.symbol.upper()
    output_path = args.output_path or os.path.join(
        "data",
        "03_model_input",
        f"{symbol.lower()}.pkl",
    )
    build_market_input(
        symbol=symbol,
        start_day=date.fromisoformat(args.start),
        end_day=date.fromisoformat(args.end),
        output_path=output_path,
        market_mode=args.market,
        apply_vader=not args.disable_vader,
        max_news_per_day=args.max_news_per_day,
        sleep_s=args.sleep_seconds,
    )