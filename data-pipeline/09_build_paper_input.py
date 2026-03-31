import os
import re
import time
import pickle
import argparse
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Optional

import httpx
import yfinance as yf
import pytz
from dateutil import parser
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


ALPACA_DEFAULT_NEWS_ENDPOINT = "https://data.alpaca.markets/v1beta1/news"
SEC_QUERY_ENDPOINT = "https://api.sec-api.io"
SEC_EXTRACT_ENDPOINT = "https://api.sec-api.io/extractor"


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


def _get_sec_key() -> str:
    sec_key = os.environ.get("SEC_KEY")
    if not sec_key:
        raise ValueError("Missing SEC_KEY in environment. SEC filings are required.")
    return sec_key


def _download_prices(symbol: str, start_day: date, end_day: date) -> Dict[date, float]:
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


def _fetch_news_for_day(
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
        request_url = url if page_token is None else f"{endpoint}?limit=50&symbols={symbol}&page_token={page_token}"
        resp = client.get(request_url, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise ValueError(f"Alpaca news request failed: {resp.status_code} {resp.text}")

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


def _download_news(
    symbol: str,
    trading_days: List[date],
    endpoint: str,
    headers: Dict[str, str],
    max_news_per_day: int,
    sleep_s: float,
) -> Dict[date, List[str]]:
    news_by_day: Dict[date, List[str]] = {}
    with httpx.Client() as client:
        for i, cur_day in enumerate(trading_days, start=1):
            texts = _fetch_news_for_day(
                client=client,
                endpoint=endpoint,
                headers=headers,
                symbol=symbol,
                cur_day=cur_day,
                max_news_per_day=max_news_per_day,
            )
            news_by_day[cur_day] = texts
            if i % 25 == 0:
                print(f"Fetched news for {i}/{len(trading_days)} trading days")
            if sleep_s > 0:
                time.sleep(sleep_s)
    return news_by_day


def _append_vader_scores(news_list: List[str], analyzer: SentimentIntensityAnalyzer) -> List[str]:
    out = []
    for text in news_list:
        scores = analyzer.polarity_scores(text)
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
    apply_vader: bool = True,
    max_news_per_day: int = 200,
    sleep_s: float = 0.0,
) -> None:
    load_dotenv()
    sec_key = _get_sec_key()

    print("Step 1/5: Downloading price data")
    prices = _download_prices(symbol=symbol, start_day=start_day, end_day=end_day)
    trading_days = sorted(prices.keys())
    print(f"Trading days: {len(trading_days)} | Range: {trading_days[0]} -> {trading_days[-1]}")

    print("Step 2/5: Downloading Alpaca news")
    news_endpoint = os.environ.get("ALPACA_NEWS_ENDPOINT", ALPACA_DEFAULT_NEWS_ENDPOINT).rstrip("/")
    headers = _build_news_headers()
    news_by_day = _download_news(
        symbol=symbol,
        trading_days=trading_days,
        endpoint=news_endpoint,
        headers=headers,
        max_news_per_day=max_news_per_day,
        sleep_s=sleep_s,
    )

    if apply_vader:
        print("Step 3/5: Appending VADER sentiment to news")
        analyzer = SentimentIntensityAnalyzer()
        for d in trading_days:
            news_by_day[d] = _append_vader_scores(news_by_day.get(d, []), analyzer)
    else:
        print("Step 3/5: Skipped sentiment augmentation")

    print("Step 4/5: Downloading SEC filings (10-K Item 7, 10-Q part1item2)")
    filing_k_map, filing_q_map = _build_filing_maps(
        symbol=symbol,
        start_day=start_day,
        end_day=end_day,
        trading_days=trading_days,
        sec_key=sec_key,
    )

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
        apply_vader=not args.disable_vader,
        max_news_per_day=args.max_news_per_day,
        sleep_s=args.sleep_seconds,
    )