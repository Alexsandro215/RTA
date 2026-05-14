import argparse
import time

import pandas as pd

from app.data.ccxt_provider import CCXTMarketDataProvider
from app.data.csv_storage import CsvMarketDataStorage
from app.data.market_data_provider import MarketDataError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch normalized crypto OHLCV data using ccxt."
    )
    parser.add_argument(
        "--exchange",
        default="binance",
        help="Exchange id supported by ccxt. Default: binance",
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Market symbol to fetch. Default: BTC/USDT",
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Candle timeframe. Examples: 1m, 5m, 1h, 1d. Default: 1h",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of candles to fetch. Default: 10",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously fetch and print the latest candle.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between fetches when --watch is enabled. Default: 10",
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Download historical candles from --since to --until using pagination.",
    )
    parser.add_argument(
        "--since",
        help="Start date or datetime for --historical. Example: 2024-01-01",
    )
    parser.add_argument(
        "--until",
        help="End date or datetime for --historical. Default: now / exchange latest",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=1000,
        help="Candles per historical request. Default: 1000",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Optional safety cap for historical pagination batches.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save fetched candles into data/raw as CSV without duplicated timestamps.",
    )
    parser.add_argument(
        "--dataset-info",
        action="store_true",
        help="Inspect the local CSV dataset without calling the exchange.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=5,
        help="Rows to print with --dataset-info. Default: 5",
    )
    return parser.parse_args()


def print_fetch_header(args: argparse.Namespace) -> None:
    print(
        "Fetched OHLCV data "
        f"exchange={args.exchange} symbol={args.symbol} "
        f"timeframe={args.timeframe} limit={args.limit}"
    )


def print_latest_candle(data: pd.DataFrame, args: argparse.Namespace) -> None:
    latest = data.iloc[-1]
    print(
        f"{args.exchange} | {args.symbol} | {args.timeframe} | "
        f"{latest['datetime']} | "
        f"open={latest['open']} high={latest['high']} "
        f"low={latest['low']} close={latest['close']} "
        f"volume={latest['volume']}"
    )


def save_data(
    data: pd.DataFrame,
    storage: CsvMarketDataStorage,
    args: argparse.Namespace,
) -> None:
    path = storage.save(
        data=data,
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    saved = storage.load(args.exchange, args.symbol, args.timeframe)
    print(f"Saved {len(saved)} total candles to {path}")


def fetch_data(
    provider: CCXTMarketDataProvider,
    args: argparse.Namespace,
) -> pd.DataFrame:
    return provider.fetch_ohlcv(
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
    )


def run_once(
    provider: CCXTMarketDataProvider,
    storage: CsvMarketDataStorage,
    args: argparse.Namespace,
) -> None:
    data = fetch_data(provider, args)
    print_fetch_header(args)
    print(data)
    if args.save:
        save_data(data, storage, args)


def run_watch(
    provider: CCXTMarketDataProvider,
    storage: CsvMarketDataStorage,
    args: argparse.Namespace,
) -> None:
    if args.interval <= 0:
        raise ValueError("interval must be greater than 0")

    print(
        "Watching OHLCV data "
        f"exchange={args.exchange} symbol={args.symbol} "
        f"timeframe={args.timeframe} limit={args.limit} "
        f"interval={args.interval}s"
    )
    print("Press Ctrl+C to stop.")

    while True:
        try:
            data = fetch_data(provider, args)
            print_latest_candle(data, args)
            if args.save:
                save_data(data, storage, args)
        except MarketDataError as exc:
            print(f"Market data error: {exc}")

        time.sleep(args.interval)


def run_historical(
    provider: CCXTMarketDataProvider,
    storage: CsvMarketDataStorage,
    args: argparse.Namespace,
) -> None:
    if not args.since:
        raise ValueError("--since is required when --historical is enabled")

    data = provider.fetch_ohlcv_history(
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=args.since,
        until=args.until,
        limit_per_request=args.batch_limit,
        max_batches=args.max_batches,
    )

    print(
        "Fetched historical OHLCV data "
        f"exchange={args.exchange} symbol={args.symbol} "
        f"timeframe={args.timeframe} rows={len(data)} "
        f"from={data.iloc[0]['datetime']} to={data.iloc[-1]['datetime']}"
    )

    if args.save:
        save_data(data, storage, args)
    else:
        print(data)


def run_dataset_info(storage: CsvMarketDataStorage, args: argparse.Namespace) -> None:
    if args.tail < 0:
        raise ValueError("tail must be greater than or equal to 0")

    path = storage.get_path(args.exchange, args.symbol, args.timeframe)
    data = storage.load(args.exchange, args.symbol, args.timeframe)

    print(
        "Local dataset "
        f"exchange={args.exchange} symbol={args.symbol} "
        f"timeframe={args.timeframe}"
    )
    print(f"path={path}")

    if data.empty:
        print("rows=0")
        print("status=not found or empty")
        return

    print(f"rows={len(data)}")
    print(f"from={data.iloc[0]['datetime']}")
    print(f"to={data.iloc[-1]['datetime']}")

    if args.tail > 0:
        print(data.tail(args.tail))


def main() -> None:
    args = parse_args()
    provider = CCXTMarketDataProvider()
    storage = CsvMarketDataStorage()

    try:
        if args.dataset_info:
            run_dataset_info(storage, args)
        elif args.historical:
            run_historical(provider, storage, args)
        elif args.watch:
            run_watch(provider, storage, args)
        else:
            run_once(provider, storage, args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except MarketDataError as exc:
        print(f"Market data error: {exc}")
    except ValueError as exc:
        print(f"Invalid input: {exc}")


if __name__ == "__main__":
    main()
