"""
NIFTY INTRADAY OPTIONS BOT — WEBSOCKET EDITION (1H CHART)
==========================================================
Broker: Dhan | Capital: Rs 15,000 | 1 Trade/Day | 1 Lot (75 qty)
Strategy: EMA 9/21 Cross + Supertrend + RSI + EMA 50
Data: WebSocket for live ticks + REST for historical candles
Expiry: Weekly Tuesday

COMPATIBLE WITH: dhanhq >= 2.1.0 (DhanContext API)

SETUP:
    pip install dhanhq==2.1.0 pandas numpy
    Get API: https://web.dhan.co → Profile → API Access

USAGE:
    python nifty_bot.py              (live trading)
    python nifty_bot.py --paper      (paper/dry run)

DISCLAIMER: Educational only. Trading involves risk of loss.
"""

import time
import threading
import datetime
import logging
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from dhanhq import DhanContext, dhanhq, MarketFeed, OrderUpdate

# ============================================================================
# CONFIG — EDIT THESE TWO LINES
# ============================================================================
CLIENT_ID    = "YOUR_CLIENT_ID"
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"

# ============================================================================
# STRATEGY PARAMS (matches Pine Script exactly)
# ============================================================================
EMA_FAST        = 9
EMA_SLOW        = 21
EMA_BIAS        = 50
ST_PERIOD       = 10
ST_MULT         = 2.0
RSI_PERIOD      = 14
RSI_LONG_MIN    = 55
RSI_LONG_MAX    = 68
RSI_SHORT_MIN   = 32
RSI_SHORT_MAX   = 45
MIN_BODY        = 10       # min candle body in points

# ============================================================================
# RISK MANAGEMENT
# ============================================================================
SL_PTS          = 45       # stop loss in Nifty points
TP_PTS          = 90       # take profit in Nifty points
USE_TRAIL       = True
TRAIL_ACTIVATE  = 60       # activate trailing after this profit
TRAIL_OFFSET    = 30       # trail behind price by this much

# ============================================================================
# OPTION SETTINGS
# ============================================================================
STRIKE_GAP      = 50       # Nifty strikes are 50 pts apart
OPTION_MODE     = "ATM"    # ATM / ITM1 / OTM1
LOT_SIZE        = 75       # 1 Nifty lot = 75 qty (updated from 65)
MAX_TRADES_DAY  = 1        # max 1 trade per day

# ============================================================================
# SESSION TIMINGS (IST)
# ============================================================================
MARKET_OPEN     = datetime.time(9, 15)
ENTRY_START     = datetime.time(10, 0)   # first signal check after 10 AM
ENTRY_END       = datetime.time(13, 30)  # no new entries after 1:30 PM
SQOFF_TIME      = datetime.time(15, 20)  # auto square off at 3:20 PM
MARKET_CLOSE    = datetime.time(15, 30)

# ============================================================================
# INSTRUMENT
# ============================================================================
NIFTY_SEC_ID    = '13'     # Nifty 50 Index security ID on Dhan
NIFTY_SYMBOL    = 'NIFTY'
CAPITAL         = 15000

# ============================================================================
# DHAN v2 API STRING CONSTANTS
# (v2.1+ uses string params, NOT class constants like dhan.NSE)
# ============================================================================
SEG_NSE_EQ      = "NSE_EQ"
SEG_NSE_FNO     = "NSE_FNO"
TXN_BUY         = "BUY"
TXN_SELL        = "SELL"
ORD_MARKET      = "MARKET"
ORD_LIMIT       = "LIMIT"
PROD_INTRADAY   = "INTRADAY"
PROD_CNC        = "CNC"

# ============================================================================
# LOGGING
# ============================================================================
os.makedirs("logs", exist_ok=True)
log_file = f"logs/bot_{datetime.date.today()}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("NiftyBot")

# ============================================================================
# DHAN CONNECTION (v2.1+ DhanContext pattern)
# ============================================================================
dhan_ctx = None
dhan = None

def connect_dhan():
    """Initialize Dhan API using DhanContext (v2.1+ pattern)"""
    global dhan_ctx, dhan
    dhan_ctx = DhanContext(CLIENT_ID, ACCESS_TOKEN)
    dhan = dhanhq(dhan_ctx)
    log.info("Connected to Dhan API (v2.1+ DhanContext)")
    return dhan_ctx, dhan

# ============================================================================
# INDICATORS (exact same math as Pine Script)
# ============================================================================

def calc_ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = avg_loss.replace(0, 1e-10)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_supertrend(df, period=10, mult=2.0):
    """Supertrend indicator"""
    hl2 = (df['high'] + df['low']) / 2
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()

    upper_band = hl2 + mult * atr
    lower_band = hl2 - mult * atr

    st = pd.Series(np.nan, index=df.index)
    direction = pd.Series(0, index=df.index)

    for i in range(period, len(df)):
        if i == period:
            st.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
            continue

        prev_dir = direction.iloc[i - 1]
        prev_st = st.iloc[i - 1]

        if prev_dir == -1:  # was bearish
            if df['close'].iloc[i] > prev_st:
                direction.iloc[i] = 1
                st.iloc[i] = lower_band.iloc[i]
            else:
                direction.iloc[i] = -1
                st.iloc[i] = min(upper_band.iloc[i], prev_st)
        else:  # was bullish
            if df['close'].iloc[i] < prev_st:
                direction.iloc[i] = -1
                st.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = 1
                st.iloc[i] = max(lower_band.iloc[i], prev_st)

    return st, direction

# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_signal(df):
    """
    Check for BUY/SELL signal on latest candle.
    Returns 'LONG', 'SHORT', or None.
    """
    if df is None or len(df) < EMA_BIAS + 5:
        log.warning(f"Need {EMA_BIAS + 5} candles, have {len(df) if df is not None else 0}")
        return None

    df = df.copy()
    df['ema_fast'] = calc_ema(df['close'], EMA_FAST)
    df['ema_slow'] = calc_ema(df['close'], EMA_SLOW)
    df['ema_bias'] = calc_ema(df['close'], EMA_BIAS)
    df['rsi'] = calc_rsi(df['close'], RSI_PERIOD)
    df['st_val'], df['st_dir'] = calc_supertrend(df, ST_PERIOD, ST_MULT)
    df['body'] = (df['close'] - df['open']).abs()

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    bull_cross = prev['ema_fast'] <= prev['ema_slow'] and curr['ema_fast'] > curr['ema_slow']
    bear_cross = prev['ema_fast'] >= prev['ema_slow'] and curr['ema_fast'] < curr['ema_slow']

    log.info(
        f"SCAN | Close:{curr['close']:.0f} EMA9:{curr['ema_fast']:.0f} "
        f"EMA21:{curr['ema_slow']:.0f} EMA50:{curr['ema_bias']:.0f} "
        f"RSI:{curr['rsi']:.1f} ST:{'BULL' if curr['st_dir'] == 1 else 'BEAR'} "
        f"Body:{curr['body']:.0f} BullX:{bull_cross} BearX:{bear_cross}"
    )

    # LONG signal
    if (bull_cross
            and curr['close'] > curr['ema_bias']
            and curr['st_dir'] == 1
            and RSI_LONG_MIN < curr['rsi'] < RSI_LONG_MAX
            and curr['body'] > MIN_BODY):
        log.info(">>> LONG SIGNAL <<<")
        return 'LONG'

    # SHORT signal
    if (bear_cross
            and curr['close'] < curr['ema_bias']
            and curr['st_dir'] == -1
            and RSI_SHORT_MIN < curr['rsi'] < RSI_SHORT_MAX
            and curr['body'] > MIN_BODY):
        log.info(">>> SHORT SIGNAL <<<")
        return 'SHORT'

    return None

# ============================================================================
# LIVE PRICE (updated by WebSocket thread)
# ============================================================================

class LivePrice:
    def __init__(self):
        self._lock = threading.Lock()
        self._ltp = 0.0
        self._updated_at = None
        self._bars_1h = []
        self._current_bar = {'time': None, 'o': 0, 'h': 0, 'l': 0, 'c': 0}

    def update(self, price):
        now = datetime.datetime.now()
        with self._lock:
            self._ltp = price
            self._updated_at = now
            self._build_candle(price, now)

    def get(self):
        with self._lock:
            return self._ltp

    def is_stale(self, max_age_secs=30):
        with self._lock:
            if self._updated_at is None:
                return True
            return (datetime.datetime.now() - self._updated_at).total_seconds() > max_age_secs

    def _build_candle(self, price, now):
        """Accumulate ticks into 1H OHLC bars"""
        bar_time = now.replace(minute=0, second=0, microsecond=0)

        if self._current_bar['time'] != bar_time:
            if self._current_bar['time'] is not None and self._current_bar['o'] > 0:
                self._bars_1h.append(dict(self._current_bar))
                if len(self._bars_1h) > 100:
                    self._bars_1h = self._bars_1h[-100:]
            self._current_bar = {'time': bar_time, 'o': price, 'h': price, 'l': price, 'c': price}
        else:
            self._current_bar['h'] = max(self._current_bar['h'], price)
            self._current_bar['l'] = min(self._current_bar['l'], price)
            self._current_bar['c'] = price

    def get_1h_dataframe(self):
        """Return completed 1H bars as DataFrame"""
        with self._lock:
            bars = list(self._bars_1h)

        if not bars:
            return None

        df = pd.DataFrame(bars)
        df = df.rename(columns={'time': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df['volume'] = 0
        return df

live_price = LivePrice()

# ============================================================================
# HISTORICAL DATA (REST API — for indicator warmup)
# v2 replaced historical_minute_charts with intraday_minute_data
# v2 intraday API: security_id, exchange_segment, instrument_type
# v2 timestamps are EPOCH-based
# ============================================================================

def _parse_v2_candle_data(data):
    """
    Parse v2 API response which returns:
    {'open': [...], 'high': [...], 'low': [...], 'close': [...], 'volume': [...], 'start_Time': [...]}
    OR older format: {'data': {'open': [...], ...}}
    The start_Time values are EPOCH timestamps in v2.
    """
    raw = None

    # Handle nested 'data' key
    if isinstance(data, dict):
        if 'data' in data:
            raw = data['data']
        elif 'open' in data:
            raw = data
        else:
            return None

    if raw is None or not isinstance(raw, dict):
        return None

    # v2 response: parallel arrays
    timestamps = raw.get('start_Time') or raw.get('timestamp') or raw.get('start_time', [])
    opens = raw.get('open', [])
    highs = raw.get('high', [])
    lows = raw.get('low', [])
    closes = raw.get('close', [])
    volumes = raw.get('volume', [])

    if not opens or len(opens) == 0:
        return None

    n = len(opens)
    volumes = volumes if len(volumes) == n else [0] * n

    df = pd.DataFrame({
        'timestamp': timestamps[:n],
        'open': opens[:n],
        'high': highs[:n],
        'low': lows[:n],
        'close': closes[:n],
        'volume': volumes[:n]
    })

    # Convert EPOCH timestamps (seconds or milliseconds)
    ts_sample = df['timestamp'].iloc[0] if len(df) > 0 else 0
    if isinstance(ts_sample, (int, float)):
        if ts_sample > 1e12:  # milliseconds
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:  # seconds
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Localize to IST if naive
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Kolkata')
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)

    df = df.set_index('timestamp').sort_index()

    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['open', 'close'])
    return df


def fetch_historical_1h():
    """
    Fetch past 1H candles via REST for EMA 50 warmup.
    Uses v2 intraday_minute_data (supports 1/5/15/25/60 min intervals,
    last 5 trading days) with fallback approaches.
    """
    # ── METHOD 1: intraday_minute_data (v2 — last 5 days, multiple intervals) ──
    try:
        log.info("Fetching intraday data via intraday_minute_data...")
        data = dhan.intraday_minute_data(
            security_id=NIFTY_SEC_ID,
            exchange_segment=SEG_NSE_EQ,
            instrument_type='INDEX'
        )
        if data:
            df = _parse_v2_candle_data(data)
            if df is not None and len(df) > 0:
                # Resample 1-min data to 1H
                df_1h = df.resample('1h').agg({
                    'open': 'first', 'high': 'max',
                    'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()
                df_1h = df_1h.between_time('09:00', '15:30')
                log.info(f"Intraday API: {len(df_1h)} hourly candles loaded")
                if len(df_1h) >= 10:
                    return df_1h
    except Exception as e:
        log.error(f"intraday_minute_data error: {e}")

    # ── METHOD 2: historical_minute_charts (v1-style, may still work) ──
    today = datetime.date.today()
    from_date = today - datetime.timedelta(days=20)

    try:
        log.info("Trying historical_minute_charts fallback...")
        data = dhan.historical_minute_charts(
            symbol=NIFTY_SYMBOL,
            exchange_segment=SEG_NSE_EQ,
            instrument_type='INDEX',
            expiry_code=0,
            from_date=str(from_date),
            to_date=str(today)
        )
        if data:
            df = _parse_v2_candle_data(data)
            if df is not None and len(df) > 0:
                df_1h = df.resample('1h').agg({
                    'open': 'first', 'high': 'max',
                    'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()
                df_1h = df_1h.between_time('09:00', '15:30')
                log.info(f"Historical API: {len(df_1h)} hourly candles loaded")
                if len(df_1h) >= 10:
                    return df_1h
    except AttributeError:
        log.warning("historical_minute_charts not available in this dhanhq version")
    except Exception as e:
        log.error(f"historical_minute_charts error: {e}")

    # ── METHOD 3: Market Quote LTP fallback ──
    try:
        log.info("Trying market_quote LTP fallback...")
        ltp_data = dhan.get_market_quote(
            security_id=NIFTY_SEC_ID,
            exchange_segment=SEG_NSE_EQ,
            instrument_type='INDEX'
        )
        if ltp_data:
            log.info(f"Market quote response received (LTP fallback only)")
    except Exception as e:
        log.warning(f"Market quote fallback error: {e}")

    log.warning("All historical data methods exhausted — relying on WebSocket candle builder")
    return None


def get_combined_1h_data():
    """Combine historical REST data + live WebSocket candles"""
    hist_df = fetch_historical_1h()
    ws_df = live_price.get_1h_dataframe()

    if hist_df is not None and ws_df is not None:
        combined = pd.concat([hist_df, ws_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        return combined.sort_index()
    elif hist_df is not None:
        return hist_df
    elif ws_df is not None:
        return ws_df
    return None

# ============================================================================
# WEBSOCKET THREADS (v2.1+ DhanContext pattern)
# ============================================================================

def start_market_feed():
    """
    WebSocket thread for real-time Nifty price ticks.
    v2.1+: MarketFeed(dhan_context, instruments, version)
    Official pattern: run_forever() → get_data() in a loop
    """
    instruments = [(MarketFeed.NSE, NIFTY_SEC_ID, MarketFeed.Ticker)]

    def feed_loop():
        while True:
            feed = None
            try:
                feed = MarketFeed(dhan_ctx, instruments, version="v2")
                log.info("WS-FEED: Connected")
                while True:
                    feed.run_forever()
                    data = feed.get_data()
                    if data and isinstance(data, dict):
                        ltp = data.get('LTP', 0)
                        if ltp and float(ltp) > 0:
                            live_price.update(float(ltp))
                    time.sleep(0.1)
            except Exception as e:
                log.warning(f"WS-FEED disconnected: {e} — reconnecting in 5s")
                if feed:
                    try:
                        feed.disconnect()
                    except Exception:
                        pass
                time.sleep(5)

    t = threading.Thread(target=feed_loop, daemon=True, name="MarketFeed")
    t.start()
    log.info("WS-FEED: Thread started")


def start_order_feed():
    """
    WebSocket thread for real-time order status updates.
    v2.1+: OrderUpdate(dhan_context) with on_update callback
    """
    def order_loop():
        while True:
            try:
                order_ws = OrderUpdate(dhan_ctx)

                def on_update(order_data):
                    d = order_data.get('Data', {})
                    log.info(
                        f"WS-ORDER: {d.get('orderId', '?')} → "
                        f"{d.get('orderStatus', '?')} | "
                        f"{d.get('tradingSymbol', '?')} | "
                        f"Qty:{d.get('quantity', '?')} | "
                        f"Price:{d.get('price', '?')}"
                    )

                order_ws.on_update = on_update
                order_ws.connect_to_dhan_websocket_sync()
            except Exception as e:
                log.warning(f"WS-ORDER disconnected: {e} — reconnecting in 5s")
                time.sleep(5)

    t = threading.Thread(target=order_loop, daemon=True, name="OrderFeed")
    t.start()
    log.info("WS-ORDER: Thread started")

# ============================================================================
# OPTION HELPERS
# ============================================================================

def get_atm_strike(spot):
    """Round spot to nearest strike"""
    return round(spot / STRIKE_GAP) * STRIKE_GAP

def pick_option(spot, direction):
    """
    Pick the correct option based on signal direction.
    LONG → buy CE, SHORT → buy PE
    """
    atm = get_atm_strike(spot)

    if direction == 'LONG':
        if OPTION_MODE == "ATM":
            return atm, "CE"
        elif OPTION_MODE == "ITM1":
            return atm - STRIKE_GAP, "CE"
        else:
            return atm + STRIKE_GAP, "CE"
    else:
        if OPTION_MODE == "ATM":
            return atm, "PE"
        elif OPTION_MODE == "ITM1":
            return atm + STRIKE_GAP, "PE"
        else:
            return atm - STRIKE_GAP, "PE"

def get_next_tuesday_expiry():
    """Get nearest Tuesday expiry. If today is Tuesday after 3:30, get next Tuesday."""
    today = datetime.date.today()
    days_to_tuesday = (1 - today.weekday()) % 7
    if days_to_tuesday == 0:
        if datetime.datetime.now().time() > datetime.time(15, 30):
            days_to_tuesday = 7
    expiry = today + datetime.timedelta(days=days_to_tuesday)
    return expiry

def find_option_security_id(strike, opt_type, expiry):
    """
    Search Dhan instrument list for matching Nifty option contract.
    Uses fetch_security_list("compact") which returns instrument master.
    """
    try:
        result = dhan.fetch_security_list("compact")
        if not result or not isinstance(result, dict) or not result.get('data'):
            log.error("Failed to fetch instrument list")
            return None

        expiry_str = expiry.strftime('%Y-%m-%d')
        strike_int = int(strike)

        instruments = result['data']
        log.info(f"Searching {len(instruments)} instruments for NIFTY {strike_int} {opt_type} exp:{expiry_str}")

        # Exact match first
        for inst in instruments:
            sym = str(inst.get('tradingSymbol', ''))
            exp = str(inst.get('expiryDate', ''))
            seg = str(inst.get('exchangeSegment', ''))

            if (seg in ('NSE_FNO', 'NSE_FO', '2')
                    and NIFTY_SYMBOL in sym
                    and str(strike_int) in sym
                    and opt_type in sym
                    and expiry_str in exp):
                sec_id = str(inst['securityId'])
                log.info(f"OPTION FOUND: {sym} | SecID: {sec_id} | Exp: {exp}")
                return sec_id

        # Fallback: closest upcoming expiry
        log.warning(f"Exact match not found — searching closest expiry...")
        candidates = []
        today_str = str(datetime.date.today())

        for inst in instruments:
            sym = str(inst.get('tradingSymbol', ''))
            exp = str(inst.get('expiryDate', ''))
            seg = str(inst.get('exchangeSegment', ''))

            if (seg in ('NSE_FNO', 'NSE_FO', '2')
                    and NIFTY_SYMBOL in sym
                    and str(strike_int) in sym
                    and opt_type in sym
                    and exp >= today_str):
                candidates.append(inst)

        if candidates:
            candidates.sort(key=lambda x: x.get('expiryDate', ''))
            best = candidates[0]
            sec_id = str(best['securityId'])
            log.info(f"CLOSEST MATCH: {best['tradingSymbol']} | SecID: {sec_id} | Exp: {best['expiryDate']}")
            return sec_id

        log.error(f"No option found for NIFTY {strike_int} {opt_type}")
        return None

    except Exception as e:
        log.error(f"Instrument search error: {e}")
        return None

# ============================================================================
# TRADE MANAGER
# ============================================================================

class TradeManager:
    STATE_FILE = "trade_state.json"

    def __init__(self, paper=False):
        self.paper = paper
        self.position = None
        self.entry_spot = 0.0
        self.trail_sl = 0.0
        self.strike = 0
        self.opt_type = None
        self.sec_id = None
        self.order_id = None
        self.entry_time = None
        self.traded_today = False
        self.today_date = None
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self._load_state()

    def _save_state(self):
        state = {
            'position': self.position,
            'entry_spot': self.entry_spot,
            'trail_sl': self.trail_sl,
            'strike': self.strike,
            'opt_type': self.opt_type,
            'sec_id': self.sec_id,
            'order_id': self.order_id,
            'entry_time': str(self.entry_time) if self.entry_time else None,
            'traded_today': self.traded_today,
            'today_date': self.today_date,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'total_pnl': self.total_pnl
        }
        try:
            with open(self.STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.error(f"State save error: {e}")

    def _load_state(self):
        if not os.path.exists(self.STATE_FILE):
            return
        try:
            with open(self.STATE_FILE) as f:
                s = json.load(f)
            self.position = s.get('position')
            self.entry_spot = s.get('entry_spot', 0)
            self.trail_sl = s.get('trail_sl', 0)
            self.strike = s.get('strike', 0)
            self.opt_type = s.get('opt_type')
            self.sec_id = s.get('sec_id')
            self.order_id = s.get('order_id')
            self.traded_today = s.get('traded_today', False)
            self.today_date = s.get('today_date')
            self.total_trades = s.get('total_trades', 0)
            self.wins = s.get('wins', 0)
            self.losses = s.get('losses', 0)
            self.total_pnl = s.get('total_pnl', 0)
            log.info(f"STATE LOADED: pos={self.position} trades={self.total_trades} "
                     f"pnl={self.total_pnl:+.0f}pts")
        except Exception as e:
            log.warning(f"State load error: {e}")

    def check_new_day(self):
        today = str(datetime.date.today())
        if self.today_date != today:
            self.traded_today = False
            self.today_date = today
            log.info("═" * 55)
            log.info(f"  NEW TRADING DAY: {today}")
            log.info("═" * 55)
            self._save_state()

    def can_trade(self):
        if self.position is not None:
            return False
        if self.traded_today:
            return False
        return True

    def enter(self, signal, spot):
        """
        Place entry order for 1 lot of ATM option.
        v2.1+: place_order uses STRING params, not class constants.
        """
        if not self.can_trade():
            log.info(f"Cannot trade: pos={self.position} traded={self.traded_today}")
            return False

        strike, opt_type = pick_option(spot, signal)
        expiry = get_next_tuesday_expiry()

        log.info(f"{'=' * 45}")
        log.info(f"ENTRY SIGNAL: {signal}")
        log.info(f"Spot: {spot:.0f} | Option: NIFTY {strike} {opt_type}")
        log.info(f"Expiry: {expiry} (Tuesday)")
        log.info(f"Qty: {LOT_SIZE} (1 lot)")
        log.info(f"{'=' * 45}")

        if self.paper:
            self.order_id = f"PAPER_{int(time.time())}"
            self.sec_id = "PAPER"
            log.info(f"[PAPER] BUY {LOT_SIZE} qty NIFTY {strike} {opt_type}")
        else:
            sec_id = find_option_security_id(strike, opt_type, expiry)
            if sec_id is None:
                log.error("Option contract not found — SKIPPING trade")
                return False

            try:
                # v2.1+ place_order: string params
                resp = dhan.place_order(
                    security_id=str(sec_id),
                    exchange_segment=SEG_NSE_FNO,
                    transaction_type=TXN_BUY,
                    quantity=LOT_SIZE,
                    order_type=ORD_MARKET,
                    product_type=PROD_INTRADAY,
                    price=0
                )

                # Handle response — may be dict or have 'data' key
                order_id = None
                if isinstance(resp, dict):
                    order_id = resp.get('orderId') or resp.get('data', {}).get('orderId')

                if not order_id:
                    log.error(f"ORDER FAILED: {resp}")
                    return False

                self.order_id = str(order_id)
                self.sec_id = str(sec_id)
                log.info(f"ORDER PLACED: ID={self.order_id}")

            except Exception as e:
                log.error(f"ORDER ERROR: {e}")
                return False

        self.position = signal
        self.entry_spot = spot
        self.strike = strike
        self.opt_type = opt_type
        self.traded_today = True
        self.entry_time = datetime.datetime.now()

        if signal == 'LONG':
            self.trail_sl = spot - SL_PTS
        else:
            self.trail_sl = spot + SL_PTS

        tp = spot + TP_PTS if signal == 'LONG' else spot - TP_PTS
        log.info(f"SL: {self.trail_sl:.0f} | TP: {tp:.0f} | Trail: {'ON' if USE_TRAIL else 'OFF'}")
        self._save_state()
        return True

    def check_exit(self, spot):
        """Check SL / TP / Trailing SL — called on every tick"""
        if self.position is None:
            return

        if self.position == 'LONG':
            unrealized = spot - self.entry_spot

            if USE_TRAIL and unrealized >= TRAIL_ACTIVATE:
                new_sl = spot - TRAIL_OFFSET
                if new_sl > self.trail_sl:
                    log.info(f"TRAIL SL: {self.trail_sl:.0f} → {new_sl:.0f} "
                             f"(profit: {unrealized:.0f} pts)")
                    self.trail_sl = new_sl
                    self._save_state()

            if spot <= self.trail_sl:
                self._execute_exit("SL HIT", spot)
            elif spot >= self.entry_spot + TP_PTS:
                self._execute_exit("TP HIT", spot)

        elif self.position == 'SHORT':
            unrealized = self.entry_spot - spot

            if USE_TRAIL and unrealized >= TRAIL_ACTIVATE:
                new_sl = spot + TRAIL_OFFSET
                if new_sl < self.trail_sl:
                    log.info(f"TRAIL SL: {self.trail_sl:.0f} → {new_sl:.0f} "
                             f"(profit: {unrealized:.0f} pts)")
                    self.trail_sl = new_sl
                    self._save_state()

            if spot >= self.trail_sl:
                self._execute_exit("SL HIT", spot)
            elif spot <= self.entry_spot - TP_PTS:
                self._execute_exit("TP HIT", spot)

    def _execute_exit(self, reason, spot):
        """Place exit order and update stats"""
        if self.position == 'LONG':
            pnl_pts = spot - self.entry_spot
        else:
            pnl_pts = self.entry_spot - spot

        pnl_rs = pnl_pts * LOT_SIZE

        log.info(f"{'=' * 45}")
        log.info(f"EXIT: {reason}")
        log.info(f"Entry: {self.entry_spot:.0f} | Exit: {spot:.0f}")
        log.info(f"P&L: {pnl_pts:+.0f} pts | Rs {pnl_rs:+.0f}")
        log.info(f"Option: NIFTY {self.strike} {self.opt_type}")
        log.info(f"{'=' * 45}")

        if not self.paper and self.sec_id and self.sec_id != "PAPER":
            try:
                # v2.1+ place_order: string params for exit
                resp = dhan.place_order(
                    security_id=str(self.sec_id),
                    exchange_segment=SEG_NSE_FNO,
                    transaction_type=TXN_SELL,
                    quantity=LOT_SIZE,
                    order_type=ORD_MARKET,
                    product_type=PROD_INTRADAY,
                    price=0
                )
                order_id = None
                if isinstance(resp, dict):
                    order_id = resp.get('orderId') or resp.get('data', {}).get('orderId')

                if order_id:
                    log.info(f"EXIT ORDER: ID={order_id}")
                else:
                    log.error(f"EXIT ORDER FAILED: {resp}")
            except Exception as e:
                log.error(f"EXIT ERROR: {e}")
        else:
            log.info(f"[PAPER] SELL {LOT_SIZE} qty NIFTY {self.strike} {self.opt_type}")

        self.total_trades += 1
        self.total_pnl += pnl_pts
        if pnl_pts > 0:
            self.wins += 1
        else:
            self.losses += 1

        self.position = None
        self.sec_id = None
        self.order_id = None
        self.entry_time = None
        self._save_state()
        self._print_stats()

    def square_off(self, spot):
        if self.position is not None:
            self._execute_exit("EOD SQUARE OFF 3:20 PM", spot)

    def _print_stats(self):
        pnl_rs = self.total_pnl * LOT_SIZE
        wr = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        log.info(f"╔═══ LIFETIME STATS ═══╗")
        log.info(f"║ Trades : {self.total_trades}")
        log.info(f"║ Wins   : {self.wins}")
        log.info(f"║ Losses : {self.losses}")
        log.info(f"║ WinRate: {wr:.0f}%")
        log.info(f"║ P&L    : {self.total_pnl:+.0f} pts")
        log.info(f"║ P&L Rs : Rs {pnl_rs:+,.0f}")
        log.info(f"║ Capital: Rs {CAPITAL + pnl_rs:,.0f}")
        log.info(f"╚═══════════════════════╝")

    def print_status(self):
        pnl_rs = self.total_pnl * LOT_SIZE
        wr = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
        pos_str = f"{self.position} {self.strike}{self.opt_type}" if self.position else "FLAT"
        log.info(f"STATUS | {pos_str} | Traded:{self.traded_today} | "
                 f"T:{self.total_trades} W:{self.wins} L:{self.losses} WR:{wr:.0f}% | "
                 f"PnL:{self.total_pnl:+.0f}pts Rs:{pnl_rs:+,.0f}")

# ============================================================================
# TIME HELPERS
# ============================================================================

def is_market_open():
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE

def is_entry_window():
    return ENTRY_START <= datetime.datetime.now().time() <= ENTRY_END

def is_sqoff_time():
    return datetime.datetime.now().time() >= SQOFF_TIME

def secs_to_next_hour():
    now = datetime.datetime.now()
    next_hr = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    return (next_hr - now).total_seconds()

# ============================================================================
# VERSION CHECK
# ============================================================================

def check_dhanhq_version():
    """Warn if dhanhq version may be incompatible"""
    try:
        import dhanhq as dh_mod
        ver = getattr(dh_mod, '__version__', 'unknown')
        log.info(f"dhanhq version: {ver}")
        if ver != 'unknown':
            major_minor = tuple(int(x) for x in ver.split('.')[:2])
            if major_minor < (2, 1):
                log.warning(f"dhanhq {ver} detected — this bot requires >= 2.1.0")
                log.warning("Run: pip install --upgrade dhanhq")
    except Exception:
        pass

# ============================================================================
# MAIN BOT LOOP
# ============================================================================

def run(paper=False):
    mode = "PAPER" if paper else "LIVE"
    log.info("╔" + "═" * 53 + "╗")
    log.info(f"║  NIFTY OPTIONS BOT — {mode} — WEBSOCKET — 1H")
    log.info(f"║  Capital: Rs {CAPITAL:,} | 1 Lot = {LOT_SIZE} qty")
    log.info(f"║  Strategy: EMA {EMA_FAST}/{EMA_SLOW}/{EMA_BIAS} + Supertrend + RSI")
    log.info(f"║  SL:{SL_PTS} TP:{TP_PTS} Trail:{TRAIL_ACTIVATE}/{TRAIL_OFFSET}")
    log.info(f"║  Entry: {ENTRY_START}-{ENTRY_END} | SqOff: {SQOFF_TIME}")
    log.info(f"║  Expiry: Tuesday | Strike: {OPTION_MODE}")
    log.info(f"║  dhanhq >= 2.1.0 (DhanContext + string params)")
    log.info("╚" + "═" * 53 + "╝")

    check_dhanhq_version()
    connect_dhan()
    start_market_feed()
    start_order_feed()

    tm = TradeManager(paper=paper)

    # Wait for first tick from WebSocket
    log.info("Waiting for live price from WebSocket...")
    timeout = 60
    for i in range(timeout):
        if live_price.get() > 0:
            log.info(f"First tick received: {live_price.get():.0f}")
            break
        time.sleep(1)
    else:
        log.warning(f"No tick after {timeout}s — will use REST data as fallback")

    last_checked_hour = -1

    while True:
        try:
            tm.check_new_day()

            if not is_market_open():
                log.info("Market closed — waiting 60s")
                time.sleep(60)
                continue

            spot = live_price.get()

            # Fallback if WebSocket is stale
            if spot <= 0 or live_price.is_stale(max_age_secs=60):
                log.warning("WebSocket stale — fetching via REST")
                try:
                    df_tmp = fetch_historical_1h()
                    if df_tmp is not None and len(df_tmp) > 0:
                        spot = float(df_tmp['close'].iloc[-1])
                except Exception:
                    pass
                if spot <= 0:
                    time.sleep(10)
                    continue

            # ── EOD SQUARE OFF at 3:20 PM ──
            if is_sqoff_time():
                tm.square_off(spot)
                log.info("Past square-off — sleeping until market close")
                time.sleep(300)
                continue

            # ── REAL-TIME SL/TP CHECK (every tick) ──
            if tm.position is not None:
                tm.check_exit(spot)

            # ── HOURLY SIGNAL CHECK ──
            now = datetime.datetime.now()
            current_hour = now.hour

            if current_hour != last_checked_hour and now.minute >= 1:
                last_checked_hour = current_hour
                log.info(f"─── 1H CANDLE CLOSED @ {now.strftime('%H:%M')} ───")

                df = get_combined_1h_data()

                if df is not None and len(df) >= EMA_BIAS + 5:
                    log.info(f"Candles available: {len(df)}")

                    if is_entry_window() and tm.can_trade():
                        signal = generate_signal(df)
                        if signal:
                            tm.enter(signal, spot)
                        else:
                            log.info("No signal this candle")
                    else:
                        reason = []
                        if not is_entry_window():
                            reason.append("outside entry window")
                        if not tm.can_trade():
                            reason.append("already traded" if tm.traded_today else "in position")
                        log.info(f"Skipping signal check: {', '.join(reason)}")
                else:
                    log.warning(f"Insufficient candles: {len(df) if df is not None else 0} "
                                f"(need {EMA_BIAS + 5})")

                tm.print_status()

            time.sleep(1)

        except KeyboardInterrupt:
            log.info("")
            log.info("Bot stopped by user — squaring off...")
            spot = live_price.get()
            if spot > 0:
                tm.square_off(spot)
            log.info("Goodbye!")
            break

        except Exception as e:
            log.error(f"UNEXPECTED ERROR: {e}", exc_info=True)
            time.sleep(10)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nifty Options Auto-Trading Bot")
    parser.add_argument("--paper", action="store_true", help="Paper trading (no real orders)")
    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════════════╗
    ║   NIFTY OPTIONS AUTO-TRADING BOT                 ║
    ║   WebSocket Edition | 1H Chart | Dhan API v2.1+  ║
    ║   Capital: Rs 15,000 | 1 Lot (75 qty) | 1/Day   ║
    ╠═══════════════════════════════════════════════════╣
    ║   Strategy: EMA 9/21/50 + Supertrend + RSI       ║
    ║   SL: 45 pts | TP: 90 pts | Trailing SL: ON     ║
    ║   Entry: 10:00-13:30 | SqOff: 15:20              ║
    ║   Expiry: Tuesday | Strike: ATM                  ║
    ╠═══════════════════════════════════════════════════╣
    ║   Requires: pip install dhanhq>=2.1.0 pandas np  ║
    ║   --paper  = dry run (no real orders)            ║
    ║   Ctrl+C   = stop + auto square-off              ║
    ╚═══════════════════════════════════════════════════╝
    """)

    if CLIENT_ID == "YOUR_CLIENT_ID":
        print("  ERROR: Set your CLIENT_ID and ACCESS_TOKEN in the script!")
        print("  Get them: https://web.dhan.co → Profile → API Access")
        print()
        sys.exit(1)

    run(paper=args.paper)
