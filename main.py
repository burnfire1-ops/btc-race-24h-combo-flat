#!/usr/bin/env python3
"""COMBO-FLAT bot for the make.fun BTC Race 24h (btc-race-24h-20261117-0603).

Strategy: entry is the I2c dip-buy (1m close 75bps below SMA90(1m) with
15m RSI(2) < 20), OR the #3c legs (>=2 of 4 on 15m closes) but only while
the market is flat (15m ATR < 20 bps of price). Exit is TP +3%, SL 1%,
a trailing stop that arms at +2% (give back 0.5%), and a 4h time stop.
Long only, max 1 open position, all-in buys, full-qty sells.

Indicator classes are copied verbatim from the backtest reference
(backtest_spike.py / backtest_spike2.py); the seeding matters.
Standard library only. No network, no wall clock, no unseeded random.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"
FEE_BPS = 4
SLIP_BPS = 2


# ---------------------------------------------------------------------------
# Indicator classes, copied VERBATIM from backtest_spike.py / backtest_spike2.py
# ---------------------------------------------------------------------------

class Ema:
    """House convention: value = mean of first span closes, then 2/(span+1) steps."""

    def __init__(self, span):
        self.span = span
        self.warm = []
        self.value = None

    def push(self, x):
        if self.value is not None:
            self.value += (2.0 / (self.span + 1)) * (x - self.value)
            return self.value
        self.warm.append(x)
        if len(self.warm) == self.span:
            self.value = sum(self.warm) / self.span
        return self.value


class Rsi:
    """Wilder's RSI(14): initial avg gain/loss over first 14 closes."""

    def __init__(self, span=14):
        self.span = span
        self.prev = None
        self.gains = []
        self.losses = []
        self.ag = None
        self.al = None
        self.value = None

    def push(self, x):
        if self.prev is None:
            self.prev = x
            return None
        ch = x - self.prev
        self.prev = x
        g = max(ch, 0.0)
        l = max(-ch, 0.0)
        if self.ag is None:
            self.gains.append(g)
            self.losses.append(l)
            if len(self.gains) == self.span:
                self.ag = sum(self.gains) / self.span
                self.al = sum(self.losses) / self.span
            return None
        self.ag = (self.ag * (self.span - 1) + g) / self.span
        self.al = (self.al * (self.span - 1) + l) / self.span
        self.value = 100.0 if self.al == 0 else 100.0 - 100.0 / (1.0 + self.ag / self.al)
        return self.value


class Bollinger:
    def __init__(self, span=20, mult=2.0):
        self.span = span
        self.mult = mult
        self.win = []

    def push(self, x):
        self.win.append(x)
        if len(self.win) > self.span:
            self.win.pop(0)
        if len(self.win) < self.span:
            return None
        m = sum(self.win) / self.span
        var = sum((v - m) ** 2 for v in self.win) / self.span
        sd = math.sqrt(var)
        return (m, m + self.mult * sd, m - self.mult * sd, 2 * self.mult * sd)


class Atr:
    def __init__(self, span=14):
        self.span = span
        self.prev_close = None
        self.trs = []
        self.value = None

    def push(self, h, l, c):
        if self.prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        self.prev_close = c
        if self.value is not None:
            self.value = (self.value * (self.span - 1) + tr) / self.span
            return self.value
        self.trs.append(tr)
        if len(self.trs) == self.span:
            self.value = sum(self.trs) / self.span
        return self.value


class Ind15:
    """All 15m indicators, updated once per completed 15m bar."""

    def __init__(self):
        self.ema12 = Ema(12)
        self.ema26 = Ema(26)
        self.ema9 = Ema(9)      # signal line on MACD
        self.ema20 = Ema(20)
        self.rsi = Rsi(14)
        self.bb = Bollinger(20, 2.0)
        self.atr = Atr(14)
        self.macd = None        # (line, signal, hist)
        self.macd_prev = None
        self.rsi_prev = None
        self.bb_hist = []       # bandwidth history for squeeze percentile
        self.closes = []
        # supply/demand zones: dicts {bottom, top, bar, kind}
        self.bars15 = []        # (o, h, l, c) per completed 15m bar
        self.demand = []        # live demand zones
        self.supply = []        # live supply zones

    def _zones_update(self):
        n = len(self.bars15)
        # zones expire after 48 15m bars (12h)
        self.demand = [z for z in self.demand if n - z["bar"] <= 48]
        self.supply = [z for z in self.supply if n - z["bar"] <= 48]
        if n < 5:
            return
        # swing low at bar n-3 (lowest low of bars n-5..n-1, confirmed by 2 bars)
        j = n - 3
        lo = self.bars15[j][2]
        if lo == min(b[2] for b in self.bars15[j - 2:j + 3]):
            c_now = self.bars15[-1][3]
            if c_now >= lo * 1.006:  # strong rally away (>=0.6%)
                self.demand.append({"bottom": lo, "top": lo * 1.002,
                                    "bar": n})
        hi = self.bars15[j][1]
        if hi == max(b[1] for b in self.bars15[j - 2:j + 3]):
            c_now = self.bars15[-1][3]
            if c_now <= hi * 0.994:  # strong drop away (>=0.6%)
                self.supply.append({"bottom": hi * 0.998, "top": hi,
                                    "bar": n})

    def push(self, o, h, l, c):
        self.closes.append(c)
        self.bars15.append((o, h, l, c))
        self._zones_update()
        e12 = self.ema12.push(c)
        e26 = self.ema26.push(c)
        sig = None
        if e12 is not None and e26 is not None:
            line = e12 - e26
            sig = self.ema9.push(line)
            self.macd_prev = self.macd
            self.macd = (line, sig, line - sig if sig is not None else None)
        self.ema20.push(c)
        self.rsi_prev = self.rsi.value
        r = self.rsi.push(c)
        bb = self.bb.push(c)
        if bb is not None:
            self.bb_hist.append(bb[3])
            if len(self.bb_hist) > 20:
                self.bb_hist.pop(0)
        self.atr.push(h, l, c)
        return {"macd": self.macd, "macd_prev": self.macd_prev, "rsi": r,
                "rsi_prev": self.rsi_prev, "bb": bb, "atr": self.atr.value,
                "ema20": self.ema20.value}


class Ind15b(Ind15):
    """Ind15 + RSI(2), ATR-ratio, relative-bandwidth percentile, Donchian."""

    def __init__(self):
        super().__init__()
        self.rsi2 = Rsi(2)
        self.rsi2_prev = None
        self.atr_hist = []
        self.bw_rel_hist = []

    def push(self, o, h, l, c):
        d = super().push(o, h, l, c)
        self.rsi2_prev = self.rsi2.value
        d["rsi2"] = self.rsi2.push(c)
        d["rsi2_prev"] = self.rsi2_prev
        if self.atr.value is not None:
            self.atr_hist.append(self.atr.value)
        bb = d["bb"]
        if bb is not None:
            rel = bb[3] / bb[0] if bb[0] else 0.0
            self.bw_rel_hist.append(rel)
            d["bw_rel"] = rel
        else:
            d["bw_rel"] = None
        cl = self.closes
        d["sma5"] = sum(cl[-5:]) / 5 if len(cl) >= 5 else None
        d["sma20"] = sum(cl[-20:]) / 20 if len(cl) >= 20 else None
        d["atr_ratio"] = None
        if len(self.atr_hist) >= 20 and self.atr.value:
            sma_atr = sum(self.atr_hist[-20:]) / 20
            d["atr_ratio"] = self.atr.value / sma_atr if sma_atr else None
        # Donchian levels EXCLUDE the current bar (else unreachable).
        hs = [x[1] for x in self.bars15]
        ls = [x[2] for x in self.bars15]
        d["donch20_high"] = max(hs[-21:-1]) if len(hs) >= 21 else None
        d["donch10_low"] = min(ls[-11:-1]) if len(ls) >= 11 else None
        hist = self.bw_rel_hist
        if len(hist) >= 20:
            win = sorted(hist[-50:])
            d["bw_pct"] = sum(1 for v in win if v <= hist[-1]) / len(win)
        else:
            d["bw_pct"] = None
        return d


# ---------------------------------------------------------------------------
# COMBO-FLAT bot
# ---------------------------------------------------------------------------

class ComboFlatBot:
    def __init__(self):
        self.price_scale = None
        self.quote_scale = None
        self.ema60 = Ema(60)
        self.ema60_hist = []
        self.ind = Ind15b()
        self.closes_1m = []
        self.c15 = []   # 1-min closes accumulating into the current 15m bar
        self.h15 = []
        self.l15 = []
        self.m15 = None  # dict from the latest completed 15m bar, else None
        self.trend = False
        self.sma90 = None
        self.pos = None  # {"entry_px","entry_bar","highest","stop_px"}
        self.cash_before = None  # cash when the buy was decided
        self.pending_stop = None  # stop computed at buy decision time

    def on_hello(self, state):
        sym = state.hello["symbols"][0]
        self.price_scale = sym["price_scale"]
        self.quote_scale = state.hello["capital"]["quote_scale"]

    def on_batch(self, state, batch):
        bar = state.latest.get(SYMBOL)
        if bar is None:
            return []
        ps = 10 ** self.price_scale
        o = int(bar["o"]) / ps
        h = int(bar["h"]) / ps
        l = int(bar["l"]) / ps
        c = int(bar["c"]) / ps
        o_depth_usd = int(bar["o_depth"][0]) / 1e6

        # --- update indicators (every batch, warmup included) ---
        e60 = self.ema60.push(c)
        self.ema60_hist.append(e60)
        self.closes_1m.append(c)
        self.c15.append(c)
        self.h15.append(h)
        self.l15.append(l)
        self.m15 = None
        if len(self.c15) == 15:
            self.m15 = self.ind.push(self.c15[0], max(self.h15),
                                     min(self.l15), self.c15[-1])
            self.c15, self.h15, self.l15 = [], [], []
        trend = False
        if e60 is not None and len(self.ema60_hist) > 60:
            prev60 = self.ema60_hist[-61]
            trend = c > e60 and prev60 is not None and e60 > prev60
        self.trend = trend
        self.sma90 = (sum(self.closes_1m[-90:]) / 90
                      if len(self.closes_1m) >= 90 else None)

        holding = state.positions.get(SYMBOL)
        if holding is not None:
            if self.pos is None:
                # Position first appears: the fill bar is this bar.
                if self.cash_before is not None and self.pending_stop is not None:
                    # cash_before is raw quote units; a must be USD, as in
                    # the backtest's fill_buy (cash there is USD float).
                    a = (self.cash_before / 10 ** self.quote_scale) * (1 - FEE_BPS / 10000.0)
                    fill_px = o * (1 + SLIP_BPS / 10000.0) * (1 + a / o_depth_usd)
                else:
                    fill_px = o  # fallback: the fill bar's open
                self.pos = {"entry_px": fill_px, "entry_bar": state.seq,
                            "highest": h, "stop_px": self.pending_stop}
                self.cash_before = None
                self.pending_stop = None
            # --- exit check (also on the fill bar, as in the backtest) ---
            self.pos["highest"] = max(self.pos["highest"], h)
            entry_px = self.pos["entry_px"]
            stop_px = self.pos["stop_px"]
            if c >= entry_px * 1.03:
                reason = "tp"
            elif stop_px is not None and c <= stop_px:
                reason = "sl"
            elif (self.pos["highest"] >= entry_px * 1.02
                    and c <= self.pos["highest"] - 0.005 * entry_px):
                reason = "trail"
            elif state.seq - self.pos["entry_bar"] >= 240:
                reason = "time"
            else:
                return []
            sys.stderr.write(
                "exit %s seq=%d entry=%.2f close=%.2f\n"
                % (reason, state.seq, entry_px, c))
            qty = holding.qty
            # Keep self.pos until the position actually disappears, so a
            # rejected sell is re-evaluated instead of re-initialised.
            return [{"side": "sell", "symbol": SYMBOL, "qty": qty,
                     "max_slippage_bps": 100}]

        # --- flat ---
        if self.pos is not None:
            self.pos = None  # position closed (sell filled)
        if state.warming_up:
            return []
        if self.m15 is None:
            return []
        m15 = self.m15

        stop = None
        # 1. I2c entry (first).
        if (self.sma90 is not None
                and c < self.sma90 * (1 - 0.0075)
                and m15["rsi2"] is not None and m15["rsi2"] < 20):
            stop = c * 0.99
        else:
            # 2. #3c-flat entry: only while the market is flat.
            atr = self.ind.atr.value
            if atr is not None and atr / c < 0.0020:
                n = 0
                bb = m15["bb"]
                r2 = m15["rsi2"]
                if bb is not None and c < bb[2]:
                    n += 1
                if r2 is not None and r2 < 10:
                    n += 1
                if m15["ema20"] is not None and atr:
                    if (c - m15["ema20"]) / atr < -2.0:
                        n += 1
                if (m15["ema20"] is not None and self.trend
                        and len(self.closes_1m) >= 2
                        and self.closes_1m[-2] <= m15["ema20"]
                        and self.closes_1m[-1] > m15["ema20"]):
                    n += 1
                if n >= 2:
                    stop = min(c * 0.995, c - 1.5 * atr)
        if stop is None:
            return []
        self.cash_before = state.cash
        self.pending_stop = stop
        sys.stderr.write(
            "entry seq=%d close=%.2f sma90=%s rsi2=%s stop=%.2f\n"
            % (state.seq, c,
               ("%.2f" % self.sma90) if self.sma90 is not None else "None",
               ("%.2f" % m15["rsi2"]) if m15["rsi2"] is not None else "None",
               stop))
        return [{"side": "buy", "symbol": SYMBOL,
                 "notional_quote_units": state.cash,
                 "max_slippage_bps": 100}]


def main():
    bot = ComboFlatBot()
    makefun_bars.run(bot.on_batch, on_hello=bot.on_hello)


if __name__ == "__main__":
    main()
