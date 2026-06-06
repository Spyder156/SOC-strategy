"""Cross-asset falsification test: does SOC's conditional gap beat plain lead-lag?

The deflationary null we must beat: "stock j's recent return predicts stock i's next
move" — lead-lag statistical arbitrage, no x_c, no hazard, no SOC. The SOC claim is that
the cross-asset effect is STATE-DEPENDENT: j's move matters MORE when i is already loaded
(small gap). That is an INTERACTION term. If it adds out-of-sample skill over the linear
lead-lag, SOC is real; if not, it's decoration on 2005 stat-arb.

No look-ahead: every feature at time t uses info up to t; the volatility scaler is LAGGED.
Train on the first 60% of time, evaluate on the last 40%.

Run: ./.venv/bin/python scripts/crossasset_falsification.py
"""

import numpy as np
import pandas as pd

from soc.data.clean import clean_quotes
from soc.data.replay_feed import STORE


def load_1s(sym):
    df = clean_quotes(pd.read_parquet(STORE / f"{sym}_quotes.parquet"))
    s = pd.Series(df["mid"].to_numpy(), index=pd.to_datetime(df["ts"].to_numpy(), unit="s"))
    return s.resample("1s").last().ffill()


def zfit(X, mean, std):
    return (X - mean) / np.where(std > 0, std, 1.0)


def fit_eval(feat_cols, Xtr, ytr, Xte, yte):
    """Least-squares linear model; return OOS correlation, sign-accuracy, gross Sharpe."""
    A = np.column_stack([Xtr[c] for c in feat_cols] + [np.ones(len(ytr))])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    Ate = np.column_stack([Xte[c] for c in feat_cols] + [np.ones(len(yte))])
    pred = Ate @ coef
    corr = np.corrcoef(pred, yte)[0, 1]
    acc = (np.sign(pred) == np.sign(yte)).mean()
    pnl = np.sign(pred) * yte               # gross: take sign each step, no costs
    sharpe = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(len(pnl))   # window-Sharpe (gross)
    return corr, acc, sharpe


def main():
    A, M = load_1s("AAPL"), load_1s("MSFT")
    df = pd.DataFrame({"A": A, "M": M}).dropna()
    df["rA"] = np.log(df["A"]).diff()
    df["rM"] = np.log(df["M"]).diff()
    df = df.dropna()
    print(f"aligned 1s bars: {len(df):,}   window overlap ok")

    # is there any lead-lag at all? cross-corr of MSFT_now with AAPL_{now+k}
    print("lead-lag check  corr(rM[t], rA[t+k]):")
    for k in (0, 1, 2, 3):
        c = np.corrcoef(df["rM"][:-k or None], df["rA"].shift(-k).dropna()[:len(df["rM"][:-k or None])])[0, 1] if k else np.corrcoef(df["rM"], df["rA"])[0, 1]
        print(f"   k={k}: {c:+.4f}")

    # features known at t  (predict target = rA[t+1])
    r = df.copy()
    r["rM_lag1"] = r["rM"].shift(1)
    # LAGGED loading state: price vs slow baseline, scaled by LAGGED vol (no look-ahead)
    base = r["A"].ewm(halflife=120).mean().shift(1)
    vol = r["rA"].abs().ewm(halflife=120).mean().shift(1)
    r["load"] = ((r["A"].shift(1) - base) / vol.replace(0, np.nan))
    r["rM_x_load"] = r["rM"] * r["load"]
    r["target"] = r["rA"].shift(-1)
    r = r.dropna()

    cols = ["rA", "rM", "rM_lag1", "load", "rM_x_load"]
    # z-score using TRAIN stats only
    cut = int(len(r) * 0.6)
    tr, te = r.iloc[:cut], r.iloc[cut:]
    mean, std = tr[cols].mean(), tr[cols].std()
    Xtr = {c: zfit(tr[c].to_numpy(), mean[c], std[c]) for c in cols}
    Xte = {c: zfit(te[c].to_numpy(), mean[c], std[c]) for c in cols}
    ytr, yte = tr["target"].to_numpy(), te["target"].to_numpy()

    models = {
        "bounce-null   (rA)":              ["rA"],
        "lead-lag NULL (rA,rM,rM_lag1)":   ["rA", "rM", "rM_lag1"],
        "SOC conditional (+load,+rM*load)":["rA", "rM", "rM_lag1", "load", "rM_x_load"],
    }
    print(f"\nOUT-OF-SAMPLE (train {cut:,} / test {len(te):,} bars):")
    print(f"{'model':<34} {'corr':>8} {'sign-acc':>9} {'gross-Sharpe':>13}")
    for name, fc in models.items():
        corr, acc, sh = fit_eval(fc, Xtr, ytr, Xte, yte)
        print(f"{name:<34} {corr:>+8.4f} {acc:>9.3f} {sh:>13.2f}")
    print("\nverdict: SOC must beat the lead-lag NULL on OOS corr/Sharpe to be more than stat-arb.")


if __name__ == "__main__":
    main()
