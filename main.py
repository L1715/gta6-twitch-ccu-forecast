"""Forecast launch-day Twitch viewers hour by hour, from comparable launches.

Method (everything comes from the case file; the sandbox has no network):

  1. Each comparable is a past big launch with hourly Twitch category
     averages spanning its release day.  For every comparable we form the
     release-day shape as log ratios to its own hour-1 count,
     log(v_h / v_1) for h = 2..24.  Ratios are scale-free, so a launch
     ten times bigger still informs the shape.
  2. Comparables are weighted by how much they resemble the target
     launch *before* the checkpoint, using only information available
     at hour 1: how hour 1 compares with the previous six hours and
     with the previous day's peak (a Gaussian kernel on those two log
     features).  Launches whose unlock had already rolled through
     Asia/Europe before UTC midnight look different from launches
     with a single global unlock later in the day, and the kernel
     separates those regimes.  Tier weights from each comparable's
     unlock note downweight early-access-contaminated and weak
     comparables, and console-only launches get a bonus weight: the
     target is a console-only rolling-midnight launch.
  3. Each hour's forecast blends two estimates of the weighted
     distribution of log ratios: empirical weighted quantiles (which
     keep the observed skew), computed on values winsorized to the
     weighted median +/- 3 robust standard deviations, and normal
     quantiles centred on the weighted median with a MAD scale
     (which stabilise the tails when the effective pool is small).
     The blend is 50/50 in log space.  A floor keeps the spread
     honest, and the bandwidth widens automatically if the kernel
     weights collapse onto too few comparables.
  4. If no comparable is usable, fall back to persistence (every
     hour at hour 1's count, spread widening like a random walk),
     so the program always writes a valid forecast.

The forecast is deterministic: fixed iteration order, no randomness,
no wall-clock or environment dependence beyond the input/output
directories.  Standard library only.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Kernel bandwidth (log units) on the two pre-checkpoint features.
BANDWIDTH = 0.75
# Minimum effective number of comparables; widen the bandwidth until met.
MIN_EFFECTIVE_N = 5.0
MAX_BANDWIDTH = 3.0
# Console-only launches resemble the target's unlock pattern most.
CONSOLE_BONUS = 2.5
# Floor on the robust log-scale (per hour), so spreads stay honest.
SD_FLOOR = 0.25
# Winsorize pooled log ratios to median +/- this many robust sigmas.
WINSOR_SIGMAS = 3.0
# Blend weight on the empirical quantiles (rest on the normal quantiles).
EMPIRICAL_WEIGHT = 0.5
# Persistence fallback spread, log units per square-root hour.
FALLBACK_SIGMA = 0.15


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp (with a trailing Z) to epoch seconds."""
    # datetime.fromisoformat needs '+00:00' on older versions; normalise.
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def tier_weight(unlock_text):
    """Weight from the comparable's unlock note tier, or None to exclude."""
    text = (unlock_text or "").lower()
    if "target-history" in text:
        return None
    if "early-access-contaminated" in text:
        return 0.3
    if "tier: weak" in text:
        return 0.1
    return 1.0


def is_console_only(platforms):
    text = platforms or ""
    return "PC" not in text and "Stadia" not in text


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def pre_checkpoint_features(hourly_map, day_start):
    """Two log features from the 24 hours before the release day.

    hourly_map maps epoch hour-start -> viewers (floored at 1).
    Returns (log(h1 / median of prior 6h), log(h1 / max of prior 24h))
    or None when the pre-day history is too sparse to trust.
    """
    hour1 = hourly_map.get(day_start)
    if hour1 is None:
        return None
    prior6 = [hourly_map[day_start - 3600 * k] for k in range(1, 7)
              if day_start - 3600 * k in hourly_map]
    prior24 = [hourly_map[day_start - 3600 * k] for k in range(1, 25)
               if day_start - 3600 * k in hourly_map]
    if len(prior6) < 3 or len(prior24) < 6:
        return None
    return (math.log(hour1 / median(prior6)),
            math.log(hour1 / max(prior24)))


def build_comparable(entry):
    """Extract a comparable's release-day curve and features, or None."""
    weight = tier_weight(entry.get("unlock"))
    if weight is None:
        return None
    try:
        day_start = parse_time(entry["day_start"])
    except (KeyError, ValueError, TypeError):
        return None
    hourly_map = {}
    for row in entry.get("hourly") or []:
        try:
            at = parse_time(row["from"])
            viewers = max(1.0, float(row["viewers"]))
        except (KeyError, ValueError, TypeError):
            continue
        hourly_map[at] = viewers
    day = [hourly_map.get(day_start + 3600 * h) for h in range(24)]
    if day[0] is None or sum(1 for v in day if v is not None) < 20:
        return None
    if is_console_only(entry.get("platforms")):
        weight *= CONSOLE_BONUS
    return {
        "weight": weight,
        "day": day,
        "features": pre_checkpoint_features(hourly_map, day_start),
    }


def weighted_quantile(pairs, level):
    """Interpolated weighted quantile of [(value, weight), ...]."""
    if not pairs:
        return 0.0
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    target = level * total
    acc = 0.0
    prev_value = ordered[0][0]
    prev_acc = 0.0
    for value, w in ordered:
        acc += w
        if acc >= target:
            if acc == prev_acc:
                return value
            frac = (target - prev_acc) / (acc - prev_acc)
            return prev_value + frac * (value - prev_value)
        prev_value = value
        prev_acc = acc
    return ordered[-1][0]


def effective_n(weights):
    total = sum(weights)
    if total <= 0:
        return 0.0
    return total * total / sum(w * w for w in weights)


def kernel_weights(comparables, target_features):
    """Base weights times the Gaussian kernel; widen until effective N fits."""
    bandwidth = BANDWIDTH
    while True:
        weights = []
        for comp in comparables:
            w = comp["weight"]
            feats = comp["features"]
            if target_features is not None and feats is not None:
                d0 = target_features[0] - feats[0]
                d1 = target_features[1] - feats[1]
                w *= math.exp(-(d0 * d0 + d1 * d1) / (2.0 * bandwidth ** 2))
            weights.append(w)
        if effective_n(weights) >= MIN_EFFECTIVE_N or bandwidth >= MAX_BANDWIDTH:
            return weights
        bandwidth *= 1.5


def hour_log_quantiles(samples):
    """Blend empirical and normal quantiles of weighted log-ratio samples.

    samples: [(log ratio, weight), ...] for one hour.  Returns the five
    log-space quantiles at LEVELS.
    """
    centre = weighted_quantile(samples, 0.5)
    deviations = [(abs(v - centre), w) for v, w in samples]
    mad = weighted_quantile(deviations, 0.5)
    sd = max(SD_FLOOR, 1.4826 * mad)
    lo = centre - WINSOR_SIGMAS * sd
    hi = centre + WINSOR_SIGMAS * sd
    clipped = [(min(hi, max(lo, v)), w) for v, w in samples]
    empirical = [weighted_quantile(clipped, tau) for tau in LEVELS]
    normal = [centre + z * sd for z in Z]
    blended = [EMPIRICAL_WEIGHT * e + (1.0 - EMPIRICAL_WEIGHT) * n
               for e, n in zip(empirical, normal)]
    # Blending two non-decreasing sequences stays non-decreasing, but
    # enforce it explicitly so rounding can never break the contract.
    for i in range(1, len(blended)):
        blended[i] = max(blended[i], blended[i - 1])
    return blended


def round2(x):
    """Round to 2 decimals, half up, returning an int for whole numbers."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def clamp_quantiles(values):
    """Clamp to (0, 1e9], round, and enforce non-decreasing order."""
    out = []
    prev = 0.0
    for v in values:
        q = min(1e9, max(0.01, v))
        q = round2(q)
        q = max(q, prev)
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    """Forecast one case.  Always returns a valid 23-row forecast."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), known[0] if known else None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    hours = [row["hour"] for row in case["forecast"]]

    comparables = []
    for entry in case.get("comparables") or []:
        comp = build_comparable(entry)
        if comp is not None:
            comparables.append(comp)

    # Target features from the case's own hourly history (hour 1 included
    # via the known print, so a sparse history still anchors correctly).
    day_start = parse_time(case["day_start"])
    history_map = {}
    for row in case.get("history_hourly") or []:
        try:
            history_map[parse_time(row["from"])] = max(1.0, float(row["viewers"]))
        except (KeyError, ValueError, TypeError):
            continue
    history_map[day_start] = p1
    target_features = pre_checkpoint_features(history_map, day_start)

    weights = kernel_weights(comparables, target_features) if comparables else []

    rows = []
    for h in hours:
        offset = h - 1  # index into a comparable's 24-hour day array
        samples = []
        for comp, w in zip(comparables, weights):
            day = comp["day"]
            if 0 <= offset < len(day) and day[offset] is not None and day[0]:
                samples.append((math.log(day[offset] / day[0]), w))
        if samples:
            log_q = hour_log_quantiles(samples)
            quantiles = [p1 * math.exp(q) for q in log_q]
        else:
            # Persistence fallback: flat median, widening spread.
            quantiles = [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(offset))
                         for z in Z]
        rows.append({"hour": h, "q": clamp_quantiles(quantiles)})
    return {"hours": rows}


def persistence_forecast(case):
    """Last-resort valid forecast: hour 1's print, widening spread."""
    known = case.get("known") or []
    p1 = 1.0
    for k in known:
        if k.get("hour") == 1:
            p1 = max(1.0, float(k["viewers"]))
            break
    rows = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        quantiles = [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))
                     for z in Z]
        rows.append({"hour": h, "q": clamp_quantiles(quantiles)})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = persistence_forecast(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
