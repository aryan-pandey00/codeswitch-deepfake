"""Summarise the real-phone acceptance test.

Notebook 15 scored 45 clips recorded on other people's own phones and wrote one row per clip.
This aggregates those rows and computes nothing new.

One speaker is outside the population measured here, and notebook 15 excluded them from the
headline rate. That exclusion is carried as an `in_scope` column rather than applied silently,
so the arithmetic stays checkable against all 45 rows.

Nothing was rejected, and zero of forty-four is not zero percent, so every rate carries the
Wilson interval from notebook 16. Rejections cluster by speaker, so the speaker-level bound is
computed too and it is the wider one."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "realvoice_scores.csv"
OUT_DIR = ROOT / "paper" / "results"

HEADS = ("baseline", "fixed")
# Repeated from notebook 15, not imported, so the exclusion is visible where the rate is written.
OUT_OF_SCOPE = {"english-foreign"}


def wilson(k, n, z=1.96):
    """95% interval for a rate. Identical to notebook 16, because several cells have zero errors."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def stratum(df, axis, level):
    """One row of the table. Every clip is a real human, so a rejection is always an error:
    no miss rate and no EER here, only false rejection."""
    row = dict(axis=axis, level=level, n_clips=len(df), n_speakers=df.speaker.nunique())
    for head in HEADS:
        rejected = df[f"rejected_{head}"]
        k = int(rejected.sum())
        lo, hi = wilson(k, len(df))
        row[f"rejected_{head}"] = k
        row[f"frr_{head}"] = 100 * k / len(df) if len(df) else float("nan")
        row[f"frr_{head}_lo"] = lo
        row[f"frr_{head}_hi"] = hi

        # A speaker counts as rejected if any of their clips was. Errors cluster by session,
        # so this is the bound that survives contact with a deployment.
        by_spk = df.groupby("speaker")[f"rejected_{head}"].any()
        k_spk = int(by_spk.sum())
        lo_s, hi_s = wilson(k_spk, len(by_spk))
        row[f"spk_rejected_{head}"] = k_spk
        row[f"spk_frr_{head}"] = 100 * k_spk / len(by_spk) if len(by_spk) else float("nan")
        row[f"spk_frr_{head}_lo"] = lo_s
        row[f"spk_frr_{head}_hi"] = hi_s

    # Margin, not just the count. A detector that accepts everything by sitting near the
    # boundary would pass the count and fail here. Negative is genuine, so the maximum is the
    # closest any clip came to being rejected.
    row["z_fixed_mean"] = df.z_fixed.mean()
    row["z_fixed_median"] = df.z_fixed.median()
    row["z_fixed_worst"] = df.z_fixed.max()
    row["band_ratio_median"] = df.band_ratio.median()
    row["secs_min"] = df.secs.min()
    row["secs_max"] = df.secs.max()
    return row


def main():
    df = pd.read_csv(IN_CSV)
    df.insert(1, "in_scope", ~df.language.isin(OUT_OF_SCOPE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "realphone_scores.csv", index=False)

    scope = df[df.in_scope]
    rows = [stratum(scope, "overall", "all in-scope clips")]
    for axis in ("source", "language", "style"):
        for level, g in scope.groupby(axis):
            rows.append(stratum(g, axis, level))
    # Reported, never averaged in: a speaker from outside the measured population is evidence
    # about generalisation, not evidence about the rate this project claims.
    held = df[~df.in_scope]
    if len(held):
        rows.append(stratum(held, "held out", "outside the measured population"))

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "realphone_acceptance.csv", index=False)

    o = table.iloc[0]
    print(f"{len(df)} clips, {df.speaker.nunique()} speakers; "
          f"{int(o.n_clips)} in scope from {int(o.n_speakers)} speakers, "
          f"{len(held)} held out\n")
    for head in HEADS:
        print(f"{head}: rejected {int(o[f'rejected_{head}'])}/{int(o.n_clips)} clips "
              f"= {o[f'frr_{head}']:.1f}%, 95% CI [{o[f'frr_{head}_lo']:.1f}, {o[f'frr_{head}_hi']:.1f}]")
        print(f"{' ' * len(head)}  by speaker {int(o[f'spk_rejected_{head}'])}/{int(o.n_speakers)} "
              f"= {o[f'spk_frr_{head}']:.1f}%, upper bound {o[f'spk_frr_{head}_hi']:.1f}%")
    print(f"\nfixed head margin: mean {o.z_fixed_mean:.2f}, median {o.z_fixed_median:.2f}, "
          f"closest {o.z_fixed_worst:.2f}")
    print("\nenergy above 4 kHz before the bottleneck, by source")
    for _, r in table[table.axis == "source"].iterrows():
        print(f"  {r.level}, n={int(r.n_clips)}: {r.band_ratio_median:.4f}")

    wrong = df[df.rejected_baseline | df.rejected_fixed]
    if len(wrong):
        print("\nclips rejected by either head")
        for _, r in wrong.iterrows():
            print(f"  {r.file} baseline z={r.z_baseline:+.3f} fixed z={r.z_fixed:+.3f}")

    print(f"\nwrote {OUT_DIR / 'realphone_scores.csv'}")
    print(f"wrote {OUT_DIR / 'realphone_acceptance.csv'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
