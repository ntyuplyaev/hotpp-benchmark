"""Dump recent MLflow runs from a SQLite-backed tracking DB.

Usage:
    python3 scripts/analysis/mlflow_dump.py <db_path> [--last N] [--name-like PATTERN]
"""
import argparse
import sqlite3
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", type=Path)
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--name-like", default=None)
    ap.add_argument("--metric-keys", default=None, help="Comma-separated metric keys to pull; defaults to a curated list")
    return ap.parse_args()


DEFAULT_METRICS = [
    "val/T-mAP",
    "val/T-mAP-weighted",
    "val/horizon-max-f-score",
    "val/next-item-accuracy",
    "val/mean-predicted-length",
    "val/mean-target-length",
    "val/optimal-transport-distance",
    "val/sequence-labels-entropy",
    "val/target-sequence-labels-entropy",
    "val/loss",
    "val/loss_diffusion",
    "val/loss_decoder__presence",
    "val/loss_decoder_labels",
    "val/loss_decoder_timestamps",
    "val/loss_decoder_next_item_labels",
    "val/loss_decoder_next_item_timestamps",
    "val/decoder_prediction_match_rate",
    "val/decoder_target_match_rate",
    "val/perf_presence_ratio",
    "test/T-mAP",
    "test/T-mAP-weighted",
    "test/horizon-max-f-score",
    "test/next-item-accuracy",
    "test/optimal-transport-distance",
    "test/sequence-labels-entropy",
    "test/mean-predicted-length",
]


def main():
    args = parse_args()
    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    metric_keys = args.metric_keys.split(",") if args.metric_keys else DEFAULT_METRICS

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cur = conn.cursor()

    # List runs ordered by end/start time desc.
    run_sql = """
        SELECT r.run_uuid, r.name, r.status, r.start_time, r.end_time, e.name as experiment_name
        FROM runs r
        LEFT JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE 1=1
    """
    params = []
    if args.name_like:
        run_sql += " AND r.name LIKE ?"
        params.append(args.name_like)
    run_sql += " ORDER BY COALESCE(r.end_time, r.start_time) DESC LIMIT ?"
    params.append(args.last)

    runs = cur.execute(run_sql, params).fetchall()
    if not runs:
        print("No runs found.")
        return

    for run_uuid, name, status, start_ms, end_ms, exp_name in runs:
        start = start_ms / 1000 if start_ms else None
        end = end_ms / 1000 if end_ms else None
        import datetime
        start_s = datetime.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M") if start else "-"
        end_s = datetime.datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M") if end else "-"
        duration_min = ((end_ms - start_ms) / 1000 / 60) if (start_ms and end_ms) else None
        print(f"\n=== {name or run_uuid[:8]} [{status}] exp={exp_name}")
        print(f"    start={start_s}  end={end_s}  duration={duration_min:.1f}m" if duration_min else f"    start={start_s}  end={end_s}")
        print(f"    run_uuid={run_uuid}")

        # Pull metrics for this run — last value for each metric_key.
        placeholders = ",".join("?" * len(metric_keys))
        metric_sql = f"""
            SELECT m.key, m.value, m.step, m.timestamp
            FROM metrics m
            WHERE m.run_uuid = ?
              AND m.key IN ({placeholders})
              AND (m.step, m.timestamp) = (
                  SELECT MAX(m2.step), MAX(m2.timestamp)
                  FROM metrics m2
                  WHERE m2.run_uuid = m.run_uuid AND m2.key = m.key
              )
            ORDER BY m.key
        """
        rows = cur.execute(metric_sql, [run_uuid, *metric_keys]).fetchall()
        # fallback: if composite (step,timestamp) max is too restrictive, use separate fallback
        if not rows:
            for k in metric_keys:
                row = cur.execute(
                    "SELECT key, value, step, timestamp FROM metrics WHERE run_uuid=? AND key=? ORDER BY step DESC, timestamp DESC LIMIT 1",
                    (run_uuid, k),
                ).fetchone()
                if row:
                    rows.append(row)

        if not rows:
            print("    (no metrics)")
            continue
        for key, value, step, _ts in rows:
            print(f"    {key:40s} = {value:.6g}   (step {step})")


if __name__ == "__main__":
    main()
