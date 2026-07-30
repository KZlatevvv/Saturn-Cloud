#!/usr/bin/env python3
"""
Saturn Cloud "Brain" — runs as a scheduled GitHub Action in the Saturn-Cloud
repo (see .github/workflows/brain.yml). Every run:

  1. Reads the CURRENT data.txt from this checkout and decrypts it with the
     same AES-256-GCM key baked into the plugin (Obf.java) — this is the
     baseline. It is NEVER read from a plaintext file in this repo; the only
     place plaintext thresholds may ever exist is a decrypted value held in
     memory for the few seconds this script runs, on GitHub's own runner.
     (master_signatures.json — the plaintext local template on your machine —
     must NEVER be committed here. This repo is public.)
  2. Pulls recent anomaly logs from the vendor telemetry backend.
  3. Nudges the baseline thresholds based on those logs (see analyze() —
     this is a v1 placeholder heuristic, not a validated tuning model; read
     the comments on that function before trusting it unattended).
  4. Re-encrypts the result and overwrites data.txt, IF it actually changed.

The actual git commit/push happens in the workflow (plain git CLI), not
here — this script only ever writes the plaintext-free data.txt file to
disk and exits 0 (changed) or 1 (nothing to do), so the workflow can decide
whether a commit is even needed.
"""

import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DATA_FILE = "data.txt"

# ---------------------------------------------------------------- config
# All of these come from GitHub Actions secrets/env — see brain.yml.
# Nothing here is a real value; if any of these is missing the run fails
# loudly instead of silently publishing garbage.
OBF_KEY_HEX = os.environ["SATURN_OBF_KEY"]              # same 32-byte key as Obf.java's KEY
CLOUD_URL = os.environ["CLOUD_URL"]                      # backend REST base URL
CLOUD_KEY = os.environ["CLOUD_KEY"]                      # privileged (read) key — NOT the plugin's embedded client-side key, see README note

LOOKBACK_HOURS = float(os.environ.get("LOOKBACK_HOURS", "24"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "20"))
MAX_ADJUST_PCT = float(os.environ.get("MAX_ADJUST_PCT", "0.10"))
TIMESTAMP_COLUMN = os.environ.get("TIMESTAMP_COLUMN", "created_at")
LOGS_TABLE = os.environ.get("LOGS_TABLE", "anticheat_logs")

KEY = bytes.fromhex(OBF_KEY_HEX)
if len(KEY) != 32:
    sys.exit(f"SATURN_OBF_KEY must be 32 bytes (64 hex chars), got {len(KEY)} bytes")


# ---------------------------------------------------------------- crypto
# Must match com.saturnanticheat.util.Obf byte-for-byte: AES/GCM/NoPadding,
# 12-byte random IV, 128-bit tag, base64(iv || ciphertext || tag). Verified
# against the real Obf.s() this key belongs to before this script shipped.

def obf_decrypt(blob_b64: str) -> str:
    raw = base64.b64decode(blob_b64)
    iv, ct_and_tag = raw[:12], raw[12:]
    plaintext = AESGCM(KEY).decrypt(iv, ct_and_tag, None)
    return plaintext.decode("utf-8")


def obf_encrypt(plaintext: str) -> str:
    iv = os.urandom(12)
    ct_and_tag = AESGCM(KEY).encrypt(iv, plaintext.encode("utf-8"), None)
    return base64.b64encode(iv + ct_and_tag).decode("ascii")


# ---------------------------------------------------------------- baseline

def load_baseline() -> dict:
    """Current published thresholds, decrypted from data.txt in this checkout.
    Empty dict on first-ever run (file missing/empty) or on any decrypt/parse
    failure — a broken baseline must never crash the run or get overwritten
    with guesses; it just means "nothing known yet, adjust nothing this run"."""
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return {}
    try:
        return json.loads(obf_decrypt(raw))
    except Exception as ex:
        print(f"WARNING: could not decrypt/parse existing {DATA_FILE} ({ex}) "
              f"— treating baseline as empty, adjusting nothing this run.")
        return {}


# ---------------------------------------------------------------- backend

def fetch_anomaly_logs() -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).isoformat()
    url = f"{CLOUD_URL}/{LOGS_TABLE}"
    params = {
        "select": f"check_name,player_uuid,violation_level,{TIMESTAMP_COLUMN}",
        TIMESTAMP_COLUMN: f"gte.{since}",
        "limit": "50000",
    }
    headers = {
        "apikey": CLOUD_KEY,
        "Authorization": f"Bearer {CLOUD_KEY}",
    }
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------- analysis
#
# THIS IS A STARTING SKELETON, NOT A VALIDATED TUNING MODEL. Ship it, watch
# it for a while, read the diffs it produces (every change is a normal git
# commit in this repo's history — that history IS the audit trail and the
# undo button; if a nudge looks wrong, `git revert` it).
#
# Only checks that ALREADY have an entry in `baseline` are ever touched —
# the brain never invents a new check key or a new field name it hasn't
# already seen published, it only nudges values it already knows the shape
# of. That's a deliberate ceiling on what an unattended run can do.
#
# Heuristic: for each check present in the baseline, look at how many
# DISTINCT players triggered it in the lookback window vs. total flags.
#   - few distinct players, many flags each  -> looks like real repeat
#     offenders, leave the check alone.
#   - many distinct players, ~1 flag each    -> broad, shallow flagging is
#     more consistent with false positives than a coordinated cheat wave;
#     nudge every numeric field in that check up by MAX_ADJUST_PCT (looser).
# Skipped entirely if fewer than MIN_SAMPLES flags in the window — not
# enough signal to move anything on that little data.
#
# This says nothing about WHICH direction "looser" means per field (a
# margin vs. a millisecond window vs. a sample count don't all want the
# same treatment) — a real v2 needs per-field semantics, which isn't
# something this script can infer on its own. Refine before fully trusting it.

def analyze(baseline: dict, logs: list[dict]) -> dict:
    by_check: dict[str, list[dict]] = {}
    for row in logs:
        by_check.setdefault(row["check_name"], []).append(row)

    updated = json.loads(json.dumps(baseline))  # deep copy
    changed_checks = []

    for check_key, fields in baseline.items():
        rows = by_check.get(check_key, [])
        if len(rows) < MIN_SAMPLES:
            continue
        distinct_players = len({r["player_uuid"] for r in rows})
        flags_per_player = len(rows) / max(distinct_players, 1)

        # broad + shallow: looks false-positive-shaped -> loosen a bit
        if distinct_players >= 5 and flags_per_player <= 1.5:
            for field_name, value in fields.items():
                if not isinstance(value, (int, float)):
                    continue
                nudged = value * (1 + MAX_ADJUST_PCT)
                updated[check_key][field_name] = (
                    round(nudged) if isinstance(value, int) else round(nudged, 6)
                )
            changed_checks.append((check_key, len(rows), distinct_players))

    if changed_checks:
        print("Adjusted checks this run:")
        for key, n, players in changed_checks:
            print(f"  {key}: {n} flags across {players} distinct players -> loosened {MAX_ADJUST_PCT:.0%}")
    else:
        print("No check crossed the adjustment threshold this run — baseline unchanged.")

    return updated


# ---------------------------------------------------------------- main

def main() -> int:
    """Always returns 0 on a successful run, whether or not anything changed
    — "no adjustment needed this cycle" is the normal case, not a failure.
    Whether data.txt actually differs (and is therefore worth committing) is
    decided by the workflow via `git diff`, not by this exit code."""
    baseline = load_baseline()
    print(f"Loaded baseline: {len(baseline)} check(s) known.")

    try:
        logs = fetch_anomaly_logs()
        print(f"Fetched {len(logs)} anomaly log row(s) from the last {LOOKBACK_HOURS}h.")
    except Exception as ex:
        print(f"WARNING: backend fetch failed ({ex}) — leaving baseline untouched.")
        logs = []

    updated = analyze(baseline, logs) if baseline else baseline

    new_json = json.dumps(updated, separators=(",", ":"), sort_keys=True)
    old_json = json.dumps(baseline, separators=(",", ":"), sort_keys=True)

    if new_json == old_json:
        print("Nothing changed — not rewriting data.txt.")
        return 0

    # a fresh random IV means the ciphertext differs even when the
    # underlying JSON doesn't — only re-encrypt (and let the workflow diff
    # data.txt) when the DECRYPTED content actually changed, checked above
    encrypted = obf_encrypt(new_json)
    with open(DATA_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(encrypted + "\n")
    print(f"Wrote updated {DATA_FILE} ({len(updated)} check(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
