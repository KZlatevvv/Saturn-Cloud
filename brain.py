#!/usr/bin/env python3
"""
Saturn Cloud "Brain" -- runs as a scheduled GitHub Action in the Saturn-Cloud
repo (see .github/workflows/brain.yml). Every run:

  1. Reads the CURRENT data.txt from this checkout and decrypts it with the
     same AES-256-GCM key baked into the plugin (Obf.java) -- this is the
     baseline. It is NEVER read from a plaintext file in this repo; the only
     place plaintext thresholds may ever exist is a decrypted value held in
     memory for the few seconds this script runs, on GitHub's own runner.
     (master_signatures.json -- the plaintext local template on your machine
     -- must NEVER be committed here. This repo is public.)
  2. Also reads brain_state.json (plaintext, not encrypted -- it holds no
     detection thresholds, just per-check adjustment counters) to enforce
     the sanity rails in analyze() below.
  3. Pulls anomaly logs from the vendor telemetry backend, spanning both the
     current evaluation window AND enough history behind it for each check
     to have its own baseline.
  4. Nudges the baseline thresholds based on those logs -- each check is
     compared against ITS OWN historical flag-rate, not one flat threshold
     shared by all 200+ checks (see analyze() for the full reasoning and its
     honest limits; this is still not a trained model). If AI_API_KEY,
     AI_API_URL and AI_MODEL are all set, every check the statistics decide
     to loosen also gets a second opinion from an LLM (see consult_ai()) --
     the AI can only talk that
     adjustment DOWN, never up; it is structurally incapable of making the
     brain more aggressive than plain statistics already would have been.
     All of this is subject to the sanity rails: a hard per-check
     adjustment-count cap, a floor against ever nudging a value to
     zero/negative, and a structural check that the set of checks never
     silently changes shape.
  5. Re-encrypts the result and overwrites data.txt, IF it actually changed
     -- immediately re-reads and decrypts what was just written to confirm
     it round-trips before trusting the run was clean.
  6. Writes brain_run_summary.txt (plain text, human-readable) with exactly
     what changed and why, for the workflow to use as the git commit
     message body -- `git log -- data.txt` becomes a real audit trail
     instead of a wall of identical timestamped commit titles.
  7. If DISCORD_WEBHOOK_URL is set, posts the same summary there. Entirely
     optional and non-fatal: a broken/missing webhook never blocks the run
     or the data.txt write.

The actual git commit/push happens in the workflow (plain git CLI), not
here -- this script only ever writes plaintext-free data.txt plus the
plaintext brain_state.json and brain_run_summary.txt to disk and exits 0
(changed) or 1 (nothing to do), so the workflow can decide whether a commit
is even needed.
"""

import base64
import json
import os
import statistics
import time
import sys
from datetime import datetime, timedelta, timezone

import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DATA_FILE = "data.txt"
STATE_FILE = "brain_state.json"
SUMMARY_FILE = "brain_run_summary.txt"

# ---------------------------------------------------------------- config
# All of these come from GitHub Actions secrets/env -- see brain.yml.
# Nothing here is a real value; if any of these is missing the run fails
# loudly instead of silently publishing garbage.
OBF_KEY_HEX = os.environ["SATURN_OBF_KEY"]              # same 32-byte key as Obf.java's KEY
CLOUD_URL = os.environ["CLOUD_URL"]                      # backend REST base URL
CLOUD_KEY = os.environ["CLOUD_KEY"]                      # privileged (read) key -- NOT the plugin's embedded client-side key, see README note

# size of the "current" evaluation window, and of each historical window
# behind it -- still a rolling window ending now, not a calendar-day bucket
# (avoids partial-day bias on runs that land early in a UTC day)
LOOKBACK_HOURS = float(os.environ.get("LOOKBACK_HOURS", "24"))
# how many PAST windows (of LOOKBACK_HOURS each) to fetch as history for
# each check's own baseline -- 6 windows * 24h = 6 days of history behind
# the current one, 7 days fetched total
HISTORY_WINDOWS = int(os.environ.get("HISTORY_WINDOWS", "6"))
# a check needs at least this many populated historical windows before its
# own baseline is trusted over the flat fallback rule below
MIN_HISTORY_WINDOWS = int(os.environ.get("MIN_HISTORY_WINDOWS", "3"))
# flags needed in the CURRENT window before considering a check at all
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "20"))
# distinct players needed in the CURRENT window
MIN_DISTINCT_PLAYERS = int(os.environ.get("MIN_DISTINCT_PLAYERS", "5"))
# how many standard deviations BELOW a check's own historical flags/player
# ratio today's ratio must fall to count as anomalous-looking (broader,
# shallower flagging than that specific check normally produces)
ANOMALY_Z = float(os.environ.get("ANOMALY_Z", "1.25"))
MAX_ADJUST_PCT = float(os.environ.get("MAX_ADJUST_PCT", "0.10"))
TIMESTAMP_COLUMN = os.environ.get("TIMESTAMP_COLUMN", "created_at")
LOGS_TABLE = os.environ.get("LOGS_TABLE", "anticheat_logs")

# --------------------------------------------------------- sanity rails
# A check that keeps getting auto-loosened run after run, forever, without
# a human ever looking at it is exactly the failure mode an unattended
# script should never be allowed to reach on its own. Once a check hits
# this many TOTAL adjustments (tracked in brain_state.json, resets only
# when a human edits/clears that entry), the brain stops touching it and
# just prints a loud warning every run until someone reviews it by hand.
MAX_ADJUST_COUNT = int(os.environ.get("MAX_ADJUST_COUNT", "20"))
# Optional -- Discord webhook URL for run notifications. Empty/unset simply
# disables this feature; nothing else in the script depends on it.
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ------------------------------------------------------- AI second opinion
# Optional. Empty/unset AI_API_KEY disables this entirely -- the statistical
# decision above stands on its own either way, exactly as it did before this
# existed. When set, every check the STATISTICS already decided to loosen
# gets a second opinion from an LLM before the nudge is applied.
#
# The AI is structurally NOT allowed to make a decision more aggressive than
# the statistics already proposed -- it can only AGREE with the statistical
# nudge, or talk it DOWN to something smaller (including all the way to "do
# nothing"). This is enforced twice: once in the prompt (asked for), and
# again in analyze() with a hard min()/max() clamp on whatever number comes
# back, so a confused or manipulated model response can never do more than a
# clean statistics-only run already could. If the API call fails, times out,
# or returns something that doesn't parse as expected JSON, this silently
# falls back to the plain statistical decision -- an AI outage or a bad key
# can never block a run or crash it, it just means one fewer opinion.
#
# Works with any OpenAI-compatible /chat/completions provider -- which one
# is deliberately NOT hardcoded here or given a default, the same way the
# detection thresholds elsewhere in this project are never shipped in the
# clear in a public repo. All three of AI_API_KEY/AI_API_URL/AI_MODEL must
# be supplied via secrets (see brain.yml) for this feature to activate; if
# any are missing, the AI step is simply skipped -- same graceful no-op as
# every other failure mode here, never a hard requirement.
AI_API_KEY = os.environ.get("AI_API_KEY", "")
AI_API_URL = os.environ.get("AI_API_URL", "")
AI_MODEL = os.environ.get("AI_MODEL", "")
# defensive cap on how many AI calls a single run will ever make, regardless
# of how many checks look anomalous at once -- keeps a pathological run
# (e.g. many checks tripping simultaneously during a real incident) from
# turning into an unbounded number of API calls
MAX_AI_CONSULTATIONS = int(os.environ.get("MAX_AI_CONSULTATIONS", "20"))

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
    failure -- a broken baseline must never crash the run or get overwritten
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
              f"-- treating baseline as empty, adjusting nothing this run.")
        return {}


def load_state() -> dict:
    """Plaintext {checkId: {"adjust_count": N, "last_adjusted": iso}}. Never
    encrypted -- it carries no detection thresholds, only meta-counters, and
    keeping it plaintext means a human can just open and edit/reset it
    directly (e.g. delete one check's entry after reviewing it) with no
    tooling required. Same empty-on-any-failure philosophy as load_baseline."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as ex:
        print(f"WARNING: could not read {STATE_FILE} ({ex}) -- starting fresh "
              f"(adjustment-count caps reset to zero for every check this run).")
        return {}


# ---------------------------------------------------------------- backend

def fetch_anomaly_logs() -> list[dict]:
    """Fetches the CURRENT window plus HISTORY_WINDOWS worth of history behind
    it, in one query -- analyze() below splits this single list back into
    windows itself, so there's only ever one round trip to the backend."""
    total_hours = LOOKBACK_HOURS * (HISTORY_WINDOWS + 1)
    since = (datetime.now(timezone.utc) - timedelta(hours=total_hours)).isoformat()
    url = f"{CLOUD_URL}/{LOGS_TABLE}"
    params = {
        "select": f"check_name,player_uuid,violation_level,{TIMESTAMP_COLUMN}",
        TIMESTAMP_COLUMN: f"gte.{since}",
        "limit": "50000",
    }
    # GET request, no body -> no Content-Type header needed (that describes
    # the body's format; VendorTelemetryManager.java's POST calls DO send
    # one, for the exact same reason this GET doesn't need it).
    headers = {
        "apikey": CLOUD_KEY,
        "Authorization": f"Bearer {CLOUD_KEY}",
    }
    response = requests.get(url, params=params, headers=headers, timeout=30)
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"{response.status_code} from the backend reading {LOGS_TABLE} -- "
            "CLOUD_KEY is valid-but-insufficiently-privileged for SELECT. "
            "This is almost always: CLOUD_KEY is set to the SAME key the "
            "plugin ships (insert/RPC-only by RLS design), not the "
            "privileged (service_role) key from your backend project's API "
            "settings. Response body: " + response.text[:300]
        )
    if response.status_code == 404:
        raise RuntimeError(
            f"404 from the backend reading {LOGS_TABLE} -- table name wrong, "
            "or it isn't in the exposed/public schema. Response body: "
            + response.text[:300]
        )
    response.raise_for_status()
    return response.json()


# ------------------------------------------------------------ AI consult

# Free-tier inference endpoints occasionally accept a connection and the
# request cleanly (DNS/TCP/TLS all fine) and then just never send a
# response back -- confirmed against this exact provider during setup: a
# single 20s wait produced 0 bytes, from two completely different networks.
# Rather than one long wait, try a few SHORT attempts instead -- a fresh
# attempt often lands on a different backend instance than the one that was
# stuck, so this recovers from that specific failure mode far more often
# than a single longer timeout would. Total worst case stays bounded either
# way: AI_RETRIES attempts * AI_TIMEOUT_SECONDS each, plus a short pause
# between attempts.
AI_TIMEOUT_SECONDS = int(os.environ.get("AI_TIMEOUT_SECONDS", "10"))
AI_RETRIES = int(os.environ.get("AI_RETRIES", "3"))
AI_RETRY_DELAY_SECONDS = float(os.environ.get("AI_RETRY_DELAY_SECONDS", "2"))


def consult_ai(check_key: str, reason: str, today_ratio: float,
                history_ratios: list[float], proposed_pct: float) -> dict | None:
    """Best-effort second opinion. Returns None on ANY failure -- including
    after exhausting all retries -- callers must treat that as "no opinion
    available," never as an error. See the AI_API_KEY comment above for the
    full reasoning on why this can only ever make analyze() MORE
    conservative, never less; this function only gathers the AI's raw
    opinion, the clamp that actually enforces that guarantee lives in
    analyze() itself so it can never be bypassed by a change here alone."""
    if not AI_API_KEY or not AI_API_URL or not AI_MODEL:
        return None
    prompt = (
        f"You are a conservative advisor reviewing an anti-cheat detection "
        f"threshold change before it goes live on real servers. Be concise.\n\n"
        f"Check: {check_key}\n"
        f"Today's flags-per-distinct-player ratio: {today_ratio:.2f}\n"
        f"This check's own historical ratios (older windows, same check): "
        f"{[round(r, 2) for r in history_ratios]}\n"
        f"Statistical reasoning already applied: {reason}\n"
        f"Statistically proposed loosening: {proposed_pct:.1%} "
        f"(the maximum a single run is ever allowed to apply)\n\n"
        f"Does this look like a genuine false-positive pattern (broad, "
        f"shallow flagging across many players -- consistent with ordinary "
        f"legitimate movement/timing edge cases) rather than a real "
        f"coordinated wave of cheating? You may only ever recommend the SAME "
        f"or a SMALLER adjustment than proposed -- never a larger one; if "
        f"unsure, recommend a smaller number or zero.\n\n"
        f"Respond with ONLY this JSON object, nothing else, no markdown "
        f"fences, no explanation outside the JSON:\n"
        f'{{"agree": true or false, "suggested_pct": <number from 0 to '
        f'{proposed_pct}>, "reasoning": "<one short sentence>"}}'
    )
    last_error = None
    for attempt in range(1, AI_RETRIES + 1):
        try:
            response = requests.post(
                AI_API_URL,
                headers={
                    "Authorization": f"Bearer {AI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": AI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 200,
                },
                timeout=AI_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            # models sometimes add stray text or markdown fences around the
            # JSON despite instructions -- extract the outermost {...}
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end == -1:
                print(f"WARNING: AI response for {check_key} had no JSON "
                      f"object in it -- ignoring, statistics decide alone "
                      f"this run.")
                return None
            verdict = json.loads(content[start:end + 1])
            if "suggested_pct" not in verdict or not isinstance(
                    verdict["suggested_pct"], (int, float)):
                print(f"WARNING: AI response for {check_key} was missing a "
                      f"usable suggested_pct -- ignoring, statistics decide "
                      f"alone this run.")
                return None
            return verdict
        except Exception as ex:
            last_error = ex
            if attempt < AI_RETRIES:
                print(f"WARNING: AI consultation for {check_key} attempt "
                      f"{attempt}/{AI_RETRIES} failed ({ex}) -- retrying in "
                      f"{AI_RETRY_DELAY_SECONDS:.0f}s.")
                time.sleep(AI_RETRY_DELAY_SECONDS)
    print(f"WARNING: AI consultation for {check_key} failed after "
          f"{AI_RETRIES} attempts ({last_error}) -- falling back to the "
          f"statistical decision alone, unaffected.")
    return None


# ---------------------------------------------------------------- analysis
#
# THIS IS A STARTING SKELETON, NOT A VALIDATED TUNING MODEL. Ship it, watch
# it for a while, read the summaries it produces (every change is a normal
# git commit in this repo's history, with a real explanation in the message
# body now -- see main() -- that history IS the audit trail and the undo
# button; if a nudge looks wrong, `git revert` it).
#
# Only checks that ALREADY have an entry in `baseline` are ever touched --
# the brain never invents a new check key or a new field name it hasn't
# already seen published, it only nudges values it already knows the shape
# of. That's a deliberate ceiling on what an unattended run can do. It also
# only ever LOOSENS -- there's no auto-tightening path, so the worst an
# unattended bad call can do is make one check briefly more permissive,
# never more punishing. Reverting a bad loosen is a `git revert`.
#
# Each check is compared against ITS OWN historical flag-rate (HISTORY_
# WINDOWS worth), not one flat rule shared by all 200+ checks: an anomaly
# is today's flags/player ratio falling ANOMALY_Z standard deviations below
# that specific check's own historical mean -- broader + shallower than
# USUAL FOR THIS CHECK is the false-positive-shaped signal. A check with
# too little history of its own falls back to a flat v1-style rule instead
# of being skipped outright or guessed at.
#
# On top of that, three sanity rails (see module docstring): a hard per-
# check adjustment-count cap (MAX_ADJUST_COUNT, tracked in brain_state.json)
# so a check can't silently drift forever unattended; a floor that refuses
# to nudge any single field to zero or negative; and a structural check in
# main() that the published check set never silently changes shape.
#
# This still says nothing about WHICH direction "looser" means per field (a
# margin vs. a millisecond window vs. a sample count don't all want the
# same treatment) -- a real v3 needs per-field semantics, which isn't
# something this script can infer on its own. Refine before fully trusting it.

def _window_index(row_time: datetime, now: datetime) -> int:
    """0 = current window, 1..HISTORY_WINDOWS = how many windows back."""
    age_hours = (now - row_time).total_seconds() / 3600.0
    return int(age_hours // LOOKBACK_HOURS)


def _ratio(rows: list[dict]) -> tuple[float, int]:
    """(flags-per-distinct-player, distinct player count) for a set of rows."""
    if not rows:
        return 0.0, 0
    distinct = len({r["player_uuid"] for r in rows})
    return len(rows) / max(distinct, 1), distinct


def analyze(baseline: dict, logs: list[dict], state: dict) -> tuple[dict, dict, list[dict]]:
    """Returns (updated_baseline, updated_state, change_log). change_log is a
    list of dicts (one per adjusted check) used both for the git commit
    message body and the Discord notification -- built once here so both
    consumers stay in sync instead of duplicating the formatting twice."""
    now = datetime.now(timezone.utc)
    by_check: dict[str, dict[int, list[dict]]] = {}
    for row in logs:
        row_time = datetime.fromisoformat(row[TIMESTAMP_COLUMN])
        window = _window_index(row_time, now)
        if window > HISTORY_WINDOWS:
            continue  # older than the history span we asked for; ignore
        by_check.setdefault(row["check_name"], {}).setdefault(window, []).append(row)

    updated = json.loads(json.dumps(baseline))  # deep copy
    new_state = json.loads(json.dumps(state))   # deep copy
    change_log = []
    ai_consultations_used = 0

    for check_key, fields in baseline.items():
        windows = by_check.get(check_key, {})
        current_rows = windows.get(0, [])
        if len(current_rows) < MIN_SAMPLES:
            continue
        today_ratio, today_players = _ratio(current_rows)
        if today_players < MIN_DISTINCT_PLAYERS:
            continue

        history_ratios = []
        for w in range(1, HISTORY_WINDOWS + 1):
            rows = windows.get(w, [])
            ratio, players = _ratio(rows)
            if players >= 2:  # a lone player's ratio is too noisy to count
                history_ratios.append(ratio)

        if len(history_ratios) >= MIN_HISTORY_WINDOWS:
            mean = statistics.mean(history_ratios)
            stdev = statistics.pstdev(history_ratios) or (mean * 0.1 + 0.01)
            z = (mean - today_ratio) / stdev
            looks_false_positive_shaped = z >= ANOMALY_Z
            reason = (f"{today_ratio:.2f} flags/player vs its own "
                      f"{mean:.2f}+/-{stdev:.2f} history over "
                      f"{len(history_ratios)} window(s) (z={z:.2f})")
        else:
            # not enough of this check's own history yet -- fall back to a
            # flat rule rather than doing nothing or guessing wildly
            looks_false_positive_shaped = today_ratio <= 1.5
            reason = (f"{today_ratio:.2f} flags/player, only "
                      f"{len(history_ratios)}/{MIN_HISTORY_WINDOWS} history "
                      f"window(s) so far -- using flat fallback rule")

        if not looks_false_positive_shaped:
            continue

        # ---- sanity rail 1: hard per-check adjustment-count cap ----
        check_state = new_state.get(check_key, {"adjust_count": 0, "last_adjusted": None})
        if check_state.get("adjust_count", 0) >= MAX_ADJUST_COUNT:
            print(f"SANITY: {check_key} looks anomalous again ({reason}) but has "
                  f"already been auto-adjusted {check_state['adjust_count']} times "
                  f"total -- MAX_ADJUST_COUNT={MAX_ADJUST_COUNT} reached. Skipping; "
                  f"a human needs to review this check and clear its entry in "
                  f"{STATE_FILE} before the brain will touch it again.")
            continue

        # ---- AI second opinion (optional) ----
        # Can only talk the adjustment DOWN from what statistics proposed,
        # never up -- see AI_API_KEY's comment for the full reasoning. The
        # min()/max() clamp below is what actually enforces that; nothing
        # the AI says can escape it even if consult_ai()'s own prompt-level
        # instruction were somehow ignored by the model.
        effective_pct = MAX_ADJUST_PCT
        if AI_API_KEY and AI_API_URL and AI_MODEL and ai_consultations_used < MAX_AI_CONSULTATIONS:
            ai_consultations_used += 1
            verdict = consult_ai(check_key, reason, today_ratio, history_ratios, MAX_ADJUST_PCT)
            if verdict is not None:
                suggested = max(0.0, min(float(verdict["suggested_pct"]), MAX_ADJUST_PCT))
                if suggested < effective_pct:
                    ai_reasoning = verdict.get("reasoning", "(no reason given)")
                    reason += f" | AI reduced {effective_pct:.0%} -> {suggested:.0%}: {ai_reasoning}"
                    effective_pct = suggested

        if effective_pct <= 0:
            print(f"  {check_key}: AI reduced this adjustment to zero -- skipping "
                  f"({reason})")
            continue

        # ---- sanity rail 2: never nudge a field to zero/negative ----
        touched_fields = {}
        skipped_fields = []
        for field_name, value in fields.items():
            if not isinstance(value, (int, float)):
                continue
            nudged = value * (1 + effective_pct)
            nudged = round(nudged) if isinstance(value, int) else round(nudged, 6)
            if nudged <= 0:
                skipped_fields.append(field_name)
                continue
            updated[check_key][field_name] = nudged
            touched_fields[field_name] = (value, nudged)
        if skipped_fields:
            print(f"SANITY: {check_key} -- refused to nudge {skipped_fields} to a "
                  f"non-positive value; left unchanged. This should never happen "
                  f"with sane starting values -- worth checking data.txt by hand.")
        if not touched_fields:
            continue  # every field on this check got vetoed by the floor above

        check_state["adjust_count"] = check_state.get("adjust_count", 0) + 1
        check_state["last_adjusted"] = now.isoformat()
        new_state[check_key] = check_state

        change_log.append({
            "check": check_key,
            "flags": len(current_rows),
            "players": today_players,
            "reason": reason,
            "fields": touched_fields,
            "adjust_count": check_state["adjust_count"],
        })

    if change_log:
        print("Adjusted checks this run:")
        for c in change_log:
            field_summary = ", ".join(
                f"{name} {old}->{new}" for name, (old, new) in c["fields"].items())
            print(f"  {c['check']}: {c['flags']} flags across {c['players']} players "
                  f"today -- {c['reason']} -> {field_summary} "
                  f"(adjustment #{c['adjust_count']}/{MAX_ADJUST_COUNT})")
    else:
        print("No check crossed the adjustment threshold this run -- baseline unchanged.")

    return updated, new_state, change_log


# ---------------------------------------------------------------- reporting

def build_summary(change_log: list[dict], total_hours: float, log_count: int) -> str:
    """Human-readable report used for BOTH the git commit message body and
    the Discord notification, so the two never drift out of sync."""
    lines = [f"Saturn Cloud Brain: {len(change_log)} check(s) adjusted "
             f"({log_count} log rows over the last {total_hours:.0f}h)."]
    for c in change_log:
        field_summary = ", ".join(
            f"{name} {old}->{new}" for name, (old, new) in c["fields"].items())
        lines.append(f"- {c['check']}: {field_summary}")
        lines.append(f"    {c['flags']} flags / {c['players']} players -- {c['reason']}")
        lines.append(f"    adjustment #{c['adjust_count']} of {MAX_ADJUST_COUNT} "
                      f"allowed before this check needs human review")
    return "\n".join(lines)


def notify_discord(summary: str, change_log: list[dict]) -> None:
    """Best-effort only. A missing/misconfigured/unreachable webhook must
    NEVER fail the run or block writing data.txt -- this is a nice-to-have
    notification, not part of the critical path."""
    if not DISCORD_WEBHOOK_URL or not change_log:
        return
    try:
        description = summary[:4000]  # Discord embed description hard limit is 4096
        payload = {
            "embeds": [{
                "title": f"Saturn AntiCheat -- {len(change_log)} check(s) auto-tuned",
                "description": description,
                "color": 0xFFD700,
            }]
        }
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code >= 300:
            print(f"WARNING: Discord webhook returned {resp.status_code}: "
                  f"{resp.text[:200]} -- continuing, this is non-fatal.")
    except Exception as ex:
        print(f"WARNING: Discord webhook failed ({ex}) -- continuing, this is non-fatal.")


# ---------------------------------------------------------------- main

def main() -> int:
    """Always returns 0 on a successful run, whether or not anything changed
    -- "no adjustment needed this cycle" is the normal case, not a failure.
    Whether data.txt/brain_state.json actually differ (and are therefore
    worth committing) is decided by the workflow via `git diff`, not by
    this exit code."""
    baseline = load_baseline()
    state = load_state()
    print(f"Loaded baseline: {len(baseline)} check(s) known. "
          f"{len(state)} check(s) have adjustment history.")

    total_hours = LOOKBACK_HOURS * (HISTORY_WINDOWS + 1)
    try:
        logs = fetch_anomaly_logs()
        print(f"Fetched {len(logs)} anomaly log row(s) spanning the last "
              f"{total_hours:.0f}h ({LOOKBACK_HOURS:.0f}h current window + "
              f"{HISTORY_WINDOWS} history window(s)).")
    except Exception as ex:
        print(f"WARNING: backend fetch failed ({ex}) -- leaving baseline untouched.")
        logs = []

    if baseline:
        updated, new_state, change_log = analyze(baseline, logs, state)
    else:
        updated, new_state, change_log = baseline, state, []

    # ---- sanity rail 3: the published check set must never silently
    # change shape -- analyze() only ever edits VALUES on keys already in
    # baseline, so this should be structurally impossible, but a hard
    # assertion here means a future bug in analyze() fails LOUD in CI
    # instead of quietly shipping a corrupted data.txt to every server.
    if set(updated.keys()) != set(baseline.keys()):
        added = set(updated.keys()) - set(baseline.keys())
        removed = set(baseline.keys()) - set(updated.keys())
        sys.exit(f"SANITY FAILURE: analyze() changed the published check set "
                  f"(added={added}, removed={removed}) -- refusing to write "
                  f"{DATA_FILE}. This is a bug in analyze(), not a data issue; "
                  f"nothing was written, the previous data.txt is untouched.")

    new_json = json.dumps(updated, separators=(",", ":"), sort_keys=True)
    old_json = json.dumps(baseline, separators=(",", ":"), sort_keys=True)
    data_changed = new_json != old_json
    state_changed = new_state != state

    if not data_changed:
        print("Nothing changed -- not rewriting data.txt.")
    else:
        # a fresh random IV means the ciphertext differs even when the
        # underlying JSON doesn't -- only re-encrypt (and let the workflow
        # diff data.txt) when the DECRYPTED content actually changed
        encrypted = obf_encrypt(new_json)
        with open(DATA_FILE, "w", encoding="utf-8", newline="\n") as f:
            f.write(encrypted + "\n")

        # ---- sanity rail 4: post-write round-trip self-check ----
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            written_back = f.read().strip()
        try:
            round_tripped = json.loads(obf_decrypt(written_back))
        except Exception as ex:
            sys.exit(f"SANITY FAILURE: just-written {DATA_FILE} failed to "
                      f"decrypt/parse back ({ex}) -- something is badly wrong "
                      f"with obf_encrypt() or the write itself. The bad file is "
                      f"still on disk for inspection; DO NOT let the workflow "
                      f"commit it.")
        if round_tripped != updated:
            sys.exit(f"SANITY FAILURE: {DATA_FILE} round-tripped to different "
                      f"content than intended -- refusing to trust this run. "
                      f"DO NOT let the workflow commit it.")
        print(f"Wrote updated {DATA_FILE} ({len(updated)} check(s)), verified "
              f"round-trip clean.")

    if state_changed:
        with open(STATE_FILE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(new_state, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote updated {STATE_FILE}.")

    summary = build_summary(change_log, total_hours, len(logs))
    with open(SUMMARY_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(summary + "\n")
    if change_log:
        notify_discord(summary, change_log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
