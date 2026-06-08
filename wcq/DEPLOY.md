# Deployment Guide — WC 2026 Quant Discord Bot

Two targets, each independently operational:
- **GitHub Actions** — three scheduled jobs (digest, pre-match, post-match)
- **Railway** — always-on live poller

---

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_WEBHOOK_URL` | Yes | — | Full Discord webhook URL |
| `WCQ_DB_PATH` | No | `wcq_bot.db` | Path to the SQLite file |
| `WCQ_SCHEDULE_PATH` | No | `data/schedule_2026.json` | Path to the match schedule cache |
| `WCQ_MARKET_CACHE_PATH` | No | `data/market_cache.json` | Polymarket/Kalshi discovery cache |
| `FOOTBALL_DATA_API_KEY` | Recommended | — | football-data.org free tier key |
| `DRY_RUN` | No | `0` | Set to `1` to print embeds without posting |
| `POLL_INTERVAL` | No | `75` | Seconds between live polls (Railway only) |
| `EDGE_THRESHOLD` | No | `0.06` | Minimum model vs market edge to alert |
| `PRICE_MOVE_THRESHOLD` | No | `0.05` | Minimum price move to alert on reprice |
| `SPREAD_THRESHOLD` | No | `0.04` | Minimum Poly–Kalshi spread to alert |
| `WCQ_KELLY_EDGE_THRESHOLD` | No | `0.03` | Minimum edge to place a paper bet |
| `WCQ_PAPER_BANKROLL` | No | `1000.0` | Paper trading starting bankroll |

---

## Part 1: GitHub Actions — three scheduled jobs

### How state persists across runs

Each job writes to a SQLite file (`wcq_bot.db`) and commits it back to the repo.
The workflow grants write permission via `GITHUB_TOKEN` (default in all repos).
The file is committed with `[skip ci]` in the message to prevent workflow loops.

**Important:** add `wcq_bot.db` to your `.gitignore` exclusion list (i.e., do NOT
gitignore it — it must be tracked). If it's currently gitignored, remove that line.

### Step 1 — Add secrets to GitHub

Go to **Settings → Secrets and variables → Actions** in your repo.

Add these **Repository secrets**:

| Secret name | Value |
|---|---|
| `DISCORD_WEBHOOK_URL` | Your Discord server webhook URL |
| `FOOTBALL_DATA_API_KEY` | Key from football-data.org (free registration) |

Optionally add as a **Repository variable** (not secret):

| Variable name | Value |
|---|---|
| `WCQ_KELLY_EDGE_THRESHOLD` | `0.03` (or your preferred threshold) |

The `WCQ_DB_PATH` secret is optional — omit it and the workflows default to
`wcq_bot.db` in the `wcq/` working directory.

### Step 2 — Download the match schedule once

Run this locally (or add a one-time manual workflow):

```bash
cd wcq
python src/bot/fixtures.py --download
git add data/schedule_2026.json
git commit -m "add WC 2026 match schedule"
git push
```

The file is a JSON cache of all 104 fixtures. Knockout matchups get added
automatically by `market_discovery.py` as teams advance.

### Step 3 — Seed the DB (first time only)

```bash
cd wcq
python src/bot/storage.py      # creates wcq_bot.db
git add -f wcq_bot.db
git commit -m "seed empty bot DB"
git push
```

### Step 4 — Enable the workflows

The three workflow files are already in `.github/workflows/`. They run automatically
once pushed to `main`. You can verify they're active under **Actions** in the GitHub UI.

To test without posting, add `DRY_RUN: "1"` to the workflow env section, or
use **Run workflow** → dispatch from the Actions tab.

### Cron schedules (UTC)

| Job | Cron | UTC time | ET equivalent |
|---|---|---|---|
| Daily digest | `0 7 * * *` | 07:00 daily | 03:00 EDT |
| Pre-match briefing | `*/30 * * * *` | every 30 min | — |
| Post-match scorecard | `*/30 * * * *` | every 30 min | — |

The briefing and scorecard jobs are idempotent — they check a dedup key before
posting, so running every 30 minutes is safe and ensures no match is missed.

### How to confirm each job is working

1. Check **Actions** → click a run → read the job log for `[daily_digest]`,
   `[pre_match]`, or `[post_match]` output lines.
2. Check that `wcq_bot.db` in the repo grows over time (newer commits).
3. Look at your Discord channel for embeds.

---

## Part 2: Railway — always-on live poller

The live poller runs as a long-lived asyncio process, waking up every 60 seconds
to check for live matches, then polling per-match Polymarket/Kalshi markets every
75 seconds during each match window.

### Step 1 — Create a Railway project

1. Go to [railway.app](https://railway.app), create a new project.
2. Connect your GitHub repo.
3. Railway will detect `wcq/` as the build root via `railway.json`.

### Step 2 — Configure the volume (SQLite persistence)

The live poller writes to the same `wcq_bot.db` file as the GitHub Actions jobs.
On Railway, use a persistent volume so the DB survives deploys.

In Railway → your service → **Volumes**:
- Mount path: `/data`
- Set `WCQ_DB_PATH` env var to `/data/wcq_bot.db`

Also mount the schedule file:
- Set `WCQ_SCHEDULE_PATH` to `/data/schedule_2026.json`
- On first deploy, copy `data/schedule_2026.json` to the volume by running a
  one-off command in the Railway shell: `cp /app/wcq/data/schedule_2026.json /data/`

### Step 3 — Set environment variables in Railway

In Railway → your service → **Variables**, add:

```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
WCQ_DB_PATH=/data/wcq_bot.db
WCQ_SCHEDULE_PATH=/data/schedule_2026.json
FOOTBALL_DATA_API_KEY=your_key_here
POLL_INTERVAL=75
EDGE_THRESHOLD=0.06
PRICE_MOVE_THRESHOLD=0.05
SPREAD_THRESHOLD=0.04
```

### Step 4 — Deploy

Push to `main` — Railway redeploys automatically. Confirm the worker is running
via **Deployments** → click the latest → check logs for:

```
[live_poller] Live poller started (poll_interval=75s, edge_threshold=6%)
```

### How to confirm the live poller is working

- During a match window, logs should show per-match polls every ~75 seconds.
- Between matches, logs show `No live or upcoming matches; sleeping 60s` every minute.
- Alerts fire to Discord when edges or reprices cross thresholds.
- Check Railway's **Metrics** tab — the process should be continuously running,
  not crashing/restarting frequently.

---

## Troubleshooting

**No embeds posting:**
- Check `DISCORD_WEBHOOK_URL` is set correctly (full URL, not just the ID).
- Set `DRY_RUN=1` and check the job log for the payload.

**DB out of sync between GitHub Actions and Railway:**
- These are two separate DB files. The GitHub Actions DB tracks calibration and
  predictions; the Railway DB tracks live alerts. They don't need to be the same
  file — the dedup key namespacing prevents duplicate alerts.
- If you want a single shared DB, host it on a network volume (e.g., Turso/libSQL)
  and change the storage module's `_conn()` to use a networked driver. For a dev
  project with a few friends, separate DBs is simpler.

**Schedule not downloaded / No fixtures today:**
- Run `python src/bot/fixtures.py --download` locally and commit the resulting
  `data/schedule_2026.json`.

**`FOOTBALL_DATA_API_KEY` rate limit:**
- The free tier allows ~10 req/min. The post-match job runs every 30 minutes and
  makes at most a handful of requests per run, so limits should not be an issue.
  If they are, increase `_RESULT_WINDOW_MIN` in `post_match.py` to widen the window.

**Elo ratings slow to compute:**
- `compute_elo(load_results())` re-trains on ~50k matches on every GitHub Actions
  run (~2–3s). This is acceptable for the current project scale. If it becomes
  a bottleneck, cache the final ratings dict to `data/elo_cache.json` and
  refresh it only when new historical data is downloaded.
