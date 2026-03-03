#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


class MetricsStore:
    def __init__(
        self,
        *,
        metrics_file: Path,
        history_limit: int,
        queue_size: int,
        poll_interval_sec: float = 0.5,
    ) -> None:
        self.metrics_file = metrics_file
        self.history_limit = max(32, int(history_limit))
        self.queue_size = max(1, int(queue_size))
        self.poll_interval_sec = max(0.2, float(poll_interval_sec))

        self._records: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)
        self._offset = 0
        self._inode: Optional[int] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="node-producer-metrics-poller",
            daemon=True,
        )

        self._bootstrap()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _bootstrap(self) -> None:
        if not self.metrics_file.is_file():
            return
        try:
            stat = self.metrics_file.stat()
            self._inode = int(getattr(stat, "st_ino", 0))
            with self.metrics_file.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = self._parse_line(line)
                    if rec is not None:
                        self._records.append(rec)
                self._offset = f.tell()
        except Exception:
            logging.exception("Failed to bootstrap metrics from %s", self.metrics_file)

    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        s = line.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            if not isinstance(obj, dict):
                return None
            return obj
        except Exception:
            return None

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval_sec):
            self._poll_once()

    def _poll_once(self) -> None:
        if not self.metrics_file.is_file():
            return
        try:
            stat = self.metrics_file.stat()
            inode = int(getattr(stat, "st_ino", 0))
            size = int(stat.st_size)

            if self._inode is None:
                self._inode = inode
            if inode != self._inode or size < self._offset:
                # rotated / truncated
                self._inode = inode
                self._offset = 0

            if size == self._offset:
                return

            with self.metrics_file.open("r", encoding="utf-8") as f:
                f.seek(self._offset)
                new_records: List[Dict[str, Any]] = []
                for line in f:
                    rec = self._parse_line(line)
                    if rec is not None:
                        new_records.append(rec)
                self._offset = f.tell()

            if not new_records:
                return

            with self._lock:
                for rec in new_records:
                    self._records.append(rec)
        except Exception:
            logging.exception("Failed to poll metrics file %s", self.metrics_file)

    def _records_list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def snapshot(self, limit: int = 300) -> Dict[str, Any]:
        records = self._records_list()
        exists = self.metrics_file.is_file()
        latest = records[-1] if records else {}

        queue_depth = latest.get("queue_depth_per_rank", [])
        consumed = latest.get("consumed_step_per_rank", [])
        waits = latest.get("consumer_wait_ms_per_rank", [])
        rank_count = max(
            len(queue_depth) if isinstance(queue_depth, list) else 0,
            len(consumed) if isinstance(consumed, list) else 0,
            len(waits) if isinstance(waits, list) else 0,
            1,
        )

        recent = records[-max(8, int(limit)) :]
        ts = [_to_float(r.get("ts"), 0.0) for r in recent]
        rate = [_to_float(r.get("produced_steps_per_sec"), 0.0) for r in recent]
        lag = [_to_int(r.get("lag_steps"), 0) for r in recent]
        produced_total = [_to_int(r.get("produced_steps_total"), 0) for r in recent]

        consumed_min_series: List[int] = []
        queue_mean_series: List[float] = []
        for r in recent:
            c = r.get("consumed_step_per_rank", [])
            q = r.get("queue_depth_per_rank", [])
            c_vals = [int(x) for x in c if isinstance(x, int) and x >= 0] if isinstance(c, list) else []
            q_vals = [float(x) for x in q if isinstance(x, (int, float)) and x >= 0] if isinstance(q, list) else []
            consumed_min_series.append(min(c_vals) if c_vals else -1)
            queue_mean_series.append(sum(q_vals) / len(q_vals) if q_vals else 0.0)

        def slope(values: List[float], times: List[float]) -> float:
            if len(values) < 2 or len(times) < 2:
                return 0.0
            dt = times[-1] - times[0]
            if dt <= 1e-6:
                return 0.0
            return (values[-1] - values[0]) / dt

        produced_step_rate = slope([float(v) for v in produced_total], ts)
        consumed_step_rate = slope([float(v) for v in consumed_min_series], ts)

        last_step = _to_int(latest.get("last_step_id"), -1)
        ranks: List[Dict[str, Any]] = []
        q_latest = queue_depth if isinstance(queue_depth, list) else []
        c_latest = consumed if isinstance(consumed, list) else []
        w_latest = waits if isinstance(waits, list) else []
        for i in range(rank_count):
            q = _to_int(q_latest[i], -1) if i < len(q_latest) else -1
            c = _to_int(c_latest[i], -1) if i < len(c_latest) else -1
            w = _to_float(w_latest[i], 0.0) if i < len(w_latest) else 0.0
            backlog = max(0, last_step - c) if (last_step >= 0 and c >= 0) else -1
            ranks.append(
                {
                    "rank": i,
                    "queue_depth": q,
                    "queue_fill_ratio": (q / self.queue_size) if q >= 0 else 0.0,
                    "consumed_step": c,
                    "backlog_steps": backlog,
                    "consumer_wait_ms": w,
                }
            )

        events: List[Dict[str, Any]] = []
        for r in records[-24:]:
            events.append(
                {
                    "ts": _to_float(r.get("ts"), 0.0),
                    "event": str(r.get("event", "")),
                    "epoch": _to_int(r.get("epoch"), -1),
                    "produced_steps_total": _to_int(r.get("produced_steps_total"), 0),
                    "lag_steps": _to_int(r.get("lag_steps"), 0),
                }
            )

        return {
            "ok": True,
            "metrics_file": str(self.metrics_file),
            "metrics_exists": bool(exists),
            "server_time": time.time(),
            "record_count": len(records),
            "queue_size": self.queue_size,
            "latest": latest,
            "pipeline": {
                "fetch_rate_steps_per_sec": _to_float(latest.get("produced_steps_per_sec"), 0.0),
                "process_rate_steps_per_sec": produced_step_rate,
                "consume_rate_steps_per_sec": consumed_step_rate,
                "lag_steps": _to_int(latest.get("lag_steps"), 0),
                "produced_total": _to_int(latest.get("produced_steps_total"), 0),
                "consumed_min": min(
                    [r["consumed_step"] for r in ranks if r["consumed_step"] >= 0],
                    default=-1,
                ),
                "producer_alive": _to_int(latest.get("producer_alive"), 0),
            },
            "history": {
                "ts": ts,
                "produced_steps_per_sec": rate,
                "lag_steps": lag,
                "produced_steps_total": produced_total,
                "consumed_min_step": consumed_min_series,
                "queue_mean_depth": queue_mean_series,
            },
            "ranks": ranks,
            "events": events,
        }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Node Producer Observatory</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
    :root {
      --bg: #090d14;
      --panel: rgba(11, 20, 30, 0.72);
      --panel-strong: rgba(13, 24, 36, 0.93);
      --text: #d7e7f0;
      --muted: #86a4b4;
      --line: rgba(110, 185, 210, 0.26);
      --cyan: #66f2ff;
      --lime: #b9ff66;
      --amber: #ffbe5e;
      --rose: #ff6d90;
      --shadow: 0 16px 34px rgba(0, 0, 0, 0.48);
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0; min-height: 100%;
      background: radial-gradient(1200px 700px at 10% -10%, #143349 0%, transparent 60%),
                  radial-gradient(900px 520px at 100% 0%, #3f1f2b 0%, transparent 65%),
                  linear-gradient(160deg, #080c13 0%, #0c1722 45%, #0a1018 100%);
      color: var(--text);
      font-family: "Rajdhani", "Trebuchet MS", sans-serif;
    }
    body::before {
      content: "";
      position: fixed; inset: 0; pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
      background-size: 46px 46px;
      opacity: 0.18;
      mix-blend-mode: screen;
    }
    .wrap {
      width: min(1400px, 96vw);
      margin: 22px auto 26px;
      display: grid;
      gap: 14px;
    }
    .top {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 14px 18px;
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 10px;
      align-items: center;
      backdrop-filter: blur(9px);
    }
    .title {
      font-size: clamp(22px, 2.8vw, 40px);
      line-height: 1;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      text-shadow: 0 0 18px rgba(104, 223, 255, 0.32);
    }
    .subtitle {
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
      letter-spacing: 0.03em;
    }
    .statusline {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .chip {
      border: 1px solid var(--line);
      padding: 7px 12px;
      border-radius: 999px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      color: var(--muted);
      background: rgba(12, 24, 35, 0.62);
    }
    .chip strong { color: var(--text); font-weight: 600; }
    .chip.live { border-color: rgba(168, 255, 105, 0.55); color: #b8ff8f; }
    .chip.dead { border-color: rgba(255, 124, 161, 0.55); color: #ff9fb9; }

    .pipeline {
      display: grid;
      grid-template-columns: 1fr auto 1fr auto 1fr;
      gap: 12px;
      align-items: stretch;
    }
    .stage {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 14px;
      position: relative;
      overflow: hidden;
    }
    .stage::after {
      content: "";
      position: absolute; inset: -100% 0 auto 0; height: 220%;
      background: linear-gradient(180deg, rgba(102,242,255,0.08), transparent 26%, transparent 74%, rgba(255,190,94,0.08));
      animation: scan 4s linear infinite;
      pointer-events: none;
    }
    @keyframes scan { to { transform: translateY(35%); } }
    .stage h3 {
      margin: 0 0 8px; font-size: 13px; text-transform: uppercase;
      letter-spacing: 0.14em; color: var(--muted);
    }
    .stage .value {
      font-family: "IBM Plex Mono", monospace;
      font-size: clamp(20px, 2.4vw, 34px);
      letter-spacing: 0.04em;
    }
    .stage .caption {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      font-family: "IBM Plex Mono", monospace;
    }
    .connector {
      align-self: center;
      width: 38px; height: 2px;
      background: linear-gradient(90deg, rgba(102,242,255,0), rgba(102,242,255,0.95), rgba(102,242,255,0));
      position: relative;
    }
    .connector::before {
      content: "";
      position: absolute; top: -4px; right: -3px;
      border-top: 5px solid transparent;
      border-bottom: 5px solid transparent;
      border-left: 8px solid rgba(102,242,255,0.95);
    }

    .main {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(9px);
      padding: 14px;
    }
    .panel h4 {
      margin: 0 0 10px;
      text-transform: uppercase;
      letter-spacing: 0.13em;
      font-size: 12px;
      color: var(--muted);
    }
    .chart {
      width: 100%;
      height: 180px;
      border-radius: 12px;
      border: 1px solid rgba(125, 188, 212, 0.2);
      background: linear-gradient(180deg, rgba(13, 24, 35, 0.92), rgba(8, 15, 23, 0.94));
    }
    .legend {
      display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px; color: var(--muted);
    }
    .dot {
      display: inline-block; width: 10px; height: 10px; border-radius: 99px; margin-right: 6px;
    }

    .ranks {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      max-height: 520px;
      overflow: auto;
      padding-right: 2px;
    }
    .rank {
      border: 1px solid rgba(132, 200, 224, 0.28);
      background: rgba(10, 18, 28, 0.86);
      border-radius: 12px;
      padding: 10px;
    }
    .rank-top {
      display: flex; justify-content: space-between; align-items: center;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px; color: var(--muted);
      margin-bottom: 6px;
    }
    .rank .v {
      font-size: 18px; color: var(--text); font-weight: 700;
    }
    .bar {
      height: 9px;
      border-radius: 99px;
      background: rgba(130, 172, 190, 0.18);
      overflow: hidden;
      margin: 7px 0 9px;
    }
    .fill {
      height: 100%;
      background: linear-gradient(90deg, var(--lime), var(--cyan));
      box-shadow: 0 0 14px rgba(118, 255, 170, 0.45);
      width: 0%;
      transition: width 0.4s ease;
    }
    .meta {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      color: var(--muted);
    }
    .events {
      margin-top: 14px;
      max-height: 230px;
      overflow: auto;
      border-radius: 12px;
      border: 1px solid rgba(132, 200, 224, 0.2);
      background: rgba(7, 13, 20, 0.88);
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      color: #c6d8e4;
    }
    .event {
      padding: 8px 10px;
      border-bottom: 1px solid rgba(123, 180, 201, 0.16);
      display: grid;
      grid-template-columns: 84px 90px 1fr;
      gap: 8px;
    }
    .event:last-child { border-bottom: none; }
    .event .ts { color: var(--muted); }
    .event .evt { color: var(--amber); text-transform: uppercase; letter-spacing: 0.04em; }

    @media (max-width: 1080px) {
      .top { grid-template-columns: 1fr; }
      .statusline { justify-content: flex-start; }
      .main { grid-template-columns: 1fr; }
      .ranks { grid-template-columns: 1fr; }
      .pipeline { grid-template-columns: 1fr; }
      .connector { display: none; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="top">
      <div>
        <div class="title">Node Producer Observatory</div>
        <div class="subtitle">Realtime Fetch -> Process -> Consume telemetry by rank</div>
      </div>
      <div class="statusline">
        <div id="chip-live" class="chip">status: <strong>booting</strong></div>
        <div class="chip">records: <strong id="record-count">0</strong></div>
        <div class="chip">file: <strong id="file-short">-</strong></div>
        <div class="chip">updated: <strong id="updated-at">-</strong></div>
      </div>
    </section>

    <section class="pipeline">
      <article class="stage">
        <h3>Acquire (Fetch)</h3>
        <div id="fetch-rate" class="value">0.00</div>
        <div class="caption">steps/s from producer heartbeat</div>
      </article>
      <div class="connector"></div>
      <article class="stage">
        <h3>Process (Pack)</h3>
        <div id="process-rate" class="value">0.00</div>
        <div class="caption">effective produced steps/s over history</div>
      </article>
      <div class="connector"></div>
      <article class="stage">
        <h3>Consume (Ranks)</h3>
        <div id="consume-rate" class="value">0.00</div>
        <div class="caption">min-rank consumed steps/s over history</div>
      </article>
    </section>

    <section class="main">
      <section class="panel">
        <h4>Pipeline Trends</h4>
        <canvas id="chart-rate" class="chart" width="1000" height="300"></canvas>
        <div class="legend">
          <span><span class="dot" style="background:#66f2ff"></span>produced steps/s</span>
          <span><span class="dot" style="background:#ffbe5e"></span>queue mean depth</span>
        </div>
        <div style="height:10px"></div>
        <canvas id="chart-lag" class="chart" width="1000" height="260"></canvas>
        <div class="legend">
          <span><span class="dot" style="background:#ff6d90"></span>lag steps</span>
          <span><span class="dot" style="background:#b9ff66"></span>produced-total minus consumed-min</span>
        </div>
        <div class="events" id="events"></div>
      </section>

      <section class="panel">
        <h4>Consumer Ranks</h4>
        <div id="ranks" class="ranks"></div>
      </section>
    </section>
  </div>

  <script>
    const REFRESH_MS = __REFRESH_MS__;
    const HISTORY_LIMIT = __HISTORY_LIMIT__;

    const byId = (id) => document.getElementById(id);
    const fmt = (v, d=2) => (Number.isFinite(v) ? Number(v).toFixed(d) : "-");
    const fmtInt = (v) => (Number.isFinite(v) ? String(Math.trunc(v)) : "-");

    function drawSeries(canvas, seriesList, colors, yMin=null, yMax=null) {
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      ctx.fillStyle = "rgba(10, 16, 24, 0.92)";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "rgba(138, 188, 208, 0.16)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 5; i++) {
        const y = (h / 5) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      let all = [];
      seriesList.forEach(arr => all = all.concat(arr.filter(Number.isFinite)));
      if (!all.length) return;
      const minV = (yMin !== null) ? yMin : Math.min(...all);
      const maxV = (yMax !== null) ? yMax : Math.max(...all);
      const span = Math.max(1e-9, maxV - minV);

      seriesList.forEach((arr, idx) => {
        if (!arr.length) return;
        ctx.strokeStyle = colors[idx % colors.length];
        ctx.lineWidth = 2.1;
        ctx.beginPath();
        arr.forEach((v, i) => {
          const x = arr.length <= 1 ? 0 : (i / (arr.length - 1)) * (w - 1);
          const y = h - ((v - minV) / span) * (h - 1);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
    }

    function renderRanks(ranks, queueSize) {
      const root = byId("ranks");
      root.innerHTML = "";
      ranks.forEach((r) => {
        const fill = Math.max(0, Math.min(100, (r.queue_fill_ratio || 0) * 100));
        const el = document.createElement("div");
        el.className = "rank";
        el.innerHTML = `
          <div class="rank-top">
            <span>rank ${r.rank}</span>
            <span>q ${fmtInt(r.queue_depth)}/${fmtInt(queueSize)}</span>
          </div>
          <div class="v">${fmtInt(r.consumed_step)}</div>
          <div class="bar"><div class="fill" style="width:${fill}%"></div></div>
          <div class="meta">
            <div>backlog: <strong>${fmtInt(r.backlog_steps)}</strong></div>
            <div>wait_ms: <strong>${fmt(r.consumer_wait_ms, 2)}</strong></div>
          </div>
        `;
        root.appendChild(el);
      });
    }

    function renderEvents(events) {
      const root = byId("events");
      root.innerHTML = "";
      [...events].reverse().slice(0, 24).forEach((e) => {
        const d = new Date((e.ts || 0) * 1000);
        const t = Number.isFinite(d.getTime()) ? d.toLocaleTimeString() : "-";
        const row = document.createElement("div");
        row.className = "event";
        row.innerHTML = `
          <span class="ts">${t}</span>
          <span class="evt">${e.event || "-"}</span>
          <span>epoch=${fmtInt(e.epoch)} total=${fmtInt(e.produced_steps_total)} lag=${fmtInt(e.lag_steps)}</span>
        `;
        root.appendChild(row);
      });
    }

    async function pull() {
      try {
        const res = await fetch(`/api/state?limit=${HISTORY_LIMIT}`, { cache: "no-store" });
        const s = await res.json();

        const live = s?.pipeline?.producer_alive ? true : false;
        const chip = byId("chip-live");
        chip.className = live ? "chip live" : "chip dead";
        chip.innerHTML = `status: <strong>${live ? "live" : "stopped"}</strong>`;

        byId("record-count").textContent = fmtInt(s.record_count || 0);
        byId("file-short").textContent = (s.metrics_file || "-").split("/").slice(-2).join("/");
        byId("updated-at").textContent = new Date().toLocaleTimeString();

        byId("fetch-rate").textContent = fmt(s.pipeline.fetch_rate_steps_per_sec, 2);
        byId("process-rate").textContent = fmt(s.pipeline.process_rate_steps_per_sec, 2);
        byId("consume-rate").textContent = fmt(s.pipeline.consume_rate_steps_per_sec, 2);

        const hist = s.history || {};
        const produced = hist.produced_steps_per_sec || [];
        const qmean = hist.queue_mean_depth || [];
        drawSeries(byId("chart-rate"), [produced, qmean], ["#66f2ff", "#ffbe5e"]);

        const lag = hist.lag_steps || [];
        const producedTotal = hist.produced_steps_total || [];
        const consumedMin = hist.consumed_min_step || [];
        const backlog = producedTotal.map((v, i) => {
          const c = consumedMin[i];
          if (!Number.isFinite(v) || !Number.isFinite(c) || c < 0) return 0;
          return Math.max(0, v - c);
        });
        drawSeries(byId("chart-lag"), [lag, backlog], ["#ff6d90", "#b9ff66"]);

        renderRanks(s.ranks || [], s.queue_size || 32);
        renderEvents(s.events || []);
      } catch (err) {
        const chip = byId("chip-live");
        chip.className = "chip dead";
        chip.innerHTML = `status: <strong>disconnected</strong>`;
      }
    }

    pull();
    setInterval(pull, REFRESH_MS);
  </script>
</body>
</html>
"""


def make_handler(store: MetricsStore, refresh_ms: int, history_limit: int):
    html = (
        HTML_TEMPLATE.replace("__REFRESH_MS__", str(int(refresh_ms)))
        .replace("__HISTORY_LIMIT__", str(int(history_limit)))
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: Dict[str, Any], code: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if parsed.path == "/api/health":
                self._json(
                    {
                        "ok": True,
                        "metrics_file": str(store.metrics_file),
                        "metrics_exists": store.metrics_file.is_file(),
                        "time": time.time(),
                    }
                )
                return

            if parsed.path == "/api/state":
                qs = parse_qs(parsed.query)
                limit = _to_int(qs.get("limit", [history_limit])[0], history_limit)
                self._json(store.snapshot(limit=limit))
                return

            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            logging.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a visual dashboard for node_producer_metrics.jsonl"
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        required=True,
        help="Path to node_producer_metrics.jsonl",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--history-limit",
        type=int,
        default=1200,
        help="Max in-memory metrics records",
    )
    parser.add_argument(
        "--history-points",
        type=int,
        default=320,
        help="Default points returned to frontend per pull",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=32,
        help="Producer queue size, for fill-ratio display",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=1000,
        help="Frontend polling interval in ms",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    store = MetricsStore(
        metrics_file=args.metrics_file,
        history_limit=args.history_limit,
        queue_size=args.queue_size,
    )
    handler = make_handler(
        store=store,
        refresh_ms=args.refresh_ms,
        history_limit=args.history_points,
    )
    httpd = ThreadingHTTPServer((args.host, int(args.port)), handler)
    logging.info(
        "Node producer dashboard serving on http://%s:%s (metrics=%s)",
        args.host,
        args.port,
        args.metrics_file,
    )
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        logging.info("Stopping dashboard...")
    finally:
        store.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
