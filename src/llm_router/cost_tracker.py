"""Cost tracking: per-request logging to SQLite, daily rollups and
budget alerts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class BudgetStatus:
    month: str
    spent_usd: float
    budget_usd: float
    fraction_used: float
    alert: str | None


class CostTracker:
    def __init__(self, db_path: str | Path = "data/costs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model TEXT NOT NULL,
                complexity TEXT,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                cached INTEGER NOT NULL DEFAULT 0,
                latency_ms INTEGER
            )"""
        )
        self._conn.commit()

    def log(self, model: str, input_tokens: int, output_tokens: int,
            cost_usd: float, complexity: str | None = None,
            cached: bool = False, latency_ms: int | None = None,
            ts: str | None = None) -> int:
        ts = ts or date.today().isoformat()
        cur = self._conn.execute(
            "INSERT INTO requests (ts, model, complexity, input_tokens,"
            " output_tokens, cost_usd, cached, latency_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, model, complexity, input_tokens, output_tokens,
             cost_usd, 1 if cached else 0, latency_ms),
        )
        self._conn.commit()
        return cur.lastrowid

    def total_cost(self, start: str | None = None,
                   end: str | None = None) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM requests"
        params: list = []
        clauses = []
        if start:
            clauses.append("ts >= ?")
            params.append(start)
        if end:
            clauses.append("ts <= ?")
            params.append(end)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return round(self._conn.execute(sql, params).fetchone()[0], 6)

    def daily_rollup(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT ts, COUNT(*), SUM(input_tokens), SUM(output_tokens),"
            " SUM(cost_usd), SUM(cached) FROM requests"
            " GROUP BY ts ORDER BY ts"
        ).fetchall()
        return [
            {"date": r[0], "requests": r[1], "input_tokens": r[2],
             "output_tokens": r[3], "cost_usd": round(r[4], 6),
             "cache_hits": r[5]}
            for r in rows
        ]

    def cost_by_model(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT model, COUNT(*), SUM(cost_usd) FROM requests"
            " GROUP BY model ORDER BY SUM(cost_usd) DESC"
        ).fetchall()
        return [{"model": r[0], "requests": r[1],
                 "cost_usd": round(r[2], 6)} for r in rows]

    def request_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    def budget_status(self, monthly_budget_usd: float,
                      warn_at_fraction: float = 0.8,
                      month: str | None = None) -> BudgetStatus:
        month = month or date.today().strftime("%Y-%m")
        spent = self.total_cost(start=f"{month}-01", end=f"{month}-31")
        fraction = spent / monthly_budget_usd if monthly_budget_usd else 0.0
        if fraction >= 1.0:
            alert = "exceeded"
        elif fraction >= warn_at_fraction:
            alert = "warn"
        else:
            alert = None
        return BudgetStatus(month=month, spent_usd=spent,
                            budget_usd=monthly_budget_usd,
                            fraction_used=round(fraction, 4), alert=alert)

    def close(self) -> None:
        self._conn.close()
