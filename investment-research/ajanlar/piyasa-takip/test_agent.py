"""Ağ GEREKTİRMEYEN birim testleri (tasarım §6 — sandbox'ta dış ağ kapalı).

Çalıştırma: python3 -m unittest test_agent -v
Kapsam: kaynak çözücüler, değişim/eşik hesabı, bayatlık, rapor üretimi.
Canlı uçtan uca test (gerçek HTTP) kullanıcı ortamında yapılır.
"""

import unittest
from datetime import datetime, timezone

import agent
from agent import Reading

STOOQ_OK = (
    "Symbol,Date,Time,Open,High,Low,Close,Volume\n"
    "^SPX,2026-07-03,22:00:07,6100.5,6150.2,6080.1,6123.45,0\n"
)
STOOQ_BAD_SYMBOL = (
    "Symbol,Date,Time,Open,High,Low,Close,Volume\n"
    "XYZ,N/D,N/D,N/D,N/D,N/D,N/D,N/D\n"
)
COINGECKO_OK = '{"bitcoin": {"usd": 62167.75}, "ethereum": {"usd": 2400.5}}'
FRED_OK = "DATE,DFF\n2026-07-01,3.58\n2026-07-02,.\n2026-07-03,3.60\n"
FRED_EMPTY = "DATE,DFF\n2026-07-01,.\n"


class ParserTests(unittest.TestCase):
    def test_stooq_ok(self):
        r = agent.parse_stooq_csv(STOOQ_OK, "^spx", "S&P 500", 2.0)
        self.assertEqual(r.value, 6123.45)
        self.assertEqual(r.asof, "2026-07-03")
        self.assertEqual(r.error, "")

    def test_stooq_unknown_symbol_returns_veri_yok(self):
        r = agent.parse_stooq_csv(STOOQ_BAD_SYMBOL, "xyz", "Bilinmez", 2.0)
        self.assertIsNone(r.value)
        self.assertIn("sembol", r.error)

    def test_coingecko_ok(self):
        rs = agent.parse_coingecko_json(COINGECKO_OK)
        by_key = {r.key: r for r in rs}
        self.assertAlmostEqual(by_key["bitcoin"].value, 62167.75)
        self.assertAlmostEqual(by_key["ethereum"].value, 2400.5)

    def test_fred_skips_missing_and_takes_last_valid(self):
        r = agent.parse_fred_csv(FRED_OK, "DFF", "Fed funds", 10.0)
        self.assertEqual(r.value, 3.60)
        self.assertEqual(r.asof, "2026-07-03")
        self.assertTrue(r.is_rate)

    def test_fred_all_missing(self):
        r = agent.parse_fred_csv(FRED_EMPTY, "DFF", "Fed funds", 10.0)
        self.assertIsNone(r.value)


class ChangeTests(unittest.TestCase):
    def _state(self, key, value, asof):
        return {key: {"value": value, "asof": asof}}

    def test_percent_change_and_breach(self):
        r = Reading("^spx", "S&P 500", 100.0, "2026-07-03", "stooq", 2.0)
        delta, breached = agent.compute_change(r, self._state("^spx", 97.0, "2026-07-02"))
        self.assertAlmostEqual(delta, 3.0928, places=3)
        self.assertTrue(breached)

    def test_below_threshold_no_alert(self):
        r = Reading("^spx", "S&P 500", 100.0, "2026-07-03", "stooq", 2.0)
        delta, breached = agent.compute_change(r, self._state("^spx", 99.0, "2026-07-02"))
        self.assertFalse(breached)

    def test_rate_change_in_basis_points(self):
        r = Reading("DGS10", "ABD 10Y", 4.55, "2026-07-03", "fred", 10.0, is_rate=True)
        delta, breached = agent.compute_change(r, self._state("DGS10", 4.40, "2026-07-02"))
        self.assertAlmostEqual(delta, 15.0, places=6)
        self.assertTrue(breached)

    def test_same_asof_means_no_new_observation(self):
        r = Reading("^spx", "S&P 500", 100.0, "2026-07-03", "stooq", 2.0)
        delta, breached = agent.compute_change(r, self._state("^spx", 90.0, "2026-07-03"))
        self.assertIsNone(delta)
        self.assertFalse(breached)

    def test_no_previous_state(self):
        r = Reading("^spx", "S&P 500", 100.0, "2026-07-03", "stooq", 2.0)
        delta, breached = agent.compute_change(r, {})
        self.assertIsNone(delta)
        self.assertFalse(breached)


class StaleAndReportTests(unittest.TestCase):
    NOW = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    def test_stale_detection(self):
        old = Reading("xauusd", "Altın", 3300.0, "2026-06-25", "stooq", 3.0)
        fresh = Reading("^spx", "S&P 500", 6123.0, "2026-07-03", "stooq", 2.0)
        self.assertTrue(agent.is_stale(old, self.NOW))
        self.assertFalse(agent.is_stale(fresh, self.NOW))

    def test_report_contains_alert_table_and_missing_data(self):
        readings = [
            Reading("^spx", "S&P 500", 100.0, "2026-07-03", "stooq", 2.0),
            Reading("usdtry", "USD/TRY", None, "", "stooq", 1.5, error="timeout"),
        ]
        state = {"^spx": {"value": 95.0, "asof": "2026-07-02"}}
        report = agent.build_report(readings, state, now=self.NOW)
        self.assertIn("⚠️ DİKKAT", report)          # %5.26 > %2 eşiği
        self.assertIn("| S&P 500 | 100.00 |", report)
        self.assertIn("VERİ YOK", report)
        self.assertIn("timeout", report)
        self.assertIn("yatırım tavsiyesi değildir", report)

    def test_report_without_alerts_has_no_alert_section(self):
        readings = [Reading("^spx", "S&P 500", 100.0, "2026-07-03", "stooq", 2.0)]
        report = agent.build_report(readings, {}, now=self.NOW)
        self.assertNotIn("DİKKAT", report)


if __name__ == "__main__":
    unittest.main()
