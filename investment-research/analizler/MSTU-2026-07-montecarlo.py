"""MSTU Monte Carlo: 2x günlük kaldıraçlı ETF'nin 1 ay / 1 yıl dağılımı.

Model: MSTR günlük log-getirisi ~ Normal(mu_d, sigma_d) (GBM varsayımı).
MSTU günlük getirisi = 2 * MSTR_getiri - günlük maliyet.
Maliyet: %1,05 ücret + ~%4,5 swap finansmanı ≈ yıllık %5,5 (VARSAYIM).
NOT: Gerçek dağılım şişman kuyrukludur; normal varsayım kuyruk risklerini AZIMSAR.
"""
import random, math, statistics

random.seed(42)
DAYS_M, DAYS_Y = 21, 252
N = 20000
COST_ANNUAL = 0.055
cost_d = COST_ANNUAL / 252

# (etiket, MSTR yıllık aritmetik drift, MSTR yıllık volatilite)
scenarios = [
    ("Ayı    (MSTR -%40/yıl, vol %100)", -0.40, 1.00),
    ("Yatay  (MSTR  %0/yıl,  vol %90)",  0.00, 0.90),
    ("Boğa   (MSTR +%60/yıl, vol %90)",  0.60, 0.90),
    ("Süper boğa (MSTR +%150/yıl, vol %110)", 1.50, 1.10),
]

def simulate(drift, vol, days):
    mu_d = drift / 252            # aritmetik günlük beklenti
    sig_d = vol / math.sqrt(252)
    out_etf, out_stk = [], []
    for _ in range(N):
        etf, stk = 1.0, 1.0
        for _ in range(days):
            r = random.gauss(mu_d, sig_d)      # günlük basit getiri yaklaşımı
            r = max(r, -0.95)                   # tek günde -%95 alt sınır
            stk *= (1 + r)
            etf *= (1 + 2 * r - cost_d)
            if etf <= 0:
                etf = 0.0
                # kalan günleri atla (tam kayıp)
                break
        out_etf.append(etf); out_stk.append(stk)
    return out_etf, out_stk

def pct(xs, p):
    s = sorted(xs); return s[int(p * (len(s) - 1))]

def report(days, label_days):
    print(f"\n===== UFUK: {label_days} ({days} işlem günü), {N} yol =====")
    hdr = f"{'Senaryo':38} | {'MSTR medyan':>11} | {'MSTU medyan':>11} | {'P(MSTU>0)':>9} | {'P(<-50%)':>8} | {'P(<-90%)':>8}"
    print(hdr); print("-" * len(hdr))
    for label, dr, vol in scenarios:
        etf, stk = simulate(dr, vol, days)
        med_e = statistics.median(etf) - 1
        med_s = statistics.median(stk) - 1
        p_up = sum(1 for x in etf if x > 1) / N
        p_50 = sum(1 for x in etf if x < 0.5) / N
        p_90 = sum(1 for x in etf if x < 0.1) / N
        print(f"{label:38} | {med_s:>10.1%} | {med_e:>10.1%} | {p_up:>8.1%} | {p_50:>7.1%} | {p_90:>7.1%}")

report(DAYS_M, "1 AY")
report(DAYS_Y, "1 YIL")
print("\nNot: Normal dağılım varsayımı; gerçek kuyruklar daha şişman → kötü uçlar azımsanmıştır.")
