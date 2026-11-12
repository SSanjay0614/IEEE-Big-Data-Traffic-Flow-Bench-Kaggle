# TF V23 — Diurnal Density Conservation SOTA (w=0.5, alpha=0.5, run<=2) + Exact-KKT ODME

Kaggle Kernel: [`sanjay0614/tf-v23-diurnal-physics-sota`](https://www.kaggle.com/code/sanjay0614/tf-v23-diurnal-physics-sota)  
Competition: [2026 IEEE Big Data — Traffic Flow Bench](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench)  
Hardware: 4-Core Kaggle CPU (`enable_gpu: false`), **0.00h GPU Quota Consumed**

---

## 1. Why V23 Exists

Our baseline V8 scored **0.86515**, with physics ($S_{\text{physics}} = 0.6720$) being heavily limited by $S_{\text{LWR}} = 0.514$.
In V23, we isolate and implement the clean, proven Hong Hanh diurnal density conservation operator directly on top of the rock-solid V8 baseline:
- Fixes the bug in earlier V11 (which had suffered from cross-day diurnal boundary corruption and the broken V10 spatial closing).
- Restricts vehicle accumulation $N = (q/v) \cdot L$ interpolation strictly within the same 288-slot calendar day (`rows // 288 == pp // 288`).
- Updates $(v, q)$ simultaneously along the characteristic curve:
  $$d = w \cdot (\ln(k_i) - \ln(k_{\text{hat}})) \quad \text{with } w = 0.5, \alpha = 0.5, \text{run} \le 2$$
  $$q_{\text{new}} = q_{\text{hat}} \cdot \exp(\alpha \cdot d), \quad v_{\text{new}} = v_{\text{hat}} \cdot \exp(-(1 - \alpha) \cdot d)$$

Target: **0.8714+ LB** (mirroring the official Hong Hanh public benchmark).

---

## 2. Component Pipeline

1. **Task 1 & 3 (State + Physics):**
   - CatBoost baseline models (`speed_d8`, `speed_d10`, `flow_d8`, `flow_d10`).
   - Diurnal density conservation operator ($w=0.5, \alpha=0.5, \text{run} \le 2$).
2. **Task 2 (Queue):**
   - Step-6 onset (0.5 LGBM + 0.5 CatBoost $\ge 0.55$) with top-1 pinch-point fallback.
   - Ongoing queue threshold: 0.40.
3. **Task 4 (ODME):**
   - High-precision exact-KKT solve with $\lambda = 0.05$ ($S_{\text{ODME}} \approx 0.994$).
4. **Compute:** 100% CPU inference (~8.5 minutes, 0.00h GPU burned).
