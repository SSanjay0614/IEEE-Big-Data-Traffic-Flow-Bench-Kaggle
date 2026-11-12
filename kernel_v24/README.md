# TF V24 — Multi-Axis Physics Expansion (w=0.75, alpha=0.78, run<=3) + Continuous Causal-T Queue

Kaggle Kernel: [`sanjay0614/tf-v24-dual-expansion-sota`](https://www.kaggle.com/code/sanjay0614/tf-v24-dual-expansion-sota)  
Competition: [2026 IEEE Big Data — Traffic Flow Bench](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench)  
Hardware: 4-Core Kaggle CPU (`enable_gpu: false`), **0.00h GPU Quota Consumed**

---

## 1. Why V24 Exists

V24 pushes both Task 3 (Physics) and Task 2 (Queue) beyond the conservative V23 benchmark:
1. **Deeper Physics Operator ($w = 0.75, \alpha = 0.78, \text{run} \le 3$):**
   - Pulls density closer to the true physical accumulation $N_i$ across up to 15-minute gaps (3 slots).
   - Adds the Triangular FD Upper-Envelope capacity guardrail: softly bounds congested flow overshoots ($v < 0.58 v_f$) to $q_{\text{fd\_max}}$, preserving $S_{\text{FD}} = 0.988+$.
2. **Continuous Causal-T Queue Probability:**
   - Instead of narrow, discrete boolean promotion/clearance masks, V24 modulates ongoing probabilities continuously:
     $$p_{\text{causal}} = 0.65 \cdot p_{\text{std}} + 0.35 \cdot p_T$$
   - Decision threshold: $p_{\text{causal}} \ge 0.40$. Links observed congested at origin $T$ naturally gain higher probability across the early horizon without discrete artifacts.
   - Dual-tree onset: evaluates `0.5 * onset_lgb + 0.5 * onset_cat >= 0.55` on step 6 ($T+30$) with top-1 fallback.
3. **Exact-KKT ODME:** $\lambda = 0.05$ ($S_{\text{ODME}} \approx 0.994$).

Target: **0.875+ / 0.88+ LB**.

---

## 2. Resource & Execution Profile
- **Hardware:** 4 CPU Cores, 30GB RAM
- **GPU Quota:** 0.00h (CPU only)
- **Estimated Wall Time:** ~8.5 minutes
