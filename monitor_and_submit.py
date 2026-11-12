import time
import subprocess
import json
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

K_V23 = "sanjay0614/tf-v23-diurnal-physics-sota"
K_V24 = "sanjay0614/tf-v24-dual-expansion-sota"
COMP = "2026-ieee-big-data-traffic-flow-bench"

submitted_v23 = False
submitted_v24 = False
ref_v23 = None
ref_v24 = None

print("Starting monitor for Traffic Flow V23 & V24...", flush=True)

start_time = time.time()
while True:
    elapsed = int(time.time() - start_time)
    print(f"\n--- Elapsed: {elapsed}s ---", flush=True)
    
    # Check V23
    if not submitted_v23:
        try:
            st23 = api.kernels_status(K_V23)
            status23 = st23.get("status")
            print(f"V23 Status: {status23}", flush=True)
            if status23 == "COMPLETE":
                print(">>> V23 is COMPLETE! Submitting to competition...", flush=True)
                cmd = [
                    sys.executable, "-m", "kaggle", "competitions", "submit",
                    COMP, "-k", K_V23, "-f", "submission.csv",
                    "-m", "v23: Diurnal Density Conservation SOTA (w=0.5, alpha=0.5, run<=2)"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, env=dict(**subprocess.os.environ, PYTHONUTF8="1"))
                print("Submission output:", res.stdout, flush=True)
                if res.stderr:
                    print("Submission stderr:", res.stderr, flush=True)
                submitted_v23 = True
            elif status23 == "ERROR":
                print(">>> V23 FAILED! Error message:", st23.get("failureMessage"), flush=True)
                try:
                    logs = api.kernels_logs(K_V23)
                    print("V23 Logs:\n", str(logs)[-1500:], flush=True)
                except Exception as e:
                    print("Could not fetch logs:", e, flush=True)
                submitted_v23 = True # stop trying
        except Exception as e:
            print(f"Error querying V23 status: {e}", flush=True)

    # Check V24
    if not submitted_v24:
        try:
            st24 = api.kernels_status(K_V24)
            status24 = st24.get("status")
            print(f"V24 Status: {status24}", flush=True)
            if status24 == "COMPLETE":
                print(">>> V24 is COMPLETE! Submitting to competition...", flush=True)
                cmd = [
                    sys.executable, "-m", "kaggle", "competitions", "submit",
                    COMP, "-k", K_V24, "-f", "submission.csv",
                    "-m", "v24: Multi-Axis Physics Expansion (w=0.75, alpha=0.78, run<=3) + Causal-T Queue"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, env=dict(**subprocess.os.environ, PYTHONUTF8="1"))
                print("Submission output:", res.stdout, flush=True)
                if res.stderr:
                    print("Submission stderr:", res.stderr, flush=True)
                submitted_v24 = True
            elif status24 == "ERROR":
                print(">>> V24 FAILED! Error message:", st24.get("failureMessage"), flush=True)
                try:
                    logs = api.kernels_logs(K_V24)
                    print("V24 Logs:\n", str(logs)[-1500:], flush=True)
                except Exception as e:
                    print("Could not fetch logs:", e, flush=True)
                submitted_v24 = True # stop trying
        except Exception as e:
            print(f"Error querying V24 status: {e}", flush=True)

    if submitted_v23 and submitted_v24:
        print("\nBoth V23 and V24 have been handled! Checking recent competition submissions...", flush=True)
        subs = api.competition_submissions(COMP)
        for s in subs[:6]:
            print(f"Ref: {s.ref}, Status: {s.status}, Score: {s.public_score}, Date: {s.date}, Desc: {s.description[:60]}", flush=True)
        break

    time.sleep(15)
