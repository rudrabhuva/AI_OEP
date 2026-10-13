"""
main.py — Project entry-point
==============================
1. Trains the ANN model by calling ann_project.train_and_evaluate().
2. Launches the Streamlit dashboard (ui.py) automatically when training
   is finished.

Usage
-----
    python main.py
"""

import subprocess
import sys
import os

# ── Make sure we can import ann_project from the same directory ───────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ann_project import train_and_evaluate, MODEL_PATH


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Student Productivity ANN — Project Launcher        ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Step 1: Train the ANN ─────────────────────────────────────────────────
    print("► Step 1 / 2 — Training the ANN model …\n")
    metrics = train_and_evaluate()

    print("\n── Training Summary ─────────────────────────────────────")
    print(f"  R² Score         : {metrics['r2']:.4f}  ({metrics['r2_pct']:.2f} %)")
    print(f"  Mean CV R²       : {metrics['cv_mean']:.4f}")
    print(f"  Target outliers  : {metrics['outliers']}")
    print(f"  Model saved at   : {metrics['model_path']}")
    print(f"  Heatmap          : {metrics['heatmap_path']}")
    print(f"  Actual vs Pred.  : {metrics['avp_path']}")
    print("─────────────────────────────────────────────────────────\n")

    # ── Step 2: Launch Streamlit UI ───────────────────────────────────────────
    ui_path = os.path.join(BASE_DIR, "ui.py")

    print("► Step 2 / 2 — Launching Streamlit dashboard …")
    print(f"  Running: streamlit run \"{ui_path}\"\n")
    print("  Press Ctrl+C to stop the server.\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", ui_path],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n  Server stopped by user.")
    except subprocess.CalledProcessError as exc:
        print(f"\n  ✗ Streamlit exited with error code {exc.returncode}.")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
