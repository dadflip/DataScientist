import sys, os, pathlib, json, shutil
_repo_root = str(pathlib.Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import pandas as pd
import warnings
import subprocess
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.4f}'.format)
print("Environment ready.")
print("-" * 50)
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True
    )
    gpus = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    if gpus:
        for idx, gpu in enumerate(gpus):
            print(f"GPU {idx} detected: {gpu}")
    else:
        print("No GPU detected (running on CPU).")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("No GPU detected (running on CPU).")
print("-" * 50)


class PipelineState:
    def __init__(self):
        self.data_raw         = {}
        self.data_cleaned     = {}
        self.data_encoded     = {}
        self.data_splits      = {}
        self.data_types       = {}
        self.meta             = {}
        self.business_context = {}
        self.models           = {}
        self.history          = []
        self.config           = {}
        self.eda_dashboard    = []  # persists PlotDashboard entries across reruns

    def log_step(self, step: str, action: str, details: dict):
        self.history.append({"step": step, "action": action, "details": details})

    def reset(self):
        """Wipe all pipeline state in-place (keeps the same object reference)."""
        self.data_raw         = {}
        self.data_cleaned     = {}
        self.data_encoded     = {}
        self.data_splits      = {}
        self.data_types       = {}
        self.meta             = {}
        self.business_context = {}
        self.models           = {}
        self.history          = []
        self.config           = {}
        self.eda_dashboard    = []

    def summary(self):
        print("=== PipelineState Summary ===")
        print(f"  Raw datasets    : {list(self.data_raw.keys())}")
        print(f"  Cleaned         : {list(self.data_cleaned.keys())}")
        print(f"  Encoded         : {list(self.data_encoded.keys())}")
        print(f"  Splits          : {list(self.data_splits.keys())}")
        print(f"  Models          : {list(self.models.keys())}")
        print(f"  History steps   : {len(self.history)}")
        print(f"  Dashboard plots : {len(self.eda_dashboard)}")

    def load_config(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.log_step("config", "loaded", {"path": config_path})

    def clean_temp_files(self, temp_dir: str = "temp"):
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        self.log_step("cleanup", "temp_files_removed", {"directory": temp_dir})


state = PipelineState()
print("State variables (state) initialised. Waiting for configuration…")
print(f"  state.eda_dashboard ready ({len(state.eda_dashboard)} plots stored).")
print("-" * 50)

# ── Reset widget ────────────────────────────────────────────────────────────
_reset_btn = widgets.Button(
    description="⟳ Reset All State",
    button_style="danger",
    layout=widgets.Layout(width="auto", height="32px")
)
_reset_msg = widgets.HTML("")

def _on_reset(b):
    state.reset()
    _reset_msg.value = (
        "<span style='color:#b45309; background:#fef3c7; border-left:4px solid #f59e0b;"
        "padding:4px 10px; font-size:0.85em; border-radius:4px;'>"
        "All state reset. Reload data & config to continue.</span>"
    )

_reset_btn.on_click(_on_reset)
display(widgets.HBox(
    [_reset_btn, _reset_msg],
    layout=widgets.Layout(align_items="center", gap="10px", margin="8px 0 0 0")
))