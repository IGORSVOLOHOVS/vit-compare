import os

# huggingface_hub reads this once, on import, and nothing has imported it yet: conftest.py comes
# first. A test that reaches for the SigLIP 2 weights then fails on the spot instead of fetching
# them, however pytest was started.
os.environ["HF_HUB_OFFLINE"] = "1"
