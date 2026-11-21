"""Download IndicMaxxer-2B from your private Hugging Face repo into SLM2/model/IndicMaxxer-2B.
Usage:  .venv\\Scripts\\python.exe get_model.py YOUR_HF_USERNAME/IndicMaxxer-2B
(log in once first:  .venv\\Scripts\\python.exe -c "from huggingface_hub import login; login()")"""
import os, sys
from huggingface_hub import snapshot_download

repo = sys.argv[1] if len(sys.argv) > 1 else input("HF repo id (user/IndicMaxxer-2B): ").strip()
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "IndicMaxxer-2B")
snapshot_download(repo, local_dir=dst, ignore_patterns=["extras/*"])
print("model ready at", os.path.abspath(dst))
