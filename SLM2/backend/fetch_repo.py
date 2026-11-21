"""Plain resumable downloader for a Hugging Face repo (the Xet/hub downloader stalls on this network).
Usage: python fetch_repo.py <repo_id> <dest_dir> [--only model.safetensors,config.json,...]"""
import os, sys, time
import requests
from huggingface_hub import HfApi, get_token, hf_hub_url

repo, dest = sys.argv[1], sys.argv[2]
SKIP = (".msgpack", ".h5", ".ot", ".onnx", ".tflite", ".mlmodel")
info = HfApi().model_info(repo, files_metadata=True)
files = [s for s in info.siblings if not s.rfilename.endswith(SKIP) and not s.rfilename.startswith((".", "flax", "tf_"))]
has_st = any(s.rfilename.endswith(".safetensors") for s in files)
files = [s for s in files if not (has_st and s.rfilename.endswith(".bin"))]       # safetensors only when available
hdr = {"Authorization": f"Bearer {get_token()}"} if get_token() else {}
for s in files:
    path = os.path.join(dest, s.rfilename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    size = s.size or 0
    if os.path.exists(path) and os.path.getsize(path) == size:
        continue
    part = path + ".part"
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size and have >= size:
            break
        try:
            with requests.get(hf_hub_url(repo, s.rfilename), headers={**hdr, "Range": f"bytes={have}-"}, stream=True,
                              timeout=(15, 30)) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if not size:
                break
        except Exception as e:
            print("retry", s.rfilename, repr(e)[:100], flush=True); time.sleep(2)
    os.replace(part, path)
    print("got", s.rfilename, size // 2**20, "MB", flush=True)
print("done", repo, flush=True)
