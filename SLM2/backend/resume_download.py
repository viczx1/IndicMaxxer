"""Resume model.safetensors from the partial .incomplete file using HTTP Range requests (retries on stalls)."""
import glob, os, shutil, sys, time
import requests
from huggingface_hub import get_token, hf_hub_url

REPO, FN = "viczx/IndicMaxxer-2B", "model.safetensors"
dst_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "IndicMaxxer-2B")
part = os.path.join(dst_dir, FN + ".part")
final = os.path.join(dst_dir, FN)
if not os.path.exists(part):
    cands = sorted(glob.glob(os.path.join(dst_dir, ".cache", "huggingface", "download", "*.incomplete")), key=os.path.getsize)
    if cands and os.path.getsize(cands[-1]) > 0:
        shutil.move(cands[-1], part)
        print("resuming from", os.path.getsize(part) // 2**20, "MB", flush=True)
    else:
        open(part, "wb").close()
url, hdr = hf_hub_url(REPO, FN), {"Authorization": f"Bearer {get_token()}"}
total = int(requests.head(url, headers=hdr, allow_redirects=True, timeout=30).headers["Content-Length"])
print("total", total // 2**20, "MB", flush=True)
while os.path.getsize(part) < total:
    have = os.path.getsize(part)
    try:
        with requests.get(url, headers={**hdr, "Range": f"bytes={have}-"}, stream=True, timeout=(15, 30)) as r:
            r.raise_for_status()
            with open(part, "ab") as f:
                last = time.time()
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    if time.time() - last > 5:
                        print(f"{os.path.getsize(part) / total * 100:.1f}%  {os.path.getsize(part) // 2**20} MB", flush=True)
                        last = time.time()
    except Exception as e:                                  # stall / reset: reconnect and continue
        print("reconnecting:", repr(e)[:120], flush=True)
        time.sleep(2)
os.replace(part, final)
print("done:", final, os.path.getsize(final) // 2**20, "MB", flush=True)
