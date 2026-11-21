# IndicMaxxer-2B app (local)

```
SLM2/
  model/IndicMaxxer-2B/   merged model (download with backend/get_model.py)
  backend/                FastAPI: streaming chat + tools + /reasoning, MMS-TTS voices  (port 8000)
  web/                    Next.js + Tailwind: landing, /chat, /voice orb               (port 3000)
```

## One-time setup
```
cd D:\IndicMaxxer\SLM2\backend
.venv\Scripts\python.exe -c "from huggingface_hub import login; login()"
.venv\Scripts\python.exe get_model.py YOUR_HF_USERNAME/IndicMaxxer-2B
```

## Run (two terminals)
```
cd D:\IndicMaxxer\SLM2\backend
.venv\Scripts\python.exe -m uvicorn server:app --port 8000
```
```
cd D:\IndicMaxxer\SLM2\web
npm run dev
```
Open http://localhost:3000 in Chrome or Edge (the voice page uses the browser's speech recognition).

- The model loads in 4-bit NF4 (~2 GB VRAM) so it fits the RTX 3050 4 GB. `set IMX_4BIT=0` for fp16 on a bigger GPU.
- Voices: Meta MMS-TTS (`facebook/mms-tts-kan`, `-hin`, `-eng`), downloaded on first use, run on CPU.
- OCR needs Tesseract installed with Kannada + Hindi data; without it, image uploads say "ocr failed" and chat continues.
