import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline


def wav(url, wav_path):
    base = os.path.splitext(wav_path)[0]
    template = base + ".%(ext)s"
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-o", template,
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args",
        "ffmpeg:-ac 1 -ar 16000 -c:a pcm_s16le",
        url,
    ]
    subprocess.check_call(cmd)
    produced = base + ".wav"
    if produced != wav_path:
        os.replace(produced, wav_path)


def citeste_audio(wav_path):
    a, sr = sf.read(wav_path, always_2d=False)
    if getattr(a, "ndim", 1) > 1:
        a = np.mean(a, axis=1)
    t = torch.tensor(a, dtype=torch.float32)
    if sr != 16000:
        import torchaudio
        r = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        t = r(t)
    return t.cpu().numpy()


def talk(x, prag=0.55, ms_v=250, ms_p=150):
    from silero_vad import get_speech_timestamps, load_silero_vad
    m = load_silero_vad()
    w = torch.tensor(x, dtype=torch.float32)
    ts = get_speech_timestamps(
        w, m,
        sampling_rate=16000,
        threshold=prag,
        min_speech_duration_ms=ms_v,
        min_silence_duration_ms=ms_p,
    )
    if not ts:
        return np.array([], dtype=np.float32)
    buc = []
    for t in ts:
        buc.append(x[t["start"]:t["end"]])
    return np.concatenate(buc) if buc else np.array([], dtype=np.float32)


def ferestre(x, sec=6.0, pas=3.0, minp=0.85):
    sr = 16000
    n = int(sec * sr)
    h = int(pas * sr)
    if len(x) <= n:
        return [x] if len(x) else []
    out = []
    for i in range(0, len(x), h):
        j = i + n
        c = x[i:j]
        if len(c) / n < minp:
            break
        out.append(c)
        if j >= len(x):
            break
    return out


def label_ai(labels):
    for lab in labels:
        s = lab.lower()
        if "ai" in s or "fake" in s or "spoof" in s or "synth" in s or "deepfake" in s or "generated" in s:
            return lab
    return None


def scoruri(pred, lab):
    out = []
    for p in pred:
        s = 0.0
        for r in p:
            if r["label"] == lab:
                s = float(r["score"])
                break
        out.append(s)
    return out


def trimmed_mean(v, t=0.15):
    if not v:
        return 0.0
    a = np.array(v, dtype=np.float32)
    a.sort()
    n = len(a)
    k = int(n * t)
    if n - 2 * k <= 0:
        return float(a.mean())
    return float(a[k:n - k].mean())


def prag(v, p=0.75):
    if not v:
        return 0.0
    return float(np.mean([1.0 if x >= p else 0.0 for x in v]))


def sig(z):
    z = float(z)
    if z >= 0:
        e = np.exp(-z)
        return float(1.0 / (1.0 + e))
    e = np.exp(z)
    return float(e / (1.0 + e))


def prob(df_v, fr_v, df_f, fr_f):
    z = (
        4.0 * (df_v - 0.62)
        + 2.0 * (df_f - 0.58)
        + 1.8 * (fr_v - 0.25)
        + 1.2 * (fr_f - 0.20)
    )
    return sig(z)


def verifica(url):
    model = "Hemgg/Deepfake-audio-detection"
    dev = 0 if torch.cuda.is_available() else -1
    clf = pipeline("audio-classification", model=model, device=dev)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "a.wav")
        wav(url, wav_path)
        x = citeste_audio(wav_path)

        v = talk(x, prag=0.55, ms_v=250, ms_p=150)

        full = ferestre(x, sec=6.0, pas=3.0, minp=0.85)
        only = ferestre(v, sec=6.0, pas=3.0, minp=0.85) if len(v) else []

        pr_full = [clf({"array": c, "sampling_rate": 16000}, top_k=None) for c in full] if full else []
        pr_only = [clf({"array": c, "sampling_rate": 16000}, top_k=None) for c in only] if only else []

    if not pr_full and not pr_only:
        return "NESIGUR", 0.5

    labels = [p["label"] for p in (pr_full[0] if pr_full else pr_only[0])]
    aiet = label_ai(labels)
    if aiet is None:
        return "NESIGUR", 0.5

    s_full = scoruri(pr_full, aiet) if pr_full else []
    s_only = scoruri(pr_only, aiet) if pr_only else []

    df_v = trimmed_mean(s_only, t=0.15)
    df_f = trimmed_mean(s_full, t=0.15)

    fr_v = prag(s_only, p=0.75)
    fr_f = prag(s_full, p=0.75)

    p = prob(df_v, fr_v, df_f, fr_f)
    verdict = "AI" if p >= 0.5 else "NU"

    return verdict, p


if __name__ == "__main__":
    url = "https://youtube.com/shorts/6k64EPE2Mk8?si=9PP4b6KK2mKeRAPe"
    v, p = verifica(url)
    print(f"Verdict: {v}")
    print(f"Probabilitate: {p*100:.1f}% sa fie DeepFake")
