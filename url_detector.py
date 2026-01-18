import joblib
import re
from urllib.parse import urlparse, parse_qs
import os

# -------------------------------
# CONFIGURARE SI CONSTANTE
# -------------------------------
PHISHING_KEYWORDS = [
    "login", "signin", "verify", "verification", "secure", "account", "update",
    "password", "reset", "confirm", "wallet", "billing", "bank", "authorize",
    "security", "support", "invoice", "refund", "token", "session", "2fa"
]

RISKY_TLDS = {
    "xyz", "zip", "click", "top", "icu", "cam", "cfd", "sbs", "rest", "fit", "shop", "mov",
    "ml", "tk", "cf", "ga", "gq"
}

BRAND_KEYWORDS = {
    "paypal", "google", "facebook", "instagram", "apple", "microsoft", "netflix",
    "amazon", "revolut", "ing", "bt", "bcr"
}

HOMOGLYPH_HINT = re.compile(r"[^\x00-\x7F]")
IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

# -------------------------------
# INCARCARE MODEL
# -------------------------------
print("https://www.merriam-webster.com/dictionary/module Incarc modelul URL...")
model = None
try:
    if os.path.exists("rf_url_model.joblib"):
        model = joblib.load("rf_url_model.joblib")
        print("https://www.merriam-webster.com/dictionary/module Model incarcat cu succes!")
    else:
        print("https://www.merriam-webster.com/dictionary/module EROARE: Fisierul 'rf_url_model.joblib' lipseste!")
except Exception as e:
    print(f"https://www.merriam-webster.com/dictionary/module EROARE CRITICA la incarcare model: {e}")


# -------------------------------
# FUNCTII AUXILIARE
# -------------------------------
def _looks_like_ip(host: str) -> bool:
    return bool(IPV4_RE.match(host or ""))


def _subdomain_count(host: str) -> int:
    parts = [p for p in (host or "").split(".") if p]
    return max(0, len(parts) - 2)


def _count_keywords(s: str) -> int:
    s = (s or "").lower()
    return sum(1 for k in PHISHING_KEYWORDS if k in s)


def _add_signal(signals, name: str, weight: int, text: str):
    signals.append({"name": name, "weight": int(weight), "text": text})


def extract_features(url: str):
    """Extrage trasaturile pentru modelul ML si metadatele."""
    u = urlparse(url)
    host = (u.hostname or "").lower()
    qs = parse_qs(u.query)

    tld = host.split(".")[-1] if "." in host else host
    url_len = len(url)
    is_http = int((u.scheme or "").lower() != "https")
    sub = _subdomain_count(host)
    is_ip = int(_looks_like_ip(host))
    param_count = len(qs.keys())
    kw = _count_keywords(url)
    risky_tld = int(tld in RISKY_TLDS)
    puny = int("xn--" in host)
    hyphens = host.count("-")
    digits = sum(ch.isdigit() for ch in host)

    # Vectorul exact pe care a fost antrenat Random Forest
    features = [
        url_len, is_http, sub, is_ip, param_count,
        kw, risky_tld, puny, hyphens, digits,
    ]

    meta = {
        "host": host, "tld": tld, "url_len": url_len,
        "https": int((u.scheme or "").lower() == "https"),
        "subdomains": sub, "ip_in_host": bool(is_ip),
        "param_count": param_count, "keyword_hits": kw,
        "risky_tld": bool(risky_tld), "punycode": bool(puny),
        "hyphens": hyphens, "digits": digits,
        "netloc": (u.netloc or ""), "query_len": len(u.query or ""),
    }
    return features, meta


def build_signals(url: str, meta: dict):
    """Reguli explicabile (Rule-based system)."""
    signals = []
    score = 0

    host = (meta.get("host") or "").lower()
    tld = (meta.get("tld") or "").lower()
    netloc = meta.get("netloc") or ""

    # HTTPS
    if not meta.get("https", 1):
        score += 15

    # IP in host
    if meta.get("ip_in_host"):
        score += 20

    # '@' phishing trick
    if netloc and "@" in netloc:
        score += 20

    # Punycode / Unicode
    if meta.get("punycode") or HOMOGLYPH_HINT.search(host):
        score += 18

    # Risky TLD
    if tld in RISKY_TLDS:
        score += 12

    # Brand spoofing heuristic
    brand = next((b for b in BRAND_KEYWORDS if b in host), None)
    if brand and (tld in RISKY_TLDS):
        score = max(score, 70)

    # Subdomains
    sub = int(meta.get("subdomains") or 0)
    if sub >= 3:
        score += 12
    elif sub == 2:
        score += 6

    # Lengths
    if (meta.get("url_len") or 0) > 140:
        score += 12
    elif (meta.get("url_len") or 0) > 95:
        score += 8

    # Params
    if (meta.get("param_count") or 0) >= 10:
        score += 10
    elif (meta.get("param_count") or 0) >= 6:
        score += 6

    # Phishing keywords
    hits = int(meta.get("keyword_hits") or 0)
    if hits >= 4:
        score += 18
    elif hits >= 2:
        score += 10
    elif hits == 1:
        score += 5

    # Hyphens / digits
    if (meta.get("hyphens") or 0) >= 3: score += 6
    if (meta.get("digits") or 0) >= 6: score += 6

    return min(100, score)


# -------------------------------
# FUNCTIA PRINCIPALA (CALL ENTRY POINT)
# -------------------------------
def analyze_url(url: str) -> int:
    """
    Primeste un URL (string) si returneaza un scor de risc (0-100).
    """
    if not url:
        return 0

    url = url.strip()

    # 1. Extrage trasaturi
    feats, meta = extract_features(url)

    # 2. Scor ML (daca modelul e incarcat)
    ml_score = 0
    if model:
        try:
            prob = float(model.predict_proba([feats])[0][1])
            ml_score = int(round(prob * 100))
        except Exception as e:
            print(f"https://www.merriam-webster.com/dictionary/module Eroare la predictie ML: {e}")

    # 3. Scor bazat pe Reguli
    rule_score = build_signals(url, meta)

    # 4. Scor final (cel mai mare dintre cele doua)
    final_score = max(ml_score, rule_score)

    return final_score