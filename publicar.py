#!/usr/bin/env python3
"""
Envia o estado do painel para onde ele é servido. Pode publicar em UM ou nos DOIS destinos:

  1) Vercel (função /api/estado + Redis):
        $env:PAINEL_URL   = "https://SEU-PROJETO.vercel.app/api/estado"
        $env:PAINEL_TOKEN = "senha-do-PUBLISH_TOKEN"
  2) Cloudflare Pages (arquivo estático estado.json: requisições ILIMITADAS no plano gratuito):
        $env:CF_PAGES_PROJETO = "nome-do-projeto-no-cloudflare"
        $env:CF_PAGES_PASTA   = "C:\\git\\eleicao2\\painel_cf"     (cópia da pasta do painel; o estado.json é gravado aqui)
     Precisa do Node e de "npx wrangler login" feito UMA vez neste PC.

Teste rápido:   python publicar.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

URL = os.environ.get("PAINEL_URL", "")
TOKEN = os.environ.get("PAINEL_TOKEN", "")
CF_PROJ = os.environ.get("CF_PAGES_PROJETO", "")
CF_DIR = os.environ.get("CF_PAGES_PASTA", "")
_ultimo_cf = [0.0]


def _vercel(corpo, tentativas=3):
    for i in range(tentativas):
        try:
            req = urllib.request.Request(URL, data=corpo, method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})
            with urllib.request.urlopen(req, timeout=15) as r:
                if r.status == 200:
                    return True
        except Exception as e:
            print(f"[publicar] Vercel, tentativa {i + 1} falhou: {e}")
            time.sleep(2 * (i + 1))
    return False


def _cloudflare(corpo, intervalo_min=90):
    """Grava estado.json e faz deploy no Cloudflare Pages (no máximo 1 a cada 90 s: o plano gratuito tem cota de deploys)."""
    if time.time() - _ultimo_cf[0] < intervalo_min:
        return None
    os.makedirs(CF_DIR, exist_ok=True)
    with open(os.path.join(CF_DIR, "estado.json"), "wb") as fh:
        fh.write(corpo)
    with open(os.path.join(CF_DIR, "_headers"), "w") as fh:     # navegador sempre revalida o estado.json
        fh.write("/estado.json\n  Cache-Control: public, max-age=0, must-revalidate\n")
    npx = shutil.which("npx") or shutil.which("npx.cmd") or "npx"
    cmd = [npx, "wrangler", "pages", "deploy", CF_DIR, "--project-name", CF_PROJ, "--branch", "main", "--commit-dirty=true"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, shell=(os.name == "nt"))
        ok = r.returncode == 0
        if not ok:
            print("[publicar] Cloudflare falhou:", (r.stderr or r.stdout)[-400:])
        _ultimo_cf[0] = time.time()
        return ok
    except Exception as e:
        print(f"[publicar] Cloudflare erro: {e}")
        return False


def publicar(estado):
    """Publica onde estiver configurado. Nunca derruba o modelo se algo falhar (só avisa)."""
    corpo = json.dumps(estado, ensure_ascii=False, default=str).encode("utf-8")
    res = []
    if URL and TOKEN:
        res.append(_vercel(corpo))
    if CF_PROJ and CF_DIR:
        r = _cloudflare(corpo)
        if r is not None:
            res.append(r)
    if not res and not (URL and TOKEN) and not (CF_PROJ and CF_DIR):
        print("[publicar] nenhum destino configurado (PAINEL_URL/PAINEL_TOKEN ou CF_PAGES_PROJETO/CF_PAGES_PASTA)")
        return False
    return all(res) if res else True


def ler():
    with urllib.request.urlopen(URL, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


if __name__ == "__main__":
    est = {"teste": True, "atualizado": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print("publicou:", publicar(est))
    if URL:
        print("lido de volta da Vercel:", ler())
    if CF_PROJ:
        print(f"confira: https://{CF_PROJ}.pages.dev/estado.json")
