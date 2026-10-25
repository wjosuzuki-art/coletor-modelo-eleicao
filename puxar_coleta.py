#!/usr/bin/env python3
"""
No PC CENTRAL (casa): busca no site o que os PCs remotos enviaram e grava na pasta da coleta, onde o ao_vivo.py lê.

    $env:COLETA_TOKEN = "senha-da-coleta"
    python puxar_coleta.py https://SEU-PROJETO.vercel.app/api/coleta --saida coleta            # domingo (pleito 3221)
    python puxar_coleta.py https://SEU-PROJETO.vercel.app/api/coleta --saida coleta --ensaio   # teste com o 1º turno

Grava coleta/coleta_p<pleito>_remoto_<parte>.jsonl (o ao_vivo.py junta tudo sozinho). Pode parar e reabrir: continua de onde parou.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime


def get(url, tok):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:          # mostra o motivo que o site devolveu (ex.: senha diferente / falta redeploy)
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}") from None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--saida", default="coleta")
    ap.add_argument("--ensaio", action="store_true")
    ap.add_argument("--intervalo", type=int, default=30)
    a = ap.parse_args()
    tok = os.environ.get("COLETA_TOKEN", "").strip()
    if not tok:
        sys.exit("Defina $env:COLETA_TOKEN antes.")
    pleito = 3220 if a.ensaio else 3221
    os.makedirs(a.saida, exist_ok=True)
    while True:
        try:
            partes = get(a.url, tok).get("partes", {})
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] erro ao listar: {e}"); time.sleep(a.intervalo); continue
        resumo = []
        for parte, total in sorted(partes.items()):
            if not parte.startswith(f"p{pleito}_"):
                continue
            arq = os.path.join(a.saida, f"coleta_p{pleito}_remoto_{parte.split('_', 1)[1]}.jsonl")
            tem = sum(1 for _ in open(arq, encoding="utf-8")) if os.path.exists(arq) else 0
            while tem < total:
                q = urllib.parse.urlencode({"parte": parte, "desde": tem})
                try:
                    linhas = get(f"{a.url}?{q}", tok).get("linhas", [])
                except Exception as e:
                    print(f"erro ao baixar {parte}: {e}"); break
                if not linhas:
                    break
                with open(arq, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(linhas) + "\n")
                tem += len(linhas)
            resumo.append(f"{parte}: {tem:,}/{total:,}")
        print(f"[{datetime.now():%H:%M:%S}] " + (" | ".join(resumo) if resumo else "nenhuma parte remota ainda"), flush=True)
        time.sleep(a.intervalo)


if __name__ == "__main__":
    main()
