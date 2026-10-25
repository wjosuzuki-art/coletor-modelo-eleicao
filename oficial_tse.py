#!/usr/bin/env python3
"""
Resultado OFICIAL agregado (Brasil + 27 UFs + exterior) direto dos arquivos de divulgação do TSE.
São 29 arquivos pequenos por rodada (1 a cada ~2 s) => leve. Grava <saida>/oficial.json, que o ao_vivo.py
inclui no painel (tabela e mapa por estado).

Uso:
    python oficial_tse.py --ensaio --uma-vez      # HOJE: confere com o 1º turno (eleição 6257)
    python oficial_tse.py                          # DOMINGO: 2º turno (eleição 6258), atualiza a cada 60 s
"""
import argparse
import gzip
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime

BASE = "https://resultados.tse.jus.br/oficial/ele2026"
UFS = ("br ac al am ap ba ce df es go ma mg ms mt pa pb pe pi pr rj rn ro rr rs sc se sp to zz").split()
UA = {"User-Agent": "Mozilla/5.0 (projecao-estatistica; pesquisa)", "Accept-Encoding": "gzip"}


def num(x):
    try:
        return float(str(x).replace(".", "").replace(",", ".")) if isinstance(x, str) and "," in str(x) else float(x)
    except Exception:
        return None


def candidatos(obj, achados):
    """Procura, em qualquer profundidade, listas 'cand' com número ('n') e votos apurados ('vap')."""
    if isinstance(obj, dict):
        if isinstance(obj.get("cand"), list):
            for c in obj["cand"]:
                if isinstance(c, dict) and "n" in c:
                    achados[str(c["n"])] = dict(nm=c.get("nmu") or c.get("nm"), vap=num(c.get("vap")), pvap=num(c.get("pvapn") or c.get("pvap")))
        for v in obj.values():
            candidatos(v, achados)
    elif isinstance(obj, list):
        for v in obj:
            candidatos(v, achados)


def baixa(url, etag):
    h = dict(UA)
    if etag:
        h["If-None-Match"] = etag
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=20) as r:
            d = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                d = gzip.decompress(d)
            return 200, json.loads(d.decode("utf-8")), r.headers.get("ETag")
    except urllib.error.HTTPError as e:
        return e.code, None, etag
    except Exception:
        return -1, None, etag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensaio", action="store_true")
    ap.add_argument("--eleicao", type=int, default=6258)
    ap.add_argument("--intervalo", type=int, default=60)
    ap.add_argument("--saida", default="coleta")
    ap.add_argument("--uma-vez", action="store_true")
    a = ap.parse_args()
    ele = 6257 if a.ensaio else a.eleicao
    e6 = f"{ele:06d}"
    os.makedirs(a.saida, exist_ok=True)
    etag, ultimo = {}, {}
    while True:
        t0 = time.time()
        for uf in UFS:
            time.sleep(1.0)                           # ~1 requisição por segundo: irrelevante perto do limite
            st, js, et = baixa(f"{BASE}/{ele}/dados/{uf}/{uf}-c0001-e{e6}-u.json", etag.get(uf))
            if st == 304 or js is None:
                if st not in (304, 200):
                    print(f"[oficial] {uf}: HTTP {st}")
                continue
            etag[uf] = et
            cands = {}
            candidatos(js, cands)
            s = js.get("s", {})
            v = js.get("v", {})
            ultimo[uf.upper()] = dict(pst=num(s.get("pstn") or s.get("pst")), st=num(s.get("st")), ts=num(s.get("ts")),
                                      vv=num(v.get("vv")), cand=cands, hora=f"{js.get('dg', '')} {js.get('hg', '')}")
        out = dict(atualizado=datetime.now().isoformat(timespec="seconds"), eleicao=ele, por_uf=ultimo)
        tmp = os.path.join(a.saida, "oficial.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
        os.replace(tmp, os.path.join(a.saida, "oficial.json"))
        br = ultimo.get("BR", {})
        top = sorted(br.get("cand", {}).items(), key=lambda kv: -(kv[1]["vap"] or 0))[:3]
        print(f"[{datetime.now():%H:%M:%S}] oficial: {len(ultimo)} abrangências | BR {br.get('pst')}% apurado | "
              + " | ".join(f"{n} {c['nm']}: {c['vap']:,.0f} ({c['pvap']}%)" for n, c in top if c["vap"] is not None), flush=True)
        if not top:
            print("[oficial] não achei candidatos no arquivo; mande o coleta/oficial.json para ajustar o leitor.")
        if a.uma_vez:
            break
        time.sleep(max(5, a.intervalo - (time.time() - t0)))


if __name__ == "__main__":
    main()
