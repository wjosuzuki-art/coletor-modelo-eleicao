#!/usr/bin/env python3
"""
COLETOR ao vivo dos boletins de urna (BU) do TSE, por seção, para o modelo.

Como funciona (arquivos oficiais da divulgação 2026):
  1. a cada ~60 s lê o índice de seções de cada UF  (.../arquivo-urna/<pleito>/config/<uf>/<uf>-p<pleito>-cs.json);
     as seções com 'da'/'ha' preenchidos já têm arquivos publicados  -> nunca pedimos o que não existe (evita 404)
  2. escolhe a próxima seção pela COBERTURA: primeiro municípios/UFs com menos seções já coletadas
  3. baixa o aux.json da seção (lista o hash e o nome do BU) e depois o -bu.dat
  4. decodifica o BU (ASN.1, especificação pública do TSE) e grava uma linha por seção em <saida>/coleta.jsonl

Uso (PowerShell), pasta com bu_v2.asn1 e bu_v1.asn1 ao lado:
    pip install asn1tools
    python coletor_tse.py --ensaio --minutos 3 --taxa 5        # ENSAIO HOJE no 1º turno real (pleito 3220)
    python coletor_tse.py                                       # DOMINGO: 2º turno (pleito 3221, eleição 6258)
    python coletor_tse.py --parte 1/3                           # se usar mais de uma máquina (IPs diferentes!)

Limites do TSE: 100 requisições/s por IP (bloqueio de 10 min se passar) e bloqueio por excesso de 404.
Padrão aqui: --taxa 20 (bem abaixo do limite). 2 requisições por seção => ~10 seções/s.
"""
import argparse
import collections
import gzip
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

BASE = os.environ.get("TSE_BASE", "https://resultados.tse.jus.br/oficial/ele2026/arquivo-urna")
UFS = ("ac al am ap ba ce df es go ma mg ms mt pa pb pe pi pr rj rn ro rr rs sc se sp to zz").split()
UA = {"User-Agent": "Mozilla/5.0 (projecao-estatistica; pesquisa)", "Accept-Encoding": "gzip"}
AQUI = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------------------- decodificação do BU
class DecodBU:
    def __init__(self):
        import asn1tools
        self.c = {}
        for v in ("v2", "v1"):
            p = os.path.join(AQUI, f"bu_{v}.asn1")
            if os.path.exists(p):
                self.c[v] = asn1tools.compile_files([p], codec="ber")
        if not self.c:
            sys.exit("Faltam bu_v2.asn1 / bu_v1.asn1 na pasta do script.")

    def decode(self, raw):
        erro = None
        for v, comp in self.c.items():
            try:
                env = comp.decode("EntidadeEnvelopeGenerico", bytearray(raw))
                bu = comp.decode("EntidadeBoletimUrna", bytearray(env["conteudo"]))
                return self._extrai(bu)
            except Exception as e:      # tenta a outra versão da especificação
                erro = e
        raise ValueError(f"BU não decodificado: {erro}")

    @staticmethod
    def _extrai(bu):
        ident = bu.get("identificacaoSecao", {})
        mz = ident.get("municipioZona", {})
        out = dict(mun=mz.get("municipio"), zona=mz.get("zona"), secao=ident.get("secao"),
                   comparecimento=bu.get("qtdEleitoresCompareceram"), emissao=str(bu.get("dataHoraEmissao", "")))
        votos, aptos = collections.Counter(), None
        for rpe in bu.get("resultadosVotacaoPorEleicao", []):
            for rv in rpe.get("resultadosVotacao", []):
                for tc in rv.get("totaisVotosCargo", []):
                    cc = tc.get("codigoCargo")
                    nome = cc[1] if isinstance(cc, tuple) else cc
                    if nome not in ("presidente", 1):
                        continue
                    aptos = rpe.get("qtdEleitoresAptos")
                    out["comparecimento_cargo"] = rv.get("qtdComparecimento")
                    for vv in tc.get("votosVotaveis", []):
                        tv = vv.get("tipoVoto")
                        q = int(vv.get("quantidadeVotos", 0))
                        if tv in ("nominal", "legenda", 1, 4):
                            votos[int(vv["identificacaoVotavel"]["codigo"])] += q
                        elif tv in ("branco", 2):
                            votos[95] += q
                        elif tv in ("nulo", 3):
                            votos[96] += q
        out["aptos"], out["votos"] = aptos, dict(votos)
        return out


# ----------------------------------------------------------------------------------------- HTTP educado
class Taxa:
    """Limita requisições por segundo (todas as threads juntas) e pausa tudo se o TSE reclamar."""
    def __init__(self, por_seg):
        self.int = 1.0 / por_seg
        self.prox = time.monotonic()
        self.trava = threading.Lock()
        self.pausa_ate = 0.0
        self.n404 = collections.deque()

    def espera(self):
        with self.trava:
            agora = time.monotonic()
            if agora < self.pausa_ate:
                time.sleep(self.pausa_ate - agora)
                agora = time.monotonic()
            self.prox = max(self.prox + self.int, agora)
            t = self.prox - agora
        if t > 0:
            time.sleep(t)

    def pausar(self, seg, motivo):
        with self.trava:
            self.pausa_ate = max(self.pausa_ate, time.monotonic() + seg)
        print(f"[taxa] PAUSA {seg}s: {motivo}", flush=True)

    def registra_404(self):
        agora = time.monotonic()
        self.n404.append(agora)
        while self.n404 and agora - self.n404[0] > 60:
            self.n404.popleft()
        if len(self.n404) >= 10:
            self.pausar(120, "10+ erros 404 em 1 minuto (o TSE bloqueia IP por isso)")
            self.n404.clear()


def http(url, taxa, etag=None):
    taxa.espera()
    h = dict(UA)
    if etag:
        h["If-None-Match"] = etag
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=20) as r:
            d = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                d = gzip.decompress(d)
            return 200, d, r.headers.get("ETag")
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return 304, None, etag
        if e.code == 404:
            taxa.registra_404()
        if e.code in (403, 429):
            taxa.pausar(660, f"HTTP {e.code} (provável bloqueio; esperando 11 min sem insistir)")
        return e.code, None, None
    except Exception as e:
        return -1, str(e).encode(), None


# ----------------------------------------------------------------------------------------- envio para o PC central
class Envio:
    """Manda as linhas coletadas para /api/coleta (Vercel) em lotes, numa thread separada.
    Se a internet falhar, guarda e reenvia depois. Guarda quantas já foram aceitas em <saida>/.enviadas_<parte>."""
    def __init__(self, a, k, n, linhas_locais):
        self.url, self.tok = a.enviar, os.environ.get("COLETA_TOKEN", "").strip()
        if not self.tok:
            sys.exit("Defina $env:COLETA_TOKEN (a senha da coleta) antes de usar --enviar.")
        self.parte = f"p{a.pleito}_{k}de{n}"
        self.marca = os.path.join(a.saida, f".enviadas_{self.parte}")
        try:
            ja = int(open(self.marca).read().strip())
        except Exception:
            ja = 0
        self.fila = collections.deque(linhas_locais[ja:])
        self.n_ok = ja
        self.lock = threading.Lock()
        if self.fila:
            print(f"[envio] {len(self.fila):,} linhas pendentes de envio de antes")
        threading.Thread(target=self._loop, daemon=True).start()

    def adiciona(self, txt):
        with self.lock:
            self.fila.append(txt)

    def _loop(self):
        while True:
            time.sleep(15)
            while True:
                with self.lock:
                    lote = [self.fila[i] for i in range(min(1500, len(self.fila)))]
                if not lote:
                    break
                corpo = json.dumps({"parte": self.parte, "linhas": lote}).encode("utf-8")
                req = urllib.request.Request(self.url, data=corpo, method="POST", headers={
                    "Content-Type": "application/json", "Authorization": f"Bearer {self.tok}"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        if r.status != 200:
                            raise RuntimeError(r.status)
                except Exception as e:
                    print(f"[envio] falhou ({e}); tento de novo em 15 s", flush=True)
                    break
                with self.lock:
                    for _ in lote:
                        self.fila.popleft()
                    self.n_ok += len(lote)
                with open(self.marca, "w") as fh:
                    fh.write(str(self.n_ok))
                print(f"[envio] +{len(lote)} linhas enviadas (total {self.n_ok:,})", flush=True)


# ----------------------------------------------------------------------------------------- coletor
class Coletor:
    def __init__(self, a):
        self.a = a
        self.p6 = f"{a.pleito:06d}"
        self.taxa = Taxa(a.taxa)
        self.dec = DecodBU()
        self.etag = {}
        self.disp = {}                 # (uf, mun, zona, sec) -> "da ha"
        self.feito = set()
        self.falhou = collections.Counter()
        self.cont_mun = collections.Counter()
        self.cont_uf = collections.Counter()
        self.tot_mun = collections.Counter()
        self.lock = threading.Lock()
        os.makedirs(a.saida, exist_ok=True)
        self.k, self.n = (int(x) for x in a.parte.split("/"))
        sufixo = f"_parte{self.k}de{self.n}" if self.n > 1 else ""
        self.base_arq = os.path.join(a.saida, f"coleta_p{a.pleito}{sufixo}")
        # retoma de onde parou (se o PC reiniciar): lê TODOS os arquivos desta parte, inclusive os blocos já fechados
        import glob as _g
        anteriores = sorted(set(_g.glob(self.base_arq + ".jsonl") + _g.glob(self.base_arq + "_b*.jsonl")))
        self.linhas_locais = []
        for arq in anteriores:
            with open(arq, encoding="utf-8") as fh:
                for ln in fh:
                    try:
                        r = json.loads(ln)
                        self.linhas_locais.append(ln.rstrip("\n"))
                        k = (r["uf"], r["mun"], r["zona"], r["secao"])
                        self.feito.add(k); self.cont_mun[(k[0], k[1])] += 1; self.cont_uf[k[0]] += 1
                    except Exception:
                        pass
        if anteriores:
            print(f"retomando: {len(self.feito):,} seções já coletadas em {len(anteriores)} arquivo(s)")
        self.arq = self._nome_arq()
        self.envio = Envio(a, self.k, self.n, self.linhas_locais) if a.enviar else None
        self.stats = collections.Counter()

    def _nome_arq(self):
        """Com --blocos N, grava um arquivo novo a cada N minutos (bom para pasta sincronizada: cada arquivo
        para de mudar logo e é enviado uma vez só, em vez de reenviar um arquivo gigante a cada linha)."""
        if not self.a.blocos:
            return self.base_arq + ".jsonl"
        b = int(time.time() // (60 * self.a.blocos))
        return f"{self.base_arq}_b{datetime.fromtimestamp(b * 60 * self.a.blocos):%H%M}.jsonl"

    def minha(self, chave):
        return (zlib.crc32('|'.join(chave).encode()) % self.n) + 1 == self.k if self.n > 1 else True

    def le_indices(self, ufs):
        novas = 0
        for uf in ufs:
            url = f"{BASE}/{self.a.pleito}/config/{uf}/{uf}-p{self.p6}-cs.json"
            st, d, et = http(url, self.taxa, self.etag.get(uf))
            self.stats[f"cs_{st}"] += 1
            if st != 200 or not d:
                continue
            self.etag[uf] = et
            try:
                cs = json.loads(d.decode("utf-8"))
            except Exception:
                continue
            abr = cs.get("abr", {})
            for ab in (abr if isinstance(abr, list) else [abr]):
                for mu in ab.get("mu", []):
                    for z in mu.get("zon", []):
                        for s in z.get("sec", []):
                            k = (uf, mu["cd"], z["cd"], s["ns"])
                            if s.get("nsp"):          # seção agregada: os votos vêm no BU da principal
                                continue
                            if k not in self.disp:
                                self.tot_mun[(uf, mu["cd"])] += 1
                                self.disp[k] = None
                            if s.get("da") and self.disp[k] is None:
                                novas += 1
                                self.disp[k] = f"{s['da']} {s.get('ha', '')}"
        return novas

    def fila(self):
        cand = [k for k, v in self.disp.items() if v and k not in self.feito and self.falhou[k] < 3 and self.minha(k)]
        random.shuffle(cand)
        # cobertura: menos coletadas no município (proporcionalmente) e na UF vão primeiro
        cand.sort(key=lambda k: (self.cont_mun[(k[0], k[1])] / max(self.tot_mun[(k[0], k[1])], 1),
                                 self.cont_uf[k[0]]))
        return cand

    def baixa_secao(self, k):
        uf, mun, zon, sec = k
        pasta = f"{BASE}/{self.a.pleito}/dados/{uf}/{mun}/{zon}/{sec}"
        st, d, _ = http(f"{pasta}/p{self.p6}-{uf}-m{mun}-z{zon}-s{sec}-aux.json", self.taxa)
        self.stats[f"aux_{st}"] += 1
        if st != 200:
            with self.lock: self.falhou[k] += 1
            return
        try:
            aux = json.loads(d.decode("utf-8"))
            hs = [h for h in aux.get("hashes", []) if any(x.get("tp") in ("bu", "busa") for x in h.get("arq", []))]
            h = hs[-1]                                     # o mais recente (se a urna foi substituída)
            nm = next(x["nm"] for x in h["arq"] if x.get("tp") in ("bu", "busa"))
        except Exception:
            with self.lock: self.falhou[k] += 1
            return
        st, raw, _ = http(f"{pasta}/{h['hash']}/{nm}", self.taxa)
        self.stats[f"bu_{st}"] += 1
        if st != 200:
            with self.lock: self.falhou[k] += 1
            return
        try:
            r = self.dec.decode(raw)
        except Exception as e:
            self.stats["bu_decod_erro"] += 1
            with self.lock: self.falhou[k] += 1
            if self.stats["bu_decod_erro"] <= 3:
                print(f"[decod] {k}: {e}", flush=True)
            return
        lin = dict(uf=uf, mun=mun, zona=zon, secao=sec, turno=self.a.turno, recebido=f"{h.get('dr', '')} {h.get('hr', '')}",
                   disponivel=self.disp.get(k), coletado=datetime.now().isoformat(timespec="seconds"),
                   aptos=r["aptos"], comparecimento=r.get("comparecimento_cargo") or r["comparecimento"],
                   votos=r["votos"], st=aux.get("st"))
        with self.lock:
            self.arq = self._nome_arq()
            txt = json.dumps(lin, ensure_ascii=False)
            with open(self.arq, "a", encoding="utf-8") as fh:
                fh.write(txt + "\n")
            if self.envio:
                self.envio.adiciona(txt)
            self.feito.add(k); self.cont_mun[(uf, mun)] += 1; self.cont_uf[uf] += 1
            self.stats["ok"] += 1

    def grava_indice(self):
        """Resumo do índice para o painel: quantas seções cada UF tem e quantas já foram divulgadas pelo TSE."""
        tot, pub = collections.Counter(), collections.Counter()
        for k, v in self.disp.items():
            tot[k[0]] += 1
            pub[k[0]] += bool(v)
        res = dict(atualizado=datetime.now().isoformat(timespec="seconds"), pleito=self.a.pleito,
                   por_uf={u: dict(total=tot[u], divulgadas=pub[u], coletadas=self.cont_uf[u]) for u in tot})
        tmp = os.path.join(self.a.saida, "indice_resumo.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        os.replace(tmp, os.path.join(self.a.saida, "indice_resumo.json"))

    def rodar(self):
        ufs = [u for u in UFS if not self.a.ufs or u in self.a.ufs.split(",")]
        fim = time.time() + 60 * self.a.minutos if self.a.minutos else float("inf")
        prox_idx = 0.0
        with ThreadPoolExecutor(max_workers=self.a.threads) as pool:
            while time.time() < fim:
                if time.time() >= prox_idx:
                    nv = self.le_indices(ufs)
                    prox_idx = time.time() + self.a.intervalo_indice
                    disp = sum(1 for v in self.disp.values() if v)
                    self.grava_indice()
                    print(f"[{datetime.now():%H:%M:%S}] índice: {disp:,} seções disponíveis (+{nv:,}) | "
                          f"coletadas {len(self.feito):,} | {dict(self.stats)}", flush=True)
                fila = self.fila()[: self.a.taxa * self.a.intervalo_indice // 2 + 1]
                if not fila:
                    time.sleep(5)
                    continue
                list(pool.map(self.baixa_secao, fila))
        if self.envio:                                  # dá até 2 min para mandar o que sobrou
            for _ in range(120):
                if not self.envio.fila:
                    break
                time.sleep(1)
            print(f"[envio] pendentes no fim: {len(self.envio.fila)}")
        print(f"fim. coletadas {len(self.feito):,} seções -> {self.base_arq}*.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensaio", action="store_true", help="usa o 1º turno de 2026 (pleito 3220), que já está no ar")
    ap.add_argument("--pleito", type=int, default=3221)
    ap.add_argument("--taxa", type=int, default=20, help="requisições por segundo (TSE bloqueia acima de 100)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--intervalo-indice", type=int, default=60, help="segundos entre leituras do índice (cache do TSE ~50 s)")
    ap.add_argument("--ufs", default="", help="ex.: ac,ap (teste); vazio = todas")
    ap.add_argument("--minutos", type=float, default=0, help="para depois de N minutos (0 = não para)")
    ap.add_argument("--parte", default="1/1", help="k/n para dividir entre máquinas com IPs diferentes")
    ap.add_argument("--saida", default="coleta")
    ap.add_argument("--enviar", default="", help="URL de /api/coleta do seu site (PCs remotos); exige $env:COLETA_TOKEN")
    ap.add_argument("--blocos", type=int, default=0,
                    help="minutos por arquivo (use 5 quando a --saida for uma pasta sincronizada: Drive/OneDrive/Dropbox)")
    a = ap.parse_args()
    if a.ensaio:
        a.pleito = 3220
    a.turno = 1 if a.pleito == 3220 else 2
    if a.taxa > 60:
        sys.exit("--taxa acima de 60 não: o limite do TSE é 100/s por IP e há outras requisições suas na mesma rede.")
    print(f"coletor: pleito {a.pleito} (turno {a.turno}) | {a.taxa} req/s | parte {a.parte} | saída {a.saida}/")
    Coletor(a).rodar()


if __name__ == "__main__":
    main()
