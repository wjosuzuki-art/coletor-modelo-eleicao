#!/usr/bin/env python3
"""
MODO AO VIVO: junta o 1º turno de 2026 (âncora, todas as seções) com o que o coletor_tse.py já baixou do 2º turno,
roda o modelo a cada N minutos e publica no painel (Vercel).

Antes (uma vez): baixe os 28 zips "bweb_1t_XX_051020261403.zip" do Portal de Dados Abertos e rode
    python extrair_bu.py pasta_zips_1t_2026 --saida bu2026_1t

Domingo, em DUAS janelas do PowerShell (mesma pasta):
    python coletor_tse.py                                        # janela 1: coleta
    python ao_vivo.py bu2026_1t coleta --nome-a Lula --nome-b Flávio   # janela 2: modelo + publicação

ENSAIO HOJE (1º turno no lugar do 2º; o swing dá ~0 e o resultado tem que bater com o 1º turno):
    python coletor_tse.py --ensaio --taxa 10 --minutos 10
    python ao_vivo.py bu2026_1t coleta --ensaio --uma-vez
O ensaio também CONFERE o decodificador: compara os votos baixados com o CSV oficial do 1º turno.
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import modelo_prob as M
from publicar import publicar
from replay_ao_vivo import montar_estado

K = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]


def carregar_1t(pasta, A, B):
    partes = []
    for f in sorted(glob.glob(os.path.join(pasta, "*_secoes.csv"))):
        df = pd.read_csv(f, usecols=lambda c: c in M.NEED or c.startswith("v_"))
        df = df[df.NR_TURNO == 1]
        if df.empty:
            continue
        vc = [c for c in df.columns if c.startswith("v_")]
        df[vc] = df[vc].fillna(0)
        for c in (f"v_{A}", f"v_{B}"):
            if c not in df:
                df[c] = 0
        df["valid1"] = df[[c for c in vc if c not in ("v_95", "v_96")]].sum(axis=1)
        df["n1"], df["a1"], df["b1"] = df[f"v_{A}"] + df[f"v_{B}"], df[f"v_{A}"], df[f"v_{B}"]
        df["aptos"] = df.QT_APTOS
        t1 = pd.to_datetime(df.DT_BU_RECEBIDO_ts, errors="coerce")
        df["ord1"] = t1.groupby(df.SG_UF).rank(pct=True).fillna(0.5).values
        outros = [c for c in vc if c not in (f"v_{A}", f"v_{B}", "v_95", "v_96")]
        df = df.rename(columns={c: "c1_" + c[2:] for c in outros})
        partes.append(df[K + ["valid1", "n1", "a1", "b1", "aptos", "ord1"] + ["c1_" + c[2:] for c in outros]])
    if not partes:
        sys.exit(f"Nenhum *_secoes.csv do 1º turno em {pasta}")
    D = pd.concat(partes, ignore_index=True)
    c1 = [c for c in D.columns if c.startswith("c1_")]
    D[c1] = D[c1].fillna(0)
    D = D[(D.n1 > 0) & (D.valid1 > 0)].copy()
    D["uf"] = D.SG_UF.str.upper()
    D["reg"] = D.uf.map(M.REG).fillna("?")
    D["mun"] = D.uf + "_" + D.CD_MUNICIPIO.astype(int).astype(str)
    D["l1"] = M.lg((D.a1 + .5) / (D.n1 + 1))
    D["outros1"] = 1 - D.n1 / D.valid1
    D["lvalid"], D["lapt"] = np.log1p(D.valid1), np.log1p(D.aptos)
    return D.reset_index(drop=True)


def carregar_coleta(pasta, pleito, A, B):
    linhas = []
    for arq in sorted(glob.glob(os.path.join(pasta, f"coleta_p{pleito}*.jsonl"))):   # junta as partes das várias máquinas
        with open(arq, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                    v = r.get("votos", {})
                    linhas.append(dict(SG_UF=r["uf"].upper(), CD_MUNICIPIO=int(r["mun"]), NR_ZONA=int(r["zona"]),
                                       NR_SECAO=int(r["secao"]), a2=int(v.get(str(A), 0)), b2=int(v.get(str(B), 0)),
                                       t2=pd.to_datetime(r.get("recebido"), format="%d/%m/%Y %H:%M:%S", errors="coerce")))
                except Exception:
                    pass
    C = pd.DataFrame(linhas, columns=K + ["a2", "b2", "t2"])
    return C.drop_duplicates(K, keep="last")


def apuracao_tse(pasta):
    try:
        with open(os.path.join(pasta, "indice_resumo.json"), encoding="utf-8") as fh:
            ind = json.load(fh)["por_uf"]
        tot = sum(v["total"] for v in ind.values())
        pub = sum(v["divulgadas"] for v in ind.values())
        reg = {}
        for u, v in ind.items():
            r = M.REG.get(u.upper(), "?")
            t, p = reg.get(r, (0, 0))
            reg[r] = (t + v["total"], p + v["divulgadas"])
        return pub / max(tot, 1), {r: round(p / max(t, 1), 4) for r, (t, p) in reg.items()}
    except Exception:
        return None, None


def ler_oficial(pasta, A, B):
    """Resultado oficial do TSE (oficial_tse.py): % apurado e share de A entre A e B, por UF e Brasil."""
    try:
        with open(os.path.join(pasta, "oficial.json"), encoding="utf-8") as fh:
            of = json.load(fh)
    except Exception:
        return None
    res = {}
    for uf, d in of.get("por_uf", {}).items():
        ca, cb = d.get("cand", {}).get(str(A), {}), d.get("cand", {}).get(str(B), {})
        va, vb = ca.get("vap") or 0, cb.get("vap") or 0
        res[uf] = dict(pst=d.get("pst"), votos_a=va, votos_b=vb, n=va + vb, share_a=(va / (va + vb)) if va + vb else None)
    return dict(atualizado=of.get("atualizado"), por_uf=res)


def projecao_por_uf(x, y, estado):
    """Projeção do modelo por UF (centro, sem intervalo): contado na amostra + previsto para o resto."""
    th_aj, rt = M.ajusta(x, y)
    cc, _ = M.corr_chegada(x, y)
    n2h = y.valid1.values * rt
    pa = M.desuav(M.ex(y.l1.values + th_aj + cc), n2h)
    Y = pd.DataFrame(dict(uf=y.uf.values, a=n2h * pa, n=n2h)).groupby("uf").sum()
    X = x.groupby("uf").agg(a=("a2", "sum"), n=("n2", "sum"))
    T = X.add(Y, fill_value=0)
    return {u: round(float(r.a / r.n), 5) for u, r in T.iterrows() if r.n > 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta_1t")
    ap.add_argument("pasta_coleta")
    ap.add_argument("--ensaio", action="store_true")
    ap.add_argument("--a", default="13"); ap.add_argument("--b", default="22")
    ap.add_argument("--nome-a", default="Lula"); ap.add_argument("--nome-b", default="Flávio")
    ap.add_argument("--intervalo", type=int, default=180, help="segundos entre atualizações do modelo")
    ap.add_argument("--min-secoes", type=int, default=3000,
                    help="não publica probabilidade antes disso (no teste de 2018, com poucas centenas de seções um método chegou a 98%% do lado errado)")
    ap.add_argument("--min-ufs", type=int, default=20, help="nem antes de ter seções de pelo menos N UFs")
    ap.add_argument("--min-por-regiao", type=int, default=100,
                    help="nem antes de CADA região do país (fora o exterior) ter pelo menos N seções na amostra")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--uma-vez", action="store_true")
    ap.add_argument("--local", default="estado_ao_vivo.json")
    cfg = ap.parse_args()
    cfg.sd_uf, cfg.sd_nac, cfg.sd_mun_min = 0.010, 0.010, 0.03
    cfg.cap_chegada, cfg.inc_chegada, cfg.seq_decidido = 0.03, 0.5, 3
    cfg.na, cfg.nb = cfg.nome_a, cfg.nome_b
    pleito = 3220 if cfg.ensaio else 3221
    rng = np.random.default_rng(7)

    D1 = carregar_1t(cfg.pasta_1t, cfg.a, cfg.b)
    import metodos as ME
    cfg.perfil = ME.prep_perfil(D1)
    print(f"1º turno: {len(D1):,} seções | {D1.uf.nunique()} UFs | share de {cfg.na} (entre os dois): {D1.a1.sum() / D1.n1.sum():.4f}")
    serie, seq = [], (0, None)
    while True:
        t0 = time.time()
        C = carregar_coleta(cfg.pasta_coleta, pleito, cfg.a, cfg.b)
        D = D1.merge(C, on=K, how="left")
        D["a2"] = pd.to_numeric(D.a2, errors="coerce").astype(float)   # coleta vazia vem como "object": força número
        D["b2"] = pd.to_numeric(D.b2, errors="coerce").astype(float)
        D["n2"] = D.a2 + D.b2
        tem = D.n2.notna() & (D.n2 > 0)
        if cfg.ensaio and tem.any():
            ok = ((D.a2 == D.a1) & (D.b2 == D.b1))[tem].mean()
            print(f"[ensaio] conferência do decodificador: {ok:.1%} das {tem.sum():,} seções baixadas batem com o CSV oficial "
                  f"(votos de {cfg.a} e {cfg.b})")
        nao_casou = len(C) - tem.sum()
        if nao_casou > 0:
            print(f"[aviso] {nao_casou:,} seções coletadas não acharam par no 1º turno (seção nova/renumerada?)")
        D["sw"] = np.where(tem, M.lg((D.a2.fillna(0) + .5) / (D.n2.fillna(0) + 1)) - D.l1, np.nan)
        x, y = D[tem].copy(), D[~tem].copy()
        por_reg = x[x.reg != "Exterior"].groupby("reg").size().reindex(
            [r for r in D1.reg.unique() if r != "Exterior"]).fillna(0)
        if len(x) >= cfg.min_secoes and x.uf.nunique() >= cfg.min_ufs and por_reg.min() >= cfg.min_por_regiao:
            agora = pd.Timestamp(datetime.now())
            pct_tse, reg_tse = apuracao_tse(cfg.pasta_coleta)
            estado, seq = montar_estado(D, agora, x, y, cfg, serie, seq, rng,
                                        "ENSAIO (1º turno no lugar do 2º)" if cfg.ensaio else "ao vivo",
                                        apur=(pct_tse, reg_tse) if pct_tse is not None else None)
            c = estado["coleta"]
            c["amostra_secoes"] = int(len(x))
            if pct_tse is not None:
                c["pct"], c["por_regiao"] = round(pct_tse, 4), reg_tse
                estado["serie"][-1]["pct"] = round(pct_tse, 4)
            estado["oficial"] = ler_oficial(cfg.pasta_coleta, cfg.a, cfg.b)
            ofbr = (estado["oficial"] or {}).get("por_uf", {}).get("BR", {}) if estado["oficial"] else {}
            estado["serie"][-1]["oficial"] = ofbr.get("share_a")
            with open(cfg.local, "w", encoding="utf-8") as fh:
                json.dump(estado, fh, ensure_ascii=False)
            ok = publicar(estado)
            print(f"[{agora:%H:%M:%S}] amostra {len(x):,} | TSE {pct_tse if pct_tse is None else f'{pct_tse:.1%}'} | "
                  f"{estado['atual']['status']} ({estado['atual']['P_lider']:.1%}) | "
                  f"{cfg.na} {100 * estado['atual']['shareA']:.2f}% | publicado: {ok}", flush=True)
        else:
            print(f"[{datetime.now():%H:%M:%S}] aguardando: {len(x)} seções de {x.uf.nunique()} UFs "
                  f"(mínimo {cfg.min_secoes} seções, {cfg.min_ufs} UFs e {cfg.min_por_regiao} por região) | "
                  f"por região: {por_reg.astype(int).to_dict()}", flush=True)
        if cfg.uma_vez:
            break
        time.sleep(max(5, cfg.intervalo - (time.time() - t0)))


if __name__ == "__main__":
    main()
