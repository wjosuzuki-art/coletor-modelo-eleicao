#!/usr/bin/env python3
"""
Série temporal de P(A vence) no 2º turno: baseline hierárquico (swing em logit) + correção pela ordem de chegada
+ camada de resíduo (XGBoost), com Monte Carlo para a incerteza e um painel em linguagem humana.

Uso (PowerShell):
    python modelo_prob.py saida_bu --nome-a Lula --nome-b Flávio --saida prob_2022
    python modelo_prob.py saida_bu --a 13 --b 17                        # 2018
    python modelo_prob.py saida_bu --desloca -0.045 --saida prob_teste  # cenário contrafactual (corrida apertada)

PASTA = os *_secoes.csv do extrair_bu.py (1º e 2º turno, várias UFs; turno/UF vêm do conteúdo).
Saídas: <saida>.csv (série), <saida>.png (gráfico), <saida>.txt (relatório), <saida>_frases.txt (painel).

Em cada instante t (replay pela hora real de DT_BU_RECEBIDO do 2º turno):
  1. seções chegadas = treino; seções que faltam = o que precisa ser previsto
  2. BASELINE : swing = logit(share2) - logit(share1), média encolhida país -> região -> UF -> município
  3. CHEGADA  : dentro de cada município, quanto o swing muda com a ORDEM DE CHEGADA DO 1º TURNO (conhecida antes);
                aplicada às seções que faltam (que tendem a ser as "tardias"). Desligue com --sem-chegada.
  4. RESÍDUO  : XGBoost aprende o que sobra; só entra depois de --gbm-min seções, e em média com o modelo 3
  5. INCERTEZA: Monte Carlo com erros compartilhados por município / UF / país + piso + incerteza da correção 3
  6. P(A vence) = fração das simulações em que o share final de A > 50%
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
lg = lambda p: np.log(p / (1 - p))
ex = lambda x: 1 / (1 + np.exp(-x))
REG = {**{u: "Norte" for u in "AC AM AP PA RO RR TO".split()},
       **{u: "Nordeste" for u in "AL BA CE MA PB PE PI RN SE".split()},
       **{u: "Centro-Oeste" for u in "DF GO MS MT".split()},
       **{u: "Sudeste" for u in "ES MG RJ SP".split()},
       **{u: "Sul" for u in "PR RS SC".split()}, "ZZ": "Exterior"}
K = ["CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
NEED = {"NR_TURNO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "NR_LOCAL_VOTACAO", "QT_APTOS", "DT_BU_RECEBIDO_ts"}
OUT = []
FRASES = []
DETECT = []
TAU_UF = 0.04   # desvio do swing entre UFs (logit), da decomposição de variância de 2022 (~5% de 0,0132 => 0,026; margem)


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def desuav(ps, n):
    """O swing é calculado com shares suavizados (a+0,5)/(n+1); isto devolve o share 'cru' previsto.
    Sem isto, seções com 80% para alguém são puxadas para 50% (viés pequeno mas sistemático)."""
    n = np.maximum(n, 1.0)
    return np.clip((ps * (n + 1) - 0.5) / n, 0.0, 1.0)


# ----------------------------------------------------------------------------------------- dados
def carregar(pasta, A, B, ufs, desloca=0.0):
    por = {}
    for f in sorted(glob.glob(os.path.join(pasta, "*_secoes.csv"))):
        df = pd.read_csv(f, usecols=lambda c: c in NEED or c.startswith("v_"))
        if {"NR_TURNO", "SG_UF", "DT_BU_RECEBIDO_ts"} - set(df.columns):
            continue
        for (uf, t), g in df.groupby(["SG_UF", "NR_TURNO"]):
            if ufs and uf not in ufs:
                continue
            por.setdefault((uf, int(t)), g.copy())
    partes = []
    for uf in sorted({u for u, _ in por}):
        if (uf, 1) not in por or (uf, 2) not in por:
            continue
        a, b = por[(uf, 1)].copy(), por[(uf, 2)].copy()
        for df in (a, b):
            vc = [c for c in df.columns if c.startswith("v_")]
            df[vc] = df[vc].fillna(0)
        if str(A).lower() == "auto":   # governador etc.: os 2 mais votados no 2º turno; A = o de MENOR número
            vc2 = [c for c in b.columns if c.startswith("v_") and c not in ("v_95", "v_96")]
            top = b[vc2].sum().sort_values(ascending=False).index[:2]
            va, vb = sorted(top, key=lambda c: int(c[2:]))
            DETECT[:] = [va[2:], vb[2:]]
            print(f"[{uf}] candidatos detectados no 2º turno: A={va[2:]} | B={vb[2:]}")
        else:
            va, vb = f"v_{A}", f"v_{B}"
        if va not in a or vb not in a or va not in b or vb not in b:
            sys.exit(f"Colunas {va}/{vb} não existem em {uf}. Confira --a/--b.")
        a["valid1"] = a[[c for c in a.columns if c.startswith("v_") and c not in ("v_95", "v_96")]].sum(axis=1)
        a["n1"], b["n2"] = a[va] + a[vb], b[va] + b[vb]
        a = a.rename(columns={va: "a1", "NR_LOCAL_VOTACAO": "loc1", "QT_APTOS": "aptos", "DT_BU_RECEBIDO_ts": "t1raw"})
        b = b.rename(columns={va: "a2"})
        outros = [c for c in a.columns if c.startswith("v_") and c not in (vb, "v_95", "v_96")]
        a = a.rename(columns={c: "c1_" + c[2:] for c in outros})
        c1 = [ "c1_" + c[2:] for c in outros]
        d = a[K + ["loc1", "aptos", "valid1", "n1", "a1", "t1raw"] + c1].merge(b[K + ["n2", "a2", "DT_BU_RECEBIDO_ts"]], on=K)
        d = d[(d.n1 > 0) & (d.n2 > 0) & (d.valid1 > 0)].copy()
        d["uf"], d["reg"] = uf, REG.get(uf, "?")
        # ordem de chegada no 1º TURNO dentro da UF (0 = primeira, 1 = última): conhecida antes do 2º turno começar
        t1 = pd.to_datetime(d.t1raw, errors="coerce")
        d["ord1"] = t1.rank(pct=True).fillna(0.5).values
        partes.append(d.drop(columns="t1raw"))
    if not partes:
        sys.exit("Nenhuma UF com os dois turnos.")
    D = pd.concat(partes, ignore_index=True)
    c1 = [c for c in D.columns if c.startswith("c1_")]
    D[c1] = D[c1].fillna(0)
    D["t2"] = pd.to_datetime(D.DT_BU_RECEBIDO_ts)
    D = D.drop(columns="DT_BU_RECEBIDO_ts")
    D["mun"] = D.uf + "_" + D.CD_MUNICIPIO.astype(str)
    if desloca:   # CENÁRIO CONTRAFACTUAL: soma 'desloca' (logit) ao voto de A em toda seção do 2º turno
        D["a2"] = D.n2 * ex(lg((D.a2 + .5) / (D.n2 + 1)) + desloca)
    p1, p2 = (D.a1 + .5) / (D.n1 + 1), (D.a2 + .5) / (D.n2 + 1)
    D["l1"], D["sw"] = lg(p1), lg(p2) - lg(p1)
    D["outros1"] = 1 - D.n1 / D.valid1
    D["lvalid"], D["lapt"] = np.log1p(D.valid1), np.log1p(D.aptos)
    # contexto do município, conhecido ANTES (só 1º turno, todas as seções)
    g = D.groupby("mun")
    D["mun_nsec"] = np.log(g.l1.transform("size"))
    D["mun_l1"] = g.apply(lambda s: np.average(s.l1, weights=s.n1), include_groups=False).reindex(D.mun).values
    D["uf_cod"], D["reg_cod"] = D.uf.astype("category").cat.codes, D.reg.astype("category").cat.codes
    return D.sort_values("t2").reset_index(drop=True)


# ----------------------------------------------------------------------------------------- baseline
def ajusta(x, alvo, kreg=50, kuf=20, kmun=10):
    """Swing e razão de tamanho encolhidos (país->região->UF->município), avaliados nas linhas de 'alvo'."""
    x = x.assign(wsw=x.sw * x.n2)
    g_th, g_rt = x.wsw.sum() / x.n2.sum(), x.n2.sum() / x.valid1.sum()

    def nivel(chave, pth, prt, k):
        t = x.groupby(chave).agg(n=("sw", "size"), W=("n2", "sum"), S=("wsw", "sum"), V1=("valid1", "sum"))
        p_th, p_rt = pth(t.index), prt(t.index)
        t["th"] = p_th + t.n / (t.n + k) * (t.S / t.W - p_th)
        t["rt"] = p_rt + t.n / (t.n + k) * (t.W / t.V1 - p_rt)
        return t

    R = nivel("reg", lambda i: g_th, lambda i: g_rt, kreg)
    r_uf = x.drop_duplicates("uf").set_index("uf").reg
    U = nivel("uf", lambda i: i.map(r_uf).map(R.th).fillna(g_th).values, lambda i: i.map(r_uf).map(R.rt).fillna(g_rt).values, kuf)
    u_mun = x.drop_duplicates("mun").set_index("mun").uf
    M = nivel("mun", lambda i: i.map(u_mun).map(U.th).values, lambda i: i.map(u_mun).map(U.rt).values, kmun)
    th = alvo.reg.map(R.th).fillna(g_th)
    rt = alvo.reg.map(R.rt).fillna(g_rt)
    th = alvo.uf.map(U.th).fillna(th)
    rt = alvo.uf.map(U.rt).fillna(rt)
    th = alvo.mun.map(M.th).fillna(th)
    rt = alvo.mun.map(M.rt).fillna(rt)
    return th.values, rt.values


# ----------------------------------------------------------------------------------------- correção pela ordem de chegada
def corr_chegada(x, y, cap=0.03, kmun=10):
    """Dentro de cada município: o swing das seções que chegam mais tarde é diferente? (efeito fixo de município)
    Covariável = ordem de chegada no 1º TURNO (ord1), que existe também para as seções que ainda não chegaram.
    Devolve a correção de swing para y (logit, limitada a ±cap) e a inclinação estimada."""
    w = x.n2.values.astype(float)
    g = x.assign(w=w, ws=x.sw * w, wo=x.ord1 * w).groupby("mun")
    n = g.size()
    ms = (g.ws.sum() / g.w.sum())
    mo = (g.wo.sum() / g.w.sum())
    ok = x.mun.map(n).values >= 3
    if ok.sum() < 500:
        return np.zeros(len(y)), 0.0
    ds = (x.sw - x.mun.map(ms)).values[ok]
    do = (x.ord1 - x.mun.map(mo)).values[ok]
    ww = w[ok]
    den = (ww * do ** 2).sum()
    if den <= 0:
        return np.zeros(len(y)), 0.0
    beta = (ww * do * ds).sum() / den
    # referência: a ordem média das seções que JÁ formaram o swing usado para cada seção que falta
    # (município, se tem chegadas; senão UF; senão país), com o mesmo encolhimento do baseline
    mo_uf = x.assign(wo=x.ord1 * w, w=w).groupby("uf").apply(lambda s: s.wo.sum() / s.w.sum(), include_groups=False)
    mo_nac = (x.ord1 * w).sum() / w.sum()
    ref_uf = y.uf.map(mo_uf).fillna(mo_nac).values
    n_m = y.mun.map(n).fillna(0).values
    ref_m = y.mun.map(mo).values
    ref = np.where(n_m > 0, ref_uf + n_m / (n_m + kmun) * (np.nan_to_num(ref_m) - ref_uf), ref_uf)
    return np.clip(beta * (y.ord1.values - ref), -cap, cap), beta


# ----------------------------------------------------------------------------------------- resíduo (GBM)
try:
    import xgboost as xgb
    MOTOR = "xgboost"
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    MOTOR = "sklearn.HistGradientBoosting (xgboost não instalado)"

FEATS = ["th_base", "l1", "outros1", "lvalid", "lapt", "mun_nsec", "mun_l1", "n_arr_mun", "uf_cod", "reg_cod", "ord1"]


def novo_gbm():
    if MOTOR == "xgboost":
        return xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, min_child_weight=50,
                                subsample=0.7, colsample_bytree=0.8, reg_lambda=20, n_jobs=-1, tree_method="hist")
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=200, l2_regularization=20)


def oof_base(x, folds, grupo):
    """Baseline previsto FORA da amostra para as seções chegadas. grupo: coluna que define as dobras."""
    cod = pd.factorize(x[grupo])[0] % folds
    th = np.zeros(len(x))
    nar = np.zeros(len(x))
    for f in range(folds):
        te = cod == f
        th[te] = ajusta(x[~te], x[te])[0]
        cont = x[~te].groupby("mun").size()
        nar[te] = x[te].mun.map(cont).fillna(0).values
    return th, nar


def residuo_gbm(x, y, th_y, rng):
    """Correção de swing (logit) para as seções y, aprendida no resíduo fora-da-amostra do baseline."""
    sub = x if len(x) <= 120_000 else x.sample(120_000, random_state=int(rng.integers(1e9)))
    th_o, nar_o = oof_base(sub, 5, "_aleat")      # dobras ao nível da seção (município parcialmente chegado)
    Xtr = sub.assign(th_base=th_o, n_arr_mun=nar_o * len(x) / len(sub))[FEATS]
    ytr = (sub.sw - th_o).values
    cont = x.groupby("mun").size()
    Xte = y.assign(th_base=th_y, n_arr_mun=y.mun.map(cont).fillna(0))[FEATS]
    m = novo_gbm()
    m.fit(Xtr, ytr, sample_weight=sub.n2.values)
    return m.predict(Xte)


# ----------------------------------------------------------------------------------------- incerteza
def escalas_erro(x, piso_uf, piso_nac, piso_mun):
    """Estima, com baseline fora-da-amostra por MUNICÍPIO, o quanto o swing erra de forma compartilhada."""
    sub = x if len(x) <= 150_000 else x.sample(150_000, random_state=1)
    th_o, _ = oof_base(sub, 5, "mun")
    r = sub.sw.values - th_o
    sub = sub.assign(r=r, rw=r * sub.n2.values)
    s_sec2 = (sub.r ** 2).mean()
    m = sub.groupby("mun").agg(n=("r", "size"), W=("n2", "sum"), RW=("rw", "sum"))
    m = m[m.n >= 2]
    rm = m.RW / m.W
    var_obs = np.average(rm ** 2, weights=m.W)
    ruido = np.average(s_sec2 / m.n, weights=m.W)
    s_mun = np.sqrt(max(var_obs - ruido, 0.0))
    s_sec = np.sqrt(s_sec2)
    s_mun = max(s_mun, piso_mun)
    # erro de ESTIMAR a média de swing com poucos municípios/seções chegados (some sozinho quando chega muita coisa)
    n_mun = x.mun.nunique()
    s_nac = np.sqrt(piso_nac ** 2 + s_mun ** 2 / max(n_mun, 1) + s_sec ** 2 / len(x))
    s_uf = np.sqrt(piso_uf ** 2 + s_mun ** 2 / max(n_mun / x.uf.nunique(), 1))
    return s_mun, s_sec, s_uf, s_nac


def simula(x, y, th, rt, esc, sims, rng, s_extra=0.0):
    """s_extra: incerteza adicional COMPARTILHADA por todas as seções que faltam (ex.: da correção de chegada)."""
    s_mun, s_sec, s_uf, s_nac = esc
    s_nac = np.sqrt(s_nac ** 2 + s_extra ** 2)
    n2h = y.valid1.values * rt
    l1 = y.l1.values
    mun_i, mun_n = pd.factorize(y.mun)
    uf_i, uf_n = pd.factorize(y.uf)
    base_a, base_n = x.a2.sum(), x.n2.sum()
    n_arr_uf = x.groupby("uf").size().reindex(uf_n).fillna(0).values
    # UF com pouca/nenhuma seção chegada: o swing dela é quase desconhecido -> sd cheio entre-UFs (tau_uf)
    s_uf_i = np.sqrt(s_uf ** 2 + TAU_UF ** 2 / (1 + n_arr_uf / 20.0))
    sh = np.empty(sims)
    for s in range(sims):
        e = (rng.normal(0, s_nac)
             + rng.normal(0, 1, len(uf_n))[uf_i] * s_uf_i[uf_i]
             + rng.normal(0, s_mun, len(mun_n))[mun_i]
             + rng.normal(0, s_sec, len(y)))      # ruído por seção (encolhe sozinho ao somar)
        p = desuav(ex(l1 + th + e), n2h)
        sh[s] = (base_a + (n2h * p).sum()) / (base_n + n2h.sum())
    return sh


# ----------------------------------------------------------------------------------------- painel legível
def painel(x, y, th, rt, p, na, nb):
    """Traduz o resultado em coisas que uma pessoa entende: votos de folga, o que precisaria acontecer para virar."""
    n2h = y.valid1.values * rt
    resto = n2h.sum()                                   # votos (A+B) que o modelo acha que ainda vêm
    a_resto = (n2h * desuav(ex(y.l1.values + th), n2h)).sum()
    A_tot, N_tot = x.a2.sum() + a_resto, x.n2.sum() + resto
    margem_votos = 2 * A_tot - N_tot                    # A - B, em votos (A+B)
    lider = na if margem_votos >= 0 else nb
    s_empate = (N_tot / 2 - x.a2.sum()) / max(resto, 1)   # share de A nos votos que faltam que empataria o total
    sh_a_resto = a_resto / max(resto, 1)
    if margem_votos >= 0:
        prev_atras, precisa_atras = 1 - sh_a_resto, 1 - s_empate
    else:
        prev_atras, precisa_atras = sh_a_resto, s_empate
    pl = p if margem_votos >= 0 else 1 - p          # p é P(A vence); converte para P(líder vence)
    pl = min(max(pl, 0.005), 0.995)
    return dict(lider=lider, margem_votos=margem_votos, resto_votos=resto, precisa_pct=100 * precisa_atras,
                prev_pct=100 * prev_atras, P_lider=pl)


def status_de(pl, lider, seq_99, n_seq):
    if pl < 0.75:
        return "INDEFINIDO"
    if pl < 0.95:
        return f"TENDÊNCIA {lider}"
    if pl < 0.99 or seq_99 < n_seq:
        return f"{lider} MUITO PROVÁVEL"
    return f"{lider} PRATICAMENTE DECIDIDO"


def frase(hora, rec, pn_share, sd, nome_a, pn, status):
    br = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")
    m = abs(pn["margem_votos"])
    resto = pn["resto_votos"]
    dif = f"{br(m / 1e6, 2)} {'milhão' if m < 2e6 else 'milhões'} de votos" if m >= 1e6 else f"{br(m / 1e3, 0)} mil votos"
    rest = f"{br(resto / 1e6, 1)} mi" if resto >= 1e5 else f"{br(resto / 1e3, 0)} mil"
    if pn["precisa_pct"] > 100:
        virar = "o outro lado não alcança nem levando 100% dos votos que faltam"
    else:
        virar = (f"o outro lado precisaria de {br(pn['precisa_pct'], 1)}% dos votos que faltam "
                 f"(o modelo prevê {br(pn['prev_pct'], 1)}%)")
    return (f"{hora} · {rec:.0%} das seções · {status} ({br(100 * pn['P_lider'])}%) · {pn['lider']} à frente por ~{dif} "
            f"({nome_a}: {br(100 * pn_share, 2)}% ±{br(196 * sd, 2)} pp) · faltam ~{rest} votos; para virar, {virar}")


# ----------------------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta")
    ap.add_argument("--a", default="13", help="número do candidato A (a probabilidade é de A vencer); "
                    "'auto' = detecta os 2 do 2º turno por UF (governador; rode UMA UF por vez com --ufs)")
    ap.add_argument("--b", default="22")
    ap.add_argument("--ufs", default="", help="lista separada por vírgula (teste rápido), ex: AP,PA,MA")
    ap.add_argument("--passo", type=int, default=5, help="minutos entre instantes do replay")
    ap.add_argument("--de", default="17:10")
    ap.add_argument("--ate", default="20:30")
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--sd-uf", type=float, default=0.010, help="piso de erro sistemático por UF (logit)")
    ap.add_argument("--sd-mun-min", type=float, default=0.03, help="piso do erro compartilhado por município (logit)")
    ap.add_argument("--sd-nac", type=float, default=0.010, help="piso de erro sistemático nacional (logit)")
    ap.add_argument("--sem-chegada", action="store_true", help="desliga a correção pela ordem de chegada")
    ap.add_argument("--cap-chegada", type=float, default=0.03, help="limite da correção de chegada por seção (logit)")
    ap.add_argument("--inc-chegada", type=float, default=0.5,
                    help="incerteza da correção de chegada, como fração do seu efeito médio (0.5 = ±50%%)")
    ap.add_argument("--gbm-min", type=int, default=20000, help="nº de seções chegadas antes do XGBoost entrar")
    ap.add_argument("--sem-gbm", action="store_true")
    ap.add_argument("--seq-decidido", type=int, default=3, help="instantes seguidos >=99%% para 'PRATICAMENTE DECIDIDO'")
    ap.add_argument("--piso-prob", type=float, default=0.005, help="P exibida nunca passa de 1-piso nem abaixo de piso")
    ap.add_argument("--min-secoes", type=int, default=300)
    ap.add_argument("--desloca", type=float, default=0.0,
                    help="contrafactual: soma este valor (logit) ao voto de A no 2º turno para simular corrida apertada. "
                         "Dica: ~0.04 logit = ~1 pp no share de A perto de 50%%")
    ap.add_argument("--nome-a", default="", help="nome de A nas frases (padrão: o número)")
    ap.add_argument("--nome-b", default="", help="nome de B nas frases (padrão: o número)")
    ap.add_argument("--saida", default="prob")
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    na, nb = (a.nome_a or a.a), (a.nome_b or a.b)

    ufs = set(filter(None, a.ufs.split(",")))
    if a.a.lower() == "auto" and len(ufs) != 1:
        sys.exit("--a auto é para disputas estaduais: use também --ufs com UMA UF (ex.: --ufs SP).")
    D = carregar(a.pasta, a.a, a.b, ufs, a.desloca)
    D["_aleat"] = np.arange(len(D))
    if DETECT:
        a.a, a.b = DETECT
        na, nb = (a.nome_a or a.a), (a.nome_b or a.b)
    truth = D.a2.sum() / D.n2.sum()
    P(f"motor do resíduo: {MOTOR} | correção de chegada: {'NÃO' if a.sem_chegada else 'sim'} | "
      f"XGBoost: {'NÃO' if a.sem_gbm else f'a partir de {a.gbm_min:,} seções, em média com o baseline'}")
    P(f"seções casadas: {len(D):,} | UFs: {D.uf.nunique()} | A={a.a} ({na}) vs B={a.b} ({nb})")
    P(f"resultado final de A (A entre A e B): {truth:.4f} | 1º turno nas mesmas seções: {D.a1.sum() / D.n1.sum():.4f}")
    dia = D.t2.dt.normalize().mode()[0]
    hh = lambda s: dia + pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:]))
    insts = pd.date_range(hh(a.de), hh(a.ate), freq=f"{a.passo}min")

    linhas, seq, lider_ant = [], 0, None
    for t in insts:
        arr = (D.t2 <= t).values
        if arr.sum() < a.min_secoes or arr.all():
            continue
        x, y = D[arr].copy(), D[~arr].copy()
        th_aj, rt_y = ajusta(x, y)
        esc = escalas_erro(x, a.sd_uf, a.sd_nac, a.sd_mun_min)
        if a.sem_chegada:
            cc, beta = np.zeros(len(y)), 0.0
        else:
            cc, beta = corr_chegada(x, y, a.cap_chegada)
        w_y = y.valid1.values * rt_y
        s_cheg = a.inc_chegada * abs(np.average(cc, weights=w_y)) if len(y) else 0.0
        th_base = th_aj + cc
        usa_gbm = (not a.sem_gbm) and len(x) >= a.gbm_min
        if usa_gbm:
            g = residuo_gbm(x, y, th_aj, rng)
            th_fin = th_aj + 0.5 * (cc + g)          # média de dois modelos: (baseline+chegada) e (baseline+XGBoost)
        else:
            th_fin = th_base
        r = {}
        for nome, th in (("base", th_base), ("final", th_fin)):
            sh = simula(x, y, th, rt_y, esc, a.sims, rng, s_extra=s_cheg)
            r[nome] = (sh.mean(), sh.std(), (sh > 0.5).mean())
        pn = painel(x, y, th_fin, rt_y, r["final"][2], na, nb)
        if pn["P_lider"] >= 0.99:
            seq = seq + 1 if pn["lider"] == lider_ant else 1
        else:
            seq = 0
        lider_ant = pn["lider"]
        stt = status_de(pn["P_lider"], pn["lider"], seq, a.seq_decidido)
        FRASES.append(frase(t.strftime("%H:%M"), arr.mean(), r["final"][0], r["final"][1], na, pn, stt))
        linhas.append(dict(hora=t.strftime("%H:%M"), recebidas=arr.mean(), status=stt, lider=pn["lider"],
                           P_lider=pn["P_lider"], margem_votos=pn["margem_votos"], resto_votos=pn["resto_votos"],
                           precisa_atras_pct=pn["precisa_pct"], prev_atras_pct=pn["prev_pct"],
                           bruto=x.a2.sum() / x.n2.sum(),
                           base_media=r["base"][0], base_sd=r["base"][1], base_P=r["base"][2],
                           final_media=r["final"][0], final_sd=r["final"][1], final_P=r["final"][2],
                           usa_gbm=usa_gbm, beta_chegada=beta, corr_chegada_media=np.average(cc, weights=w_y),
                           s_mun=esc[0], s_sec=esc[1]))
        l = linhas[-1]
        print(f"{l['hora']} {l['recebidas']:6.1%} | base {l['base_media']:.4f}±{l['base_sd']:.4f} P={l['base_P']:.3f} "
              f"| final {l['final_media']:.4f}±{l['final_sd']:.4f} P={l['final_P']:.3f} | chegada {l['corr_chegada_media']:+.4f} "
              f"| {stt}", flush=True)
    S = pd.DataFrame(linhas)
    if S.empty:
        sys.exit("Nenhum instante com seções suficientes (veja --de/--min-secoes).")
    for m in ("base", "final"):
        S[f"{m}_erro_pp"] = 100 * (S[f"{m}_media"] - truth)
        S[f"{m}_z"] = (truth - S[f"{m}_media"]) / S[f"{m}_sd"]
        S[f"{m}_P_exibida"] = S[f"{m}_P"].clip(a.piso_prob, 1 - a.piso_prob)
    S["verdade"] = truth
    S.to_csv(a.saida + ".csv", index=False)
    with open(a.saida + "_frases.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(FRASES) + "\n")

    P(f"\n=== SÉRIE (replay; verdade = {truth:.4f}) ===")
    cols = ["hora", "recebidas", "bruto", "base_media", "base_sd", "base_P_exibida", "final_media", "final_sd",
            "final_P_exibida", "base_erro_pp", "final_erro_pp", "corr_chegada_media"]
    P(S[cols].round(4).to_string(index=False))
    P("\n=== ERRO E CALIBRAÇÃO (base = baseline+chegada; final = com XGBoost quando entra) ===")
    for m in ("base", "final"):
        P(f"{m:>5}: |erro| médio {S[f'{m}_erro_pp'].abs().mean():.3f} pp | máx {S[f'{m}_erro_pp'].abs().max():.3f} pp | "
          f"erro médio (com sinal) {S[f'{m}_erro_pp'].mean():+.3f} pp | cobertura 95%: {(S[f'{m}_z'].abs() < 1.96).mean():.0%} | "
          f"desvio de z: {S[f'{m}_z'].std():.2f} (~1 = bem calibrado; >1 = otimista)")
    lado_errado = ((S.lider != (na if truth >= 0.5 else nb)) & (S.P_lider >= 0.75)).sum()
    certo = S[(S.lider == (na if truth >= 0.5 else nb)) & S.status.str.contains("PRATICAMENTE")]
    P(f"instantes com status apontando o LADO ERRADO (>=75%): {lado_errado}")
    P(f"primeiro 'PRATICAMENTE DECIDIDO' do lado certo: {certo.hora.iloc[0] if len(certo) else 'nunca'}"
      + (f" ({certo.recebidas.iloc[0]:.0%} das seções)" if len(certo) else ""))
    P("Obs.: z de instantes vizinhos é muito correlacionado; isto é UMA eleição, não calibração de verdade.")
    P("\n=== PAINEL (cole isto) ===")
    P("\n".join(FRASES[::3]))
    with open(a.saida + ".txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        xs = pd.to_datetime(S.hora, format="%H:%M")
        for m, c in (("base", "tab:blue"), ("final", "tab:red")):
            ax[0].plot(xs, 100 * S[f"{m}_media"], c=c, label=m)
            ax[0].fill_between(xs, 100 * (S[f"{m}_media"] - 1.96 * S[f"{m}_sd"]), 100 * (S[f"{m}_media"] + 1.96 * S[f"{m}_sd"]), color=c, alpha=.15)
            ax[1].plot(xs, 100 * S[f"{m}_P_exibida"], c=c, label=f"P({na} vence) {m}")
        ax[0].axhline(50, c="k", lw=.8); ax[0].axhline(100 * truth, c="g", ls="--", label="resultado final")
        ax[0].set_ylabel(f"% de {na} entre {na} e {nb}"); ax[0].legend()
        for yv in (75, 95, 99):
            ax[1].axhline(yv, c="gray", lw=.5, ls=":"); ax[1].axhline(100 - yv, c="gray", lw=.5, ls=":")
        ax[1].set_ylabel(f"P({na} vence) %"); ax[1].set_ylim(-2, 102); ax[1].legend()
        ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        plt.tight_layout(); plt.savefig(a.saida + ".png", dpi=130)
    except Exception as e:
        print("gráfico não gerado:", e)
    print(f"\ngravado: {a.saida}.csv, {a.saida}.png, {a.saida}.txt, {a.saida}_frases.txt")


if __name__ == "__main__":
    main()
