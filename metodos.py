#!/usr/bin/env python3
"""
Métodos ALTERNATIVOS de projeção, para mostrar lado a lado com o modelo principal (A = swing hierárquico + chegada).

  B. TRANSFERÊNCIA DE VOTOS: o swing de cada seção depende do PERFIL do 1º turno (share de A, votos de cada
     candidato eliminado, tamanho). Regressão ponderada nas seções já coletadas + hierarquia no resíduo.
     Hipótese: quem votou em X no 1º turno migra parecido em todo lugar.
  C. CONTAGEM RÁPIDA ESTRATIFICADA (estilo conteo rápido do México): estratos = UF x faixa de voto de A no 1º turno.
     Em cada estrato, estimador de razão com as seções coletadas; não usa swing seção a seção.
     Hipótese: dentro do estrato, a amostra representa o estrato.

Todas as funções recebem x (seções com 2º turno) e y (sem) no formato do modelo_prob.carregar / ao_vivo.
"""
import numpy as np
import pandas as pd
from math import erf, sqrt

import modelo_prob as M

N_OUTROS = 5          # quantos candidatos eliminados entram como variável no método B


def _phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def prep_perfil(D):
    """Cria as variáveis de perfil do 1º turno (s1_<n> = share do candidato n entre os válidos). Faz 1 vez."""
    c1 = [c for c in D.columns if c.startswith("c1_")]
    if not c1:
        return []
    top = D[c1].sum().sort_values(ascending=False).index[:N_OUTROS]
    cols = []
    for c in top:
        nome = "s1_" + c[3:]
        D[nome] = D[c] / D.valid1
        cols.append(nome)
    return cols


# ----------------------------------------------------------------------------------------- B
def metodo_B(x, y, esc, sims, rng, perfil):
    X = lambda d: np.column_stack([np.ones(len(d)), d.l1.values, d.lvalid.values] + [d[c].values for c in perfil])
    w = x.n2.values.astype(float)
    Xx = X(x)
    lam = 1e-3 * np.eye(Xx.shape[1]) * w.sum()                       # ridge leve (estabiliza com pouca amostra)
    lam[0, 0] = 0
    b = np.linalg.solve(Xx.T @ (Xx * w[:, None]) + lam, Xx.T @ (x.sw.values * w))
    th_r, rt = M.ajusta(x.assign(sw=x.sw.values - Xx @ b), y)            # o que o perfil não explica: hierarquia
    th = X(y) @ b + th_r
    sh = M.simula(x, y, th, rt, esc, sims, rng)
    return dict(nome="Transferência de votos", shareA=float(sh.mean()), lo=float(np.quantile(sh, .025)),
                hi=float(np.quantile(sh, .975)), PA=float((sh > 0.5).mean()))


# ----------------------------------------------------------------------------------------- C
def metodo_C(x, y, n_faixas=5, piso=0.0015, k_min=30):
    """Estimador de RAZÃO estratificado com variável auxiliar (o voto de cada candidato no 1º turno):
    em cada estrato,  votos2_A(faltam) = votos1_A(faltam) x [soma votos2_A / soma votos1_A](coletadas); idem B.
    Não usa logit, swing seção a seção, hierarquia nem correção de chegada."""
    D = pd.concat([x.assign(_x=1), y.assign(_x=0)])
    D["b1_"] = D.n1 - D.a1
    D["b2_"] = D.n2 - D.a2
    cortes = np.quantile(D.l1, np.linspace(0, 1, n_faixas + 1)[1:-1])
    D["estr"] = D.uf + "_" + pd.Series(np.searchsorted(cortes, D.l1.values), index=D.index).astype(str)
    D["nac"] = "BR"
    xs, ys = D[D._x == 1], D[D._x == 0].copy()
    ys["grp"] = None
    for chave in ("nac", "reg", "uf", "estr"):                          # do mais geral ao mais específico
        k = xs.groupby(chave).size()
        ok = set(k[k >= k_min].index)
        sel = ys[chave].isin(ok)
        ys.loc[sel, "grp"] = chave + ":" + ys.loc[sel, chave].astype(str)
    A_tot, B_tot = xs.a2.sum(), xs.b2_.sum()
    vA = vB = 0.0
    for grp, gy in ys.groupby("grp"):
        chave, key = grp.split(":", 1)
        gx = xs[xs[chave].astype(str) == key]
        n = len(gx)
        for c1, c2 in (("a1", "a2"), ("b1_", "b2_")):
            r = gx[c2].sum() / max(gx[c1].sum(), 1)
            X_y = gy[c1].sum()
            e = gx[c2] - r * gx[c1]
            v_r = (e ** 2).sum() / max(n - 1, 1) / n / max(gx[c1].mean(), 1e-9) ** 2
            if c2 == "a2":
                A_tot += r * X_y; vA += X_y ** 2 * v_r
            else:
                B_tot += r * X_y; vB += X_y ** 2 * v_r
    T = A_tot + B_tot
    share = A_tot / T
    var = (B_tot ** 2 * vA + A_tot ** 2 * vB) / T ** 4           # delta (sem covariância: conservador)
    frac_resto = (ys.a1.sum() + ys.b1_.sum()) / max(D.a1.sum() + D.b1_.sum(), 1)
    sd = sqrt(var + (piso * frac_resto) ** 2)
    return dict(nome="Razão estratificada (estilo contagem rápida)", shareA=float(share), lo=float(share - 1.96 * sd),
                hi=float(share + 1.96 * sd), PA=float(_phi((share - 0.5) / sd)) if sd > 0 else float(share > .5))


# ----------------------------------------------------------------------------------------- consenso
def consenso(metodos, na, nb, seq):
    """Status conservador e com linguagem neutra: só sobe de nível quando TODOS os métodos concordam.
    Faixas testadas nos replays de 2022 e 2018: <75% disputa aberta | 75–95% vantagem provável |
    95–99% forte tendência | >=99% nos três, por 3 atualizações seguidas: vencedor projetado."""
    lid = [na if m["PA"] >= 0.5 else nb for m in metodos]
    pl = [m["PA"] if l == na else 1 - m["PA"] for m, l in zip(metodos, lid)]
    if len(set(lid)) > 1:
        k = max(lid.count(na), lid.count(nb))
        quem = na if lid.count(na) >= lid.count(nb) else nb
        return f"Disputa aberta (métodos divergem: {k} de {len(metodos)} com {quem} à frente)", 0
    lider, pmin = lid[0], min(pl)
    seq = seq + 1 if pmin >= 0.99 else 0
    if pmin >= 0.99 and seq >= 3:
        return f"Vencedor projetado: {lider}", seq
    if pmin >= 0.95:
        return f"Forte tendência: {lider}", seq
    if pmin >= 0.75:
        return f"Vantagem provável: {lider}", seq
    return f"Disputa aberta ({lider} à frente nos três métodos, sem folga)", seq
