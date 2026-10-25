#!/usr/bin/env python3
"""
Replay de 2022 comparando os 3 métodos (A swing+chegada | B transferência | C contagem rápida) e o CONSENSO.

Uso (mesma pasta de modelo_prob.py e metodos.py):
    python compara_metodos.py saida_bu
    python compara_metodos.py saida_bu --desloca -0.045 --saida cmp_flavio_leve
Cole o <saida>.txt no chat.
"""
import argparse

import numpy as np
import pandas as pd

import metodos as ME
import modelo_prob as M

OUT = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def metodo_A(x, y, esc, sims, rng):
    th, rt = M.ajusta(x, y)
    cc, _ = M.corr_chegada(x, y)
    w = y.valid1.values * rt
    s_ch = 0.5 * abs(np.average(cc, weights=w))
    sh = M.simula(x, y, th + cc, rt, esc, sims, rng, s_extra=s_ch)
    return dict(nome="Swing + chegada", shareA=float(sh.mean()), lo=float(np.quantile(sh, .025)),
                hi=float(np.quantile(sh, .975)), PA=float((sh > .5).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta")
    ap.add_argument("--a", default="13"); ap.add_argument("--b", default="22")
    ap.add_argument("--nome-a", default="Lula"); ap.add_argument("--nome-b", default="Flávio")
    ap.add_argument("--desloca", type=float, default=0.0)
    ap.add_argument("--passo", type=int, default=10)
    ap.add_argument("--de", default="17:10"); ap.add_argument("--ate", default="20:00")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--ufs", default="")
    ap.add_argument("--saida", default="cmp_metodos")
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    D = M.carregar(a.pasta, a.a, a.b, set(filter(None, a.ufs.split(","))), a.desloca)
    perfil = ME.prep_perfil(D)
    truth = D.a2.sum() / D.n2.sum()
    na, nb = a.nome_a, a.nome_b
    certo = na if truth >= .5 else nb
    P(f"seções {len(D):,} | verdade ({na}) {truth:.4f} | perfil do método B: {perfil}")
    dia = D.t2.dt.normalize().mode()[0]
    hh = lambda s: dia + pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:]))
    rows, seq = [], 0
    P(f"{'hora':>5} {'rec':>5} | {'A erro':>7} {'P(A)':>6} | {'B erro':>7} {'P(B)':>6} | {"C erro":>7} {"P(C)":>6} | consenso")
    for t in pd.date_range(hh(a.de), hh(a.ate), freq=f"{a.passo}min"):
        arr = (D.t2 <= t).values
        if arr.sum() < 300 or arr.all():
            continue
        x, y = D[arr], D[~arr]
        esc = M.escalas_erro(x, 0.010, 0.010, 0.03)
        ms = [metodo_A(x, y, esc, a.sims, rng), ME.metodo_B(x, y, esc, a.sims, rng, perfil), ME.metodo_C(x, y)]
        stt, seq = ME.consenso(ms, na, nb, seq)
        r = dict(hora=t.strftime("%H:%M"), rec=arr.mean(), consenso=stt)
        for k, m in zip("ABC", ms):
            r[f"{k}_erro"] = 100 * (m["shareA"] - truth)
            r[f"{k}_P"] = m["PA"]
            r[f"{k}_z"] = (truth - m["shareA"]) / max((m["hi"] - m["lo"]) / 3.92, 1e-9)
        rows.append(r)
        P(f"{r['hora']:>5} {r['rec']:5.0%} | {r['A_erro']:+7.3f} {r['A_P']:6.3f} | {r['B_erro']:+7.3f} {r['B_P']:6.3f} | "
          f"{r['C_erro']:+7.3f} {r['C_P']:6.3f} | {stt}")
    S = pd.DataFrame(rows)
    P("\n=== RESUMO (erro em pp do share final; z ~ N(0,1) se o intervalo estiver calibrado) ===")
    for k, nome in zip("ABC", ("A swing+chegada", "B transferência", "C razão estratif.")):
        pl = np.where(S[f"{k}_P"] >= .5, S[f"{k}_P"], 1 - S[f"{k}_P"])
        lado = np.where(S[f"{k}_P"] >= .5, na, nb)
        errado = ((lado != certo) & (pl >= .75)).sum()
        dec = S[(lado == certo) & (pl >= .99)]
        P(f"{nome:>20}: |erro| médio {S[f'{k}_erro'].abs().mean():.3f} | máx {S[f'{k}_erro'].abs().max():.3f} | "
          f"viés {S[f'{k}_erro'].mean():+.3f} | desvio de z {S[f'{k}_z'].std():.2f} | lado errado >=75%: {errado} | "
          f"1º >=99% certo: {dec.hora.iloc[0] if len(dec) else 'nunca'}")
    dc = S[S.consenso.str.startswith("Vencedor projetado") & S.consenso.str.endswith(certo)]
    err_c = S[S.consenso.str.contains("Vencedor|Forte|Vantagem") & ~S.consenso.str.contains(certo)]
    P(f"{'CONSENSO':>20}: 1º 'Vencedor projetado': {dc.hora.iloc[0] if len(dc) else 'nunca'} | "
      f"instantes apontando o lado errado: {len(err_c)}")
    with open(a.saida + ".txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    print(f"\ngravado: {a.saida}.txt")


if __name__ == "__main__":
    main()
