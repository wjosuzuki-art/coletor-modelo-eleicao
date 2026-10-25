#!/usr/bin/env python3
"""
2018 não traz a hora de RECEBIMENTO do BU, só a de ENCERRAMENTO da urna. Este script cria uma hora de chegada
"sintética, mas realista":   chegada = encerramento (2018) + atraso sorteado da distribuição REAL de 2022
do mesmo município e do mesmo turno (se o município não existir em 2022: a da UF).

Uso:
    python chegada_2018.py ..\\bu2018 C:\\git\\eleicao\\saida_bu ..\\bu2018_sint
depois rode o compara_metodos.py em ..\\bu2018_sint.

Fuso: em 2018 o horário de votação era o LOCAL (Acre fechava às 19h de Brasília); o script converte.
Limite: a ORDEM de chegada dentro do município fica parcialmente aleatória; então o teste da correção de chegada
em 2018 é fraco. O resto (swing, calibração, métodos B e C) vale normalmente.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

if len(sys.argv) != 4:
    sys.exit(__doc__)
p2018, p2022, saida = sys.argv[1:]
os.makedirs(saida, exist_ok=True)
rng = np.random.default_rng(2018)
# horas a SOMAR ao horário local de 2018 para virar horário de Brasília (28/10/2018, sem horário de verão)
FUSO = {"AC": 2, "AM": 1, "RR": 1, "RO": 1, "MT": 1, "MS": 1}

# atrasos reais de 2022 (minutos), por turno x município e turno x UF
lag = []
for f in glob.glob(os.path.join(p2022, "*_secoes.csv")):
    d = pd.read_csv(f, usecols=lambda c: c in ("NR_TURNO", "SG_UF", "CD_MUNICIPIO", "DT_BU_RECEBIDO_ts", "DT_ENCERRAMENTO_ts"))
    if "DT_ENCERRAMENTO_ts" not in d:
        continue
    m = (pd.to_datetime(d.DT_BU_RECEBIDO_ts) - pd.to_datetime(d.DT_ENCERRAMENTO_ts)).dt.total_seconds() / 60
    d["lag"] = m
    lag.append(d[(m > 0) & (m < 600)][["NR_TURNO", "SG_UF", "CD_MUNICIPIO", "lag"]])
L = pd.concat(lag)
por_mun = {k: g.lag.values for k, g in L.groupby(["NR_TURNO", "SG_UF", "CD_MUNICIPIO"])}
por_uf = {k: g.lag.values for k, g in L.groupby(["NR_TURNO", "SG_UF"])}
print(f"atrasos de 2022: {len(L):,} seções | mediana {L.lag.median():.0f} min")

for f in sorted(glob.glob(os.path.join(p2018, "*_secoes.csv"))):
    d = pd.read_csv(f)
    if "DT_ENCERRAMENTO_ts" not in d:
        print("sem DT_ENCERRAMENTO:", f); continue
    # em 2018 cada UF votou no HORÁRIO LOCAL (em 2022 foi tudo no horário de Brasília): converte para Brasília
    enc = pd.to_datetime(d.DT_ENCERRAMENTO_ts) + pd.to_timedelta(d.SG_UF.map(FUSO).fillna(0), unit="h")
    lags = np.empty(len(d))
    for i, (t, uf, mu) in enumerate(zip(d.NR_TURNO, d.SG_UF, d.CD_MUNICIPIO)):
        arr = por_mun.get((t, uf, mu))
        if arr is None or len(arr) < 3:
            arr = por_uf.get((t, uf))
        lags[i] = rng.choice(arr) if arr is not None else 30.0
    d["DT_BU_RECEBIDO_ts"] = enc + pd.to_timedelta(lags, unit="m")
    d.to_csv(os.path.join(saida, os.path.basename(f)), index=False)
    print(f"{os.path.basename(f)}: {len(d):,} seções | chegada sintética {d.DT_BU_RECEBIDO_ts.min():%H:%M}–{d.DT_BU_RECEBIDO_ts.max():%H:%M}")
print("pronto ->", saida)
