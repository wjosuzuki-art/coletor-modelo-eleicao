#!/usr/bin/env python3
"""
ENSAIO GERAL do painel: reproduz a noite de 2022 em velocidade acelerada e publica no site a cada passo,
como se fosse ao vivo. Serve para testar Vercel + Redis + painel HOJE, antes de domingo.

Uso (mesma pasta do modelo_prob.py e do publicar.py):
    python replay_ao_vivo.py saida_bu --nome-a Lula --nome-b Flávio --velocidade 20
    python replay_ao_vivo.py saida_bu --desloca -0.045 --velocidade 30      # cenário apertado
    python replay_ao_vivo.py saida_bu --local estado.json                     # sem internet: só grava o arquivo

--velocidade 20 = 20x mais rápido (um passo de 5 min do replay vira 15 s de verdade).
O formato do estado é o mesmo que o modo ao vivo de domingo vai publicar.
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

import metodos as ME
import modelo_prob as M
from publicar import publicar


def projecao_uf(x, y, th, rt):
    """Projeção do método A por UF: contado na amostra + previsto para o resto (só o centro)."""
    n2h = y.valid1.values * rt
    pa = M.desuav(M.ex(y.l1.values + th), n2h)
    Y = pd.DataFrame(dict(uf=y.uf.values, a=n2h * pa, n=n2h)).groupby("uf").sum()
    X = x.groupby("uf").agg(a=("a2", "sum"), n=("n2", "sum"))
    T = X.add(Y, fill_value=0)
    return ({u: round(float(r.a / r.n), 5) for u, r in T.iterrows() if r.n > 0},
            {u: int(r.n) for u, r in T.iterrows() if r.n > 0})


def oficial_simulado(D, arr):
    """No replay, as seções que já 'chegaram' são exatamente o que o TSE teria contado: simula a contagem oficial
    (% apurado e share de A por UF e Brasil) para o mapa e a tabela ficarem iguais aos de domingo."""
    x = D[arr]
    tot = D.groupby("uf").size()
    g = x.groupby("uf").agg(a=("a2", "sum"), n=("n2", "sum"), k=("a2", "size")).reindex(tot.index)
    res = {u: dict(pst=round(100 * float((r.k if r.k == r.k else 0) / tot[u]), 2),
                   share_a=(float(r.a / r.n) if r.n == r.n and r.n > 0 else None),
                   n=(int(r.n) if r.n == r.n else 0)) for u, r in g.iterrows()}
    res["BR"] = dict(pst=round(100 * float(arr.mean()), 2), share_a=float(x.a2.sum() / x.n2.sum()) if len(x) else None,
                     n=int(x.n2.sum()))
    return dict(por_uf=res, simulado=True)


VERSAO = "2026.10.25-r3"


def montar_estado(D, t, x, y, cfg, serie, seq_ant, rng, modo, apur=None):
    """apur = (fração apurada pelo TSE no país, {região: fração}). No replay, calculada das seções 'chegadas'."""
    th_aj, rt_y = M.ajusta(x, y)
    esc = M.escalas_erro(x, cfg.sd_uf, cfg.sd_nac, cfg.sd_mun_min)
    cc, beta = M.corr_chegada(x, y, cfg.cap_chegada)
    w_y = y.valid1.values * rt_y
    s_cheg = cfg.inc_chegada * abs(np.average(cc, weights=w_y)) if len(y) else 0.0
    th = th_aj + cc
    sh = M.simula(x, y, th, rt_y, esc, cfg.sims, rng, s_extra=s_cheg)
    pA, mA, sd = (sh > 0.5).mean(), sh.mean(), sh.std()
    pn = M.painel(x, y, th, rt_y, pA, cfg.na, cfg.nb)
    seq, lider_ant = seq_ant
    seq = (seq + 1 if pn["lider"] == lider_ant else 1) if pn["P_lider"] >= 0.99 else 0
    stt = M.status_de(pn["P_lider"], pn["lider"], seq, cfg.seq_decidido)
    rec = len(x) / len(D)
    hora = t.strftime("%H:%M")
    fr = M.frase(hora, rec, mA, sd, cfg.na, pn, stt)
    q = np.quantile(sh, [0.025, 0.975])
    # métodos alternativos + consenso (o painel mostra os três lado a lado)
    mA = dict(nome="Swing + chegada", shareA=float(mA), lo=float(q[0]), hi=float(q[1]), PA=float(pA))
    ms = [mA]
    for f in (lambda: ME.metodo_B(x, y, esc, cfg.sims, rng, getattr(cfg, "perfil", [])), lambda: ME.metodo_C(x, y)):
        try:
            ms.append(f())
        except Exception as e:                      # um método alternativo nunca derruba o painel
            print(f"[métodos] falhou: {e}")
    cons, cfg._seq_cons = ME.consenso(ms, cfg.na, cfg.nb, getattr(cfg, "_seq_cons", 0))
    # TRAVA: além da probabilidade, só anuncia vencedor com apuração mínima no país e em TODAS as regiões
    if apur is None:
        chegou = pd.Series(np.isin(D.index, x.index), index=D.index)
        apur = (float(chegou.mean()), chegou.groupby(D.reg).mean().to_dict())
    pct_tse, reg_tse = apur
    reg_min = min([v for r, v in (reg_tse or {}).items() if r != "Exterior"] or [0])
    min_pct, min_reg = getattr(cfg, "min_apur_decl", 0.15), getattr(cfg, "min_reg_decl", 0.05)
    trava = dict(prob=cons.startswith("Vencedor projetado"), seq=True,
                 apuracao=(pct_tse or 0) >= min_pct, regioes=reg_min >= min_reg,
                 min_pct=min_pct, min_reg=min_reg, pct=round(pct_tse or 0, 4), reg_min=round(reg_min, 4))
    if cons.startswith("Vencedor projetado") and not (trava["apuracao"] and trava["regioes"]):
        falta = []
        if not trava["apuracao"]: falta.append(f"{min_pct:.0%} de apuração")
        if not trava["regioes"]: falta.append(f"{min_reg:.0%} em todas as regiões")
        cons = f"Forte tendência: {cons.split(': ', 1)[1]} (aguardando {' e '.join(falta)})"
    hist = getattr(cfg, "_hist", [])
    st_h = cons if "aguardando" in cons else cons.split(" (")[0]
    if not hist or hist[-1]["status"] != st_h:
        hist.append(dict(hora=t.strftime("%H:%M"), pct=round(pct_tse or 0, 4), status=st_h,
                         shareA=round(float(mA["shareA"]), 5), P=round(float(min(max(m["PA"], 1 - m["PA"]) for m in ms)), 4)))
    cfg._hist = hist
    if cons.startswith("Vencedor projetado") and not getattr(cfg, "_primeira", None):
        cfg._primeira = dict(lider=cons.split(": ", 1)[1], hora=t.strftime("%H:%M"), pct=round(pct_tse or 0, 4))
    mA = mA["shareA"]
    serie.append(dict(hora=hora, pct=round(rec, 4), shareA=round(float(mA), 5), lo=round(float(q[0]), 5),
                      hi=round(float(q[1]), 5), PA=round(float(np.clip(pA, .005, .995)), 4),
                      bruto=round(float(x.a2.sum() / x.n2.sum()), 5),
                      **{f"PA_{k}": round(float(np.clip(m["PA"], .005, .995)), 4) for k, m in zip("ABC", ms)}))
    por_reg = (pd.Series(np.isin(D.index, x.index), index=D.index).groupby(D.reg).mean().round(4).to_dict())
    por_uf = (pd.Series(np.isin(D.index, x.index), index=D.index).groupby(D.uf).mean().round(4).to_dict())
    pj_uf, pj_votos = projecao_uf(x, y, th, rt_y)
    estado = dict(
        atualizado=datetime.now().isoformat(timespec="seconds"), modo=modo, hora_dados=hora,
        nomes=dict(a=cfg.na, b=cfg.nb),
        coleta=dict(secoes_recebidas=int(len(x)), secoes_total=int(len(D)), pct=round(rec, 4),
                    votos_contados=int(x.n2.sum()), contagem_bruta_A=round(float(x.a2.sum() / x.n2.sum()), 5),
                    por_regiao=por_reg, por_uf=por_uf),
        atual=dict(status=stt, lider=pn["lider"], P_lider=round(float(pn["P_lider"]), 4),
                   shareA=round(float(mA), 5), ic95=[round(float(q[0]), 5), round(float(q[1]), 5)],
                   margem_votos=int(pn["margem_votos"]), resto_votos=int(pn["resto_votos"]),
                   precisa_pct=round(float(pn["precisa_pct"]), 2), prev_pct=round(float(pn["prev_pct"]), 2),
                   frase=fr),
        serie=serie,
        consenso=cons,
        primeira_projecao=getattr(cfg, "_primeira", None),
        projecao_uf=pj_uf,
        projecao_uf_votos=pj_votos,
        trava=trava,
        historico=hist[-30:],
        snapshot=dict(id=f"{t:%Y-%m-%d %H:%M}", dados_ate=hora, gerado_em=datetime.now().strftime("%H:%M:%S"),
                      apuracao_tse=round(pct_tse or 0, 4), secoes_modelo=int(len(x)), modo=modo, versao=VERSAO),
        metodos=[dict(nome=m["nome"], shareA=round(m["shareA"], 5), lo=round(m["lo"], 5), hi=round(m["hi"], 5),
                      PA=round(float(np.clip(m["PA"], .005, .995)), 4)) for m in ms],
        aviso="Estimativa estatística não oficial, calculada a partir dos boletins de urna já divulgados pelo TSE.",
    )
    return estado, (seq, pn["lider"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta")
    ap.add_argument("--a", default="13"); ap.add_argument("--b", default="22")
    ap.add_argument("--nome-a", default="Lula"); ap.add_argument("--nome-b", default="Flávio")
    ap.add_argument("--desloca", type=float, default=0.0)
    ap.add_argument("--passo", type=int, default=5, help="minutos do replay entre publicações")
    ap.add_argument("--velocidade", type=float, default=20)
    ap.add_argument("--de", default="17:05"); ap.add_argument("--ate", default="20:30")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--local", default="", help="grava o estado neste arquivo .json (além de publicar)")
    cfg = ap.parse_args()
    cfg.sd_uf, cfg.sd_nac, cfg.sd_mun_min = 0.010, 0.010, 0.03
    cfg.cap_chegada, cfg.inc_chegada, cfg.seq_decidido = 0.03, 0.5, 3
    cfg.na, cfg.nb = cfg.nome_a, cfg.nome_b
    rng = np.random.default_rng(7)

    D = M.carregar(cfg.pasta, cfg.a, cfg.b, set(), cfg.desloca)
    cfg.perfil = ME.prep_perfil(D)
    dia = D.t2.dt.normalize().mode()[0]
    hh = lambda s: dia + pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:]))
    serie, seq = [], (0, None)
    print(f"replay de {len(D):,} seções | resultado final de A: {D.a2.sum() / D.n2.sum():.4f} | velocidade {cfg.velocidade}x")
    for t in pd.date_range(hh(cfg.de), hh(cfg.ate), freq=f"{cfg.passo}min"):
        t0 = time.time()
        arr = (D.t2 <= t).values
        if arr.sum() < 300 or arr.all():
            continue
        estado, seq = montar_estado(D, t, D[arr], D[~arr], cfg, serie, seq, rng, "ensaio (replay de eleição passada)")
        estado["oficial"] = oficial_simulado(D, arr)
        estado["serie"][-1]["oficial"] = estado["oficial"]["por_uf"]["BR"]["share_a"]
        if cfg.local:
            with open(cfg.local, "w", encoding="utf-8") as fh:
                json.dump(estado, fh, ensure_ascii=False)
        ok = publicar(estado)
        print(f"{estado['hora_dados']} {estado['coleta']['pct']:6.1%} | {estado['atual']['status']} "
              f"({estado['atual']['P_lider']:.1%}) | publicado: {ok}", flush=True)
        espera = cfg.passo * 60 / cfg.velocidade - (time.time() - t0)
        if espera > 0:
            time.sleep(espera)
    print("fim do ensaio.")


if __name__ == "__main__":
    main()
