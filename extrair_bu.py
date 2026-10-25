#!/usr/bin/env python3
"""
Extrai do Boletim de Urna do TSE (arquivos bweb_<turno>_<UF>_<data>.zip) só o que o
modelo precisa: uma linha por seção, com votos por candidato e horários do BU.

Uso:
    python extrair_bu.py bweb_2t_SP_301020221800.zip            # um arquivo
    python extrair_bu.py pasta_com_os_zips --saida resultado    # todos os .zip/.csv da pasta

Saídas, por arquivo de entrada (prefixo = nome do arquivo, ou --saida):
    <prefixo>_secoes.csv   uma linha por seção (votos v_<numero>, aptos, horários)
    <prefixo>_curva.csv    % acumulado de seções e de eleitores aptos por minuto de chegada

Notas sobre os dados reais (conferidas num arquivo de 2022):
  - os vazios vêm escritos como #NULO# e #NE# (com # no final);
  - votos vazios (candidato sem voto na seção) viram 0 nas colunas v_*;
  - DS_TIPO_URNA veio como "APURADA" (parece status da seção, não tipo de urna).
"""
import argparse
import os
import sys
import zipfile

import pandas as pd

COLS = [
    "NR_TURNO", "SG_UF", "CD_MUNICIPIO", "NM_MUNICIPIO", "NR_ZONA", "NR_SECAO",
    "NR_LOCAL_VOTACAO", "DS_CARGO_PERGUNTA", "NR_VOTAVEL", "QT_VOTOS",
    "QT_APTOS", "QT_COMPARECIMENTO", "QT_ABSTENCOES",
    "DT_BU_RECEBIDO", "DT_EMISSAO_BU", "DT_ENCERRAMENTO",
    "DS_AGREGADAS", "DS_TIPO_URNA",
]
CHAVE = ["NR_TURNO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
META = [
    "NM_MUNICIPIO", "NR_LOCAL_VOTACAO", "QT_APTOS", "QT_COMPARECIMENTO",
    "QT_ABSTENCOES", "DT_BU_RECEBIDO", "DT_EMISSAO_BU", "DT_ENCERRAMENTO",
    "DS_AGREGADAS", "DS_TIPO_URNA",
]
DATAS = ["DT_BU_RECEBIDO", "DT_EMISSAO_BU", "DT_ENCERRAMENTO"]
# nomes de coluna que mudaram entre eleições (2018 tem "SG_ UF" com espaço, "QT_COMPARECIDOS" etc.)
ALIAS = {"QT_COMPARECIDOS": "QT_COMPARECIMENTO", "QT_COMPARECIMENTOS": "QT_COMPARECIMENTO", "UF": "SG_UF"}


def norm(c):
    c = str(c).replace("\ufeff", "").replace(" ", "").strip().upper()
    return ALIAS.get(c, c)
VAZIOS = {"#NULO#": pd.NA, "#NE#": pd.NA, "#NULO": pd.NA, "#NE": pd.NA}


def abrir(caminho):
    """Devolve algo que o pandas sabe ler: caminho do csv ou o csv dentro do zip."""
    if not caminho.lower().endswith(".zip"):
        return caminho
    zf = zipfile.ZipFile(caminho)
    csvs = [i for i in zf.infolist() if i.filename.lower().endswith(".csv")]
    if not csvs:
        sys.exit(f"Nenhum .csv dentro de {caminho}.")
    maior = max(csvs, key=lambda i: i.file_size)
    if len(csvs) > 1:
        print(f"Vários CSVs no zip; usando o maior: {maior.filename}")
    return zf.open(maior)


def para_data(serie):
    try:
        return pd.to_datetime(serie, dayfirst=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):  # pandas antigo, sem format="mixed"
        return pd.to_datetime(serie, dayfirst=True, errors="coerce")


def processar(arquivo, saida, cargo, chunk):
    partes, vistos, total = [], set(), 0
    leitor = pd.read_csv(
        abrir(arquivo), sep=";", quotechar='"', encoding="latin-1", dtype=str,
        usecols=lambda c: norm(c) in COLS, chunksize=chunk,
    )
    for ch in leitor:
        ch.columns = [norm(c) for c in ch.columns]
        total += len(ch)
        if "DS_CARGO_PERGUNTA" not in ch.columns:
            sys.exit("Coluna DS_CARGO_PERGUNTA não existe neste arquivo. Colunas: " + ", ".join(ch.columns))
        vistos.update(ch["DS_CARGO_PERGUNTA"].dropna().unique())
        partes.append(ch[ch["DS_CARGO_PERGUNTA"].str.strip().str.lower() == cargo.lower()])
        print(f"\rlinhas lidas: {total:,}", end="", flush=True)
    print()

    df = pd.concat(partes, ignore_index=True)
    if df.empty:
        print(f"Nenhuma linha com cargo '{cargo}'. Cargos encontrados: {sorted(vistos)}")
        return

    df = df.replace(VAZIOS)
    df["QT_VOTOS"] = pd.to_numeric(df["QT_VOTOS"], errors="coerce")

    falta = [c for c in CHAVE + ["NR_VOTAVEL", "QT_VOTOS"] if c not in df.columns]
    if falta:
        sys.exit(f"Faltam colunas {falta}. Colunas lidas: {list(df.columns)}")
    dup = int(df.duplicated(CHAVE + ["NR_VOTAVEL"]).sum())
    votos = df.pivot_table(index=CHAVE, columns="NR_VOTAVEL", values="QT_VOTOS", aggfunc="first")
    votos.columns = [f"v_{c}" for c in votos.columns]
    votos = votos.fillna(0)
    meta = df.drop_duplicates(CHAVE).set_index(CHAVE)[[c for c in META if c in df.columns]]
    sec = meta.join(votos).reset_index()

    for c in ["QT_APTOS", "QT_COMPARECIMENTO", "QT_ABSTENCOES"]:
        if c in sec:
            sec[c] = pd.to_numeric(sec[c], errors="coerce")
    for c in DATAS:
        if c in sec:
            sec[c + "_ts"] = para_data(sec[c])

    print("\n=== DIAGNÓSTICO ===")
    print(f"turno(s): {sorted(sec['NR_TURNO'].unique())} | UF(s): {sorted(sec['SG_UF'].unique())}")
    print(f"linhas lidas: {total:,} | linhas do cargo: {len(df):,} | seções: {len(sec):,}")
    print(f"linhas duplicadas (seção x votável): {dup}")
    if "DS_AGREGADAS" in sec:
        print(f"seções com DS_AGREGADAS preenchido: {int(sec['DS_AGREGADAS'].notna().sum()):,}")
    for c in DATAS:
        if c not in sec:
            print(f"{c}: coluna ausente")
            continue
        ts = sec[c + "_ts"]
        print(f"\n{c}: exemplo bruto = {sec[c].dropna().iloc[0] if sec[c].notna().any() else None!r}")
        print(f"  interpretadas: {ts.notna().mean():.1%}", end="")
        if ts.notna().any():
            horas = ts.dropna().dt.strftime("%H:%M:%S").nunique()
            print(f" | de {ts.min()} até {ts.max()} | horários distintos: {horas}")
        else:
            print()

    sec.to_csv(f"{saida}_secoes.csv", index=False)
    print(f"\ngravado: {saida}_secoes.csv")

    if "DT_BU_RECEBIDO_ts" in sec:
        t = sec.dropna(subset=["DT_BU_RECEBIDO_ts"])
        if t["DT_BU_RECEBIDO_ts"].dt.strftime("%H:%M:%S").nunique() > 1:
            t = t.assign(minuto=t["DT_BU_RECEBIDO_ts"].dt.floor("min"))
            g = t.groupby("minuto").agg(secoes=("NR_SECAO", "size"), aptos=("QT_APTOS", "sum")).sort_index()
            g["pct_secoes"] = g["secoes"].cumsum() / len(sec) * 100
            g["pct_aptos"] = g["aptos"].cumsum() / sec["QT_APTOS"].sum() * 100
            g.to_csv(f"{saida}_curva.csv")
            print(f"gravado: {saida}_curva.csv ({len(g)} minutos)")
        else:
            print("DT_BU_RECEBIDO não tem horário distinto: curva de chegada não gerada.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", help="arquivo .zip/.csv do boletim de urna, ou pasta com vários")
    ap.add_argument("--cargo", default="Presidente", help="texto de DS_CARGO_PERGUNTA (padrão: Presidente)")
    ap.add_argument("--saida", default=None,
                    help="prefixo de saída (arquivo único) ou pasta de saída (modo pasta). Padrão: nome do arquivo")
    ap.add_argument("--chunk", type=int, default=500_000, help="linhas por pedaço")
    a = ap.parse_args()

    if os.path.isdir(a.entrada):
        arquivos = sorted(
            os.path.join(a.entrada, f) for f in os.listdir(a.entrada)
            if f.lower().endswith((".zip", ".csv"))
        )
        if not arquivos:
            sys.exit("Nenhum .zip/.csv na pasta.")
        pasta_saida = a.saida or "saida_bu"
        os.makedirs(pasta_saida, exist_ok=True)
        for n, f in enumerate(arquivos, 1):
            nome = os.path.splitext(os.path.basename(f))[0]
            print(f"\n##### [{n}/{len(arquivos)}] {f}")
            processar(f, os.path.join(pasta_saida, nome), a.cargo, a.chunk)
    else:
        nome = os.path.splitext(os.path.basename(a.entrada))[0]
        processar(a.entrada, a.saida or nome, a.cargo, a.chunk)


if __name__ == "__main__":
    main()
