# Projeção em tempo real do 2º turno a partir dos boletins de urna

Modelo de projeção eleitoral que roda durante a apuração. A cada lote de boletins de urna (BU)
divulgado pelo TSE, ele estima o resultado das seções que ainda não chegaram, quantifica a
incerteza por simulação e publica projeção, intervalo e probabilidade de vitória num painel web.

> **Não é resultado oficial.** O resultado oficial é publicado pela Justiça Eleitoral.

---

## O problema

A contagem parcial é uma amostra **não aleatória** do eleitorado: a ordem de chegada das seções
correlaciona com região, porte do município, urbanização e perfil de voto. Extrapolar a parcial
diretamente gera viés sistemático e grande no início da noite.

A estratégia é usar o **1º turno da mesma seção** como variável auxiliar. Ele é conhecido para
100% das seções antes de o 2º turno começar. O modelo não estima o voto em nível, e sim a
**mudança** entre turnos. Essa mudança varia muito menos entre seções e é muito mais previsível
a partir das seções vizinhas que já chegaram.

## Pipeline

```
TSE (cs.json, aux.json, *-bu.dat)
   │  coletor_tse.py ×4 máquinas (--parte k/n, hash estável da seção)
   ▼
/api/coleta (Vercel + Upstash Redis) ──► puxar_coleta.py ──► coleta/*.jsonl
                                                              │
oficial_tse.py (resultado agregado BR/UF) ────────────────────┤
                                                              ▼
                               ao_vivo.py: modelo + 3 métodos + consenso (a cada ~2 min)
                                                              │
                                                  publicar.py ▼
                                     painel (Vercel /api/estado e/ou Cloudflare Pages)
```

### Coleta e decodificação

- O coletor lê o índice de seções por UF (`cs.json`, cacheado ~50 s) e, para cada seção nova,
  baixa o `aux.json` e o arquivo `-bu.dat`.
- O BU é **ASN.1 BER**. O coletor decodifica com as especificações oficiais (`bu_v1.asn1`,
  `bu_v2.asn1`) e extrai os votos por cargo e o horário de recebimento.
- O decodificador foi validado contra os totais oficiais do 1º turno de 2026: 100% das seções
  amostradas batem.
- **Taxa e bloqueios.** Há um limitador de taxa por máquina, com recuo em 403/429. O coletor não
  repete requisições que deram 404, para não acionar o bloqueio de IP. A coleta prioriza a cobertura
  geográfica (UFs e municípios menos representados primeiro).
- **Distribuição.** A coleta é repartida entre máquinas por `crc32(seção) mod n`. A partição é
  pseudoaleatória e independente de geografia, então a perda de uma máquina reduz a amostra sem
  enviesá-la.
- **Tolerância a falhas.** Os arquivos de saída são rotativos e cada máquina retoma de onde parou.
  As máquinas remotas enviam lotes ao endpoint `/api/coleta`, protegido por token; a máquina central
  os puxa.

## Modelo principal: swing hierárquico com correção de chegada

**Notação.** Para cada seção $i$:

- $a_i$ = votos em A e $n_i$ = votos em A + B;
- o share é suavizado como $p_i = (a_i + 0{,}5)/(n_i + 1)$;
- o **swing** é $s_i = \text{logit}(p_i^{(2)}) - \text{logit}(p_i^{(1)})$.

**1. Encolhimento hierárquico (Bayes empírico).**

- A média ponderada do swing é estimada em quatro níveis: país → região → UF → município.
- Cada nível é puxado para o nível acima conforme o número de seções $n_g$ já chegadas:

$$\hat\theta_g = \hat\theta_{\text{pai}} + \frac{n_g}{n_g + k}\left(\bar s_g - \hat\theta_{\text{pai}}\right), \qquad k_{\text{reg}}=50,\ k_{\text{UF}}=20,\ k_{\text{mun}}=10$$

- A mesma estrutura estima a razão de comparecimento $n^{(2)}/\text{válidos}^{(1)}$.
- Uma seção pendente recebe o $\hat\theta$ do nível mais específico que tiver dados.

**2. Correção pela ordem de chegada.**

- Mesmo dentro de um município, as seções que chegam tarde diferem das que chegam cedo.
- A covariável é a ordem de chegada da seção no **1º turno** dentro da UF ($\text{ord}_1 \in [0,1]$).
  Ela é conhecida também para as seções que ainda não chegaram.
- A inclinação $\beta$ vem de uma regressão do swing em $\text{ord}_1$ **com efeito fixo de
  município**. É, portanto, identificada só por variação intramunicipal.
- A correção aplicada é $\beta(\text{ord}_1 - \overline{\text{ord}}_{\text{ref}})$, limitada a
  ±0,03 em logit.
- 50% do valor absoluto da correção entra como incerteza adicional compartilhada.

**3. Reconstrução do share.**

- A previsão é feita na escala suavizada e depois devolvida à escala crua com
  $\tilde p = \text{clip}\big((p(n+1) - 0{,}5)/n\big)$.
- Sem isso, haveria um viés pequeno e sistemático em direção a 50% nas seções com share extremo.

### Incerteza

A projeção nacional sai de Monte Carlo com **erros compartilhados em quatro níveis**:

$$\text{logit}\,\hat p_i = \text{logit}\,p_i^{(1)} + \hat\theta_i + \varepsilon_{\text{nac}} + \varepsilon_{\text{UF}(i)} + \varepsilon_{\text{mun}(i)} + \varepsilon_i$$

- **Como as escalas são estimadas.** Com um baseline *fora da amostra*, em validação cruzada
  agrupada por município (5 dobras), sobre as seções já chegadas.
- **Variância entre municípios.** É obtida por decomposição: variância observada das médias
  municipais menos o ruído de amostragem.
- **Componentes nacional e por UF.** Incluem o erro de estimar a média com poucos municípios
  chegados, que desaparece à medida que a amostra cresce.
- **Pisos.** O desvio nacional e o por UF têm piso de 0,01 em logit; o municipal, de 0,03.
- **UFs com pouca amostra.** Recebem um termo extra $\tau_{\text{UF}}^2/(1 + n_{\text{UF}}/20)$,
  com $\tau_{\text{UF}} = 0{,}04$, calibrado na decomposição de variância de 2022.
- **O resultado de cada simulação** é contado + previsto.
- **O que o painel mostra:** a probabilidade é a fração das simulações em que A vence, e o
  intervalo é o 2,5–97,5%.

O intervalo é **condicional ao modelo**. Ele não cobre erro da fonte oficial nem mudanças de
comportamento sem precedente nos dados históricos.

## Métodos alternativos

Os três métodos rodam em paralelo e têm hipóteses de identificação diferentes. A ideia é que um
erro estrutural de um deles apareça como divergência entre eles.

| método | hipótese | estimador |
|---|---|---|
| **A. Swing + chegada** | seções pendentes mudam como as vizinhas já chegadas, corrigido pela ordem de chegada | descrito acima |
| **B. Transferência de votos** | o swing depende do perfil do 1º turno (quem votou nos eliminados migra de forma parecida em todo lugar) | WLS com ridge do swing em $\text{logit}\,p^{(1)}$, $\log(\text{válidos})$ e shares dos 5 maiores candidatos eliminados; encolhimento hierárquico no resíduo; mesma simulação de A |
| **C. Razão estratificada** | dentro do estrato, a amostra representa o estrato | estimador de razão com variável auxiliar (votos de cada candidato no 1º turno), estratos UF × quintil de $\text{logit}\,p^{(1)}$, *fallback* estrato → UF → região → país com mínimo de 30 seções; variância pelo método delta + piso proporcional à fração pendente |

O método C é deliberadamente simples: sem logit, sem hierarquia e sem correção de chegada.
Funciona como âncora de estilo contagem rápida.

## Regra de anúncio

O status usa linguagem neutra e só sobe de nível quando **todos os métodos concordam**. A variável
de referência é a menor probabilidade entre os três métodos.

| status | condição |
|---|---|
| Disputa aberta | métodos divergem no líder, ou menor probabilidade < 75% |
| Vantagem provável | 75–95% |
| Forte tendência | 95–99% |
| **Vencedor projetado** | ≥ 99% nos três métodos, por **3 atualizações seguidas**, **e** ≥ 15% das seções apuradas no país, **e** ≥ 5% em cada região (exceto exterior) |

**Travas de publicação.** Antes de qualquer probabilidade aparecer no painel são exigidas
3.000 seções na amostra, 20 UFs e 100 seções por região. Essas travas vieram do backtest de 2018:
com poucas centenas de seções, um único método chegou a 98% para o lado errado.

O painel guarda um **histórico auditável** de cada mudança de status, com horário dos dados,
apuração, projeção e menor probabilidade.

## Validação

O backtest reproduz apurações passadas **minuto a minuto, na ordem real de chegada**, com o
mesmo código que roda ao vivo (`replay_ao_vivo.py`, `compara_metodos.py`).

- **2022 (Lula × Bolsonaro).**
  - A projeção ficou a menos de 0,1 ponto do resultado final desde as primeiras atualizações.
  - Nenhum anúncio para o lado errado.
- **2018 (Haddad × Bolsonaro).**
  - Os microdados não têm horário de recebimento do BU. A chegada foi reconstruída
    (`chegada_2018.py`): horário de encerramento convertido para o fuso de Brasília + atraso
    amostrado da distribuição de 2022 por município.
  - Os erros foram uma ordem de grandeza maiores, e o método B foi o melhor.
  - Os desvios padronizados ficaram abaixo de 1, então os intervalos **não** foram apertados.
  - A regra de consenso bloqueou um anúncio errado: num ponto da série, A estava em 98,5% e C em
    100% para o lado errado, enquanto B divergia.
- **Cenário contrafactual.** `--desloca` soma uma constante em logit ao voto do 2º turno em toda
  seção, para testar disputas apertadas com a mesma ordem de chegada.

### Por que não há gradient boosting no ao vivo

`modelo_prob.py` inclui um corretor de resíduo com XGBoost, treinado fora da amostra, usado apenas
em análise offline. Ele não entra no pipeline ao vivo por três razões:

- As primeiras seções são uma amostra enviesada, e um modelo flexível aprende padrões que não se
  transferem para as seções pendentes.
- A calibração da incerteza fica sem fundamento claro.
- O ganho aparece só quando os três métodos já convergiram.

## Arquivos

| arquivo | função |
|---|---|
| `extrair_bu.py` | microdados de BU do TSE (CSV/ZIP) → uma linha por seção; normaliza nomes de coluna entre anos |
| `chegada_2018.py` | reconstrução da ordem de chegada para 2018 |
| `modelo_prob.py` | modelo principal, escalas de erro, simulação, backtest e diagnósticos |
| `metodos.py` | métodos B e C e regra de consenso |
| `compara_metodos.py` | backtest comparando os três métodos |
| `replay_ao_vivo.py` | ensaio ponta a ponta: reproduz uma noite passada e publica como se fosse ao vivo |
| `coletor_tse.py` | coleta e decodificação ASN.1 dos BUs (`bu_v1.asn1`, `bu_v2.asn1`) |
| `puxar_coleta.py` | consolida na máquina central os lotes enviados pelas máquinas remotas |
| `oficial_tse.py` | resultado oficial agregado (Brasil, UFs e exterior) |
| `ao_vivo.py` | loop da noite da eleição |
| `publicar.py` | publica o estado (Vercel e/ou Cloudflare Pages) |

## Reproduzir um backtest

```bash
pip install -r requirements.txt          # xgboost é opcional
python extrair_bu.py pasta_com_os_zips_do_TSE --saida saida_bu
python compara_metodos.py saida_bu       # erro e probabilidade de cada método ao longo da noite
python replay_ao_vivo.py saida_bu --nome-a Lula --nome-b Bolsonaro --local estado.json --velocidade 50
```

Os microdados de boletim de urna são públicos no Portal de Dados Abertos do TSE e não estão neste
repositório. Os tokens de publicação e coleta são lidos de variáveis de ambiente (`PAINEL_TOKEN`,
`COLETA_TOKEN`) e nunca ficam versionados.
#   c o l e t o r - m o d e l o - e l e i c a o  
 