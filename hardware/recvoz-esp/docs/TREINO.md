# Treino — da gravação à base embarcada

> Como a base de locutores nasce e como se prova que ela presta, antes de
> ir pra flash. A classificação em si está em `CLASSIFICACAO.md`; o DSP em
> `MFCC.md`.

## 0. Mapa em 30 segundos

```
gravação (INMP441, 16 kHz) ──► audio_exemplo/locutor_<nome>_s<N>.wav
  ──► notebook: EDA + comparou 5 métodos ──► k-NN venceu (93.3%)
  ──► tools/export_mfcc.py ──► enroll_default.h (flash) + fixtures de teste
  ──► validação leave-one-out + escolha do τ ──► firmware classifica
```

## 1. Coleta de dados (protocolo)

- Mesmo mic da demo (**INMP441**), ~15 cm da boca, sala quieta.
- **~15–20 s por sessão, 2 sessões por locutor** (sessões diferentes
  capturam variação natural de voz/posição).
- Grave com `tools/grab_raw.py -p /dev/ttyACM0 -d 15` (projeto `inmp441_mic`).
- Nome: `audio_exemplo/locutor_<nome>_s<numero>.wav` (minúsculas).
  Arquivos fora da convenção são ignorados pelo export.
- Base atual: **ADRIAN 2 sessões + PEDRO 4 sessões, tudo no mic**.

## 2. O que o notebook provou

`eda_speaker_recognition.ipynb` fatia cada áudio em **segmentos de 2 s**
(125 frames), tira o **vetor médio 39-dim** por segmento e compara
5 classificadores no mesmo split (Monte Carlo, 15 seeds):

| Método | Acurácia |
|---|---|
| k-NN (k=3, cosseno, peso 1/d) | **93.3% ± 4.8%** ← escolhido |
| Centroide + cosseno | 86.7% ± 7.1% |
| MLP (64) | 90.8% ± 8.0% |
| GMM por locutor | 93.3% ± 6.8% |
| SVM (RBF) | 53.8% |

k-NN empata com o GMM na média (93.3%) com **menos desvio** — e é *lazy*
(cadastrar = armazenar vetores, sem treino) — perfeito pro enroll
embarcado. A queda vs a base antiga (99.4%) é esperada: tudo gravado no
mesmo mic, com variabilidade real de sessão. Equações em `CLASSIFICACAO.md` §3.

## 3. Export: do wav à flash (`tools/export_mfcc.py`)

```
para cada locutor:
  concatena sessões ──► filtra só blocos de 100 ms com RMS > 300 ──► pipeline MFCC
  ──► fatia em blocos de 125 frames ──► média por bloco ──► vetor 39-dim
```

- **Gate de voz:** só blocos acima do VAD viram protótipo — igual às
  janelas ao vivo (`capture_voiced`). Silêncio na base deslocava `d`
  pra 0.01+ (medido); com o gate, a base atual tem **10 Adrian + 14 Pedro**.
- Saídas:
  - `components/speaker_db/enroll_default.h` — base de fábrica (vai pra flash);
  - `components/mfcc/mfcc_tables.h` — mel CSR + SOS + Savitzky-Golay (só muda se o DSP mudar);
  - `tools/mfcc_test/*` — fixtures do teste de paridade;
  - `tools/mfcc_test/enroll_vectors.npz` — base p/ validação offline.
- Regenere com `.venv/bin/python tools/export_mfcc.py` (exige `librosa==1.0.0`).

## 4. Validação offline (gate antes de flashar)

**a) Paridade DSP** — `make -C tools/mfcc_hosttest run` tem que dar `PASS`:
C do firmware vs librosa no mesmo áudio (bandpass 1.6e-06, mel 5e-05,
**cosseno do vetor médio dist 6e-08**).

**b) Leave-one-out na base** — cada vetor classificado contra os demais
(k=3, sem c0):

```
mesma-voz máx 0.0119   outra-voz mín 0.0219   → gap limpo
τ = 0.017 (meio do gap) → 24/24, 0 rejeições
```

**c) Regra do τ:** tem que caber entre `same_max` e `other_min` com margem
dos dois lados. Sem gap (overlap), não existe τ bom — o remédio é dado
melhor (mais sessões, mesmo mic), não τ mágico. O τ final sempre se
confirma em sala (`docs/CALIBRACAO.md`).

## 5. Versões da base (fábrica × NVS)

| Camada | Onde | O quê |
|---|---|---|
| Fábrica | flash (`enroll_default.h`) | base do export; imutável sem reflash |
| Extras | NVS `recvoz` | até 3 locutores × 8 vetores (enroll ao vivo) |
| Override | bit `ovr` em NVS | enroll com nome de fábrica aposenta aqueles vetores |

`reset` apaga extras + override (volta pra fábrica). `list` mostra a base ativa.
