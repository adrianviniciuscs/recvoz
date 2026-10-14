# mfcc — MFCC 39-dim bit a bit igual ao notebook

Implementação em C puro (só libc + libm, sem dependências do IDF) da
cadeia de `eda_speaker_recognition.ipynb`:

```
int16 @16kHz -> /32768 -> Butter 80–7920 Hz (filtfilt) -> STFT 1024/256
Hann periódica, center + zero-pad -> |.|² -> mel128 slaney ->
10·log10 (ref 1.0, top_db 80) -> DCT-II ortho (13) ->
deltas Savitzky-Golay largura 9 -> 39-dim/frame
```

Janela de decisão: **32000 amostras (2 s) → 126 frames → vetor médio 39**
(o que o k-NN classifica na F2).

## Uso

```c
#include "mfcc.h"

static float m[39 * 126];
int nf = mfcc_process_window(janela_int16_2s, m); // 126
float mean[39];
mfcc_mean_vector(m, nf, mean);
```

Transiente de ~135 KB (malloc/free dentro da chamada).

## Paridade com o Python (gate da F1)

O **mesmo** `mfcc_dsp.c` é compilado no host com gcc e comparado contra
o librosa deste venv (`tools/export_mfcc.py` gera tabelas + esperados):

```bash
.venv/bin/python tools/export_mfcc.py   # gera mfcc_tables.h + tools/mfcc_test/*
make -C tools/mfcc_hosttest run         # PASS exigido
```

Resultado atual: bandpass 1.6e-06 · mel 5e-05 · **cos(vetor médio) dist 6e-08**.
O erro frame-a-frame (< 2.1) vem do piso de ruído do float32 em frames de
quase-silêncio — inofensivo p/ cosseno (documentado no hosttest).

## Supressão de ruído (opcional, `noise_sup.c`)

Subtração espectral em potência (Berouti), na mão, sem esp-dsp:

```
silêncio (VAD) ──► perfil N[k] ──► voz: P[k] = max(P[k] − 1.2·N[k], 0.05·P[k])
```

Entra entre `frame_power()` e `mel_apply()` (~2 KB de perfil, 513 sub+max
por frame). Calibração por lote de 0.5 s de silêncio passa pelo **mesmo**
filtfilt da voz (`mfcc_noise_calibrate()`), por isso o perfil casa.
Nasce **desligada** (paridade acima intacta); o `main` liga no boot e
auto-calibra no silêncio do idle (`ns`/`nscal` no console).

Teste próprio (sintético, determinístico): `make -C tools/ns_hosttest run`.
Cobre passthrough desligado, matemática, piso, EMA, paridade fim-a-fim com
ruído zero (bit-idêntica) e denoising funcional (seno+ruído branco:
distância ao limpo 0.0307 → 0.0223).

## Regenerar tabelas

`mfcc_tables.h` é **gerado** (mel CSR exato do librosa + SOS + zi +
pesos SG2). Só regenere se mudar algum parâmetro do notebook — e aí
re-treine tudo (F2):

```bash
.venv/bin/python tools/export_mfcc.py
```

Versão travada: **librosa 1.x** (o `delta` do 0.10 usava outra fórmula;
`requirements.txt` fixa `librosa==1.0.0`).
