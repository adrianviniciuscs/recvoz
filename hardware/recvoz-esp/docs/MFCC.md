# MFCC 39-dim no firmware — especificação e implementação

> **Invariante deste projeto:** o C em `components/mfcc/mfcc_dsp.c` produz
> **numericamente o mesmo** que `eda_speaker_recognition.ipynb` no mesmo áudio.
> Qualquer mudança de parâmetro aqui exige re-treino (F2) e novo `PASS` no
> teste de paridade. Versão travada: **librosa 1.0.0** (ver §8).

## 0. Parâmetros (idênticos ao notebook — não mexer)

| Símbolo | Valor | Origem |
|---|---|---|
| `sr` | 16000 Hz | `TARGET_SR` |
| Banda | 80 – 7920 Hz | Butter 5ª, `[80/8000, min(8000/8000, 0.99)]` |
| `n_fft` / `hop` / janela | 1024 / 256 / Hann **periódica** | `N_FFT, HOP_LENGTH` |
| Mel | 128 bandas **Slaney**, `fmin=0`, `fmax=8000` | default `melspectrogram` |
| Log | `10·log10`, `ref=1.0`, `amin=1e-10`, `top_db=80` | default `power_to_db` |
| DCT | tipo II **ortonormal**, 13 primeiros | default `mfcc` |
| Deltas | Savitzky-Golay, largura 9, ordens 1 e 2 | default `delta` |
| Segmento | média temporal → vetor 39 | `mean_vecs` |

Janela de decisão: **32000 amostras (2 s) → 126 frames → média 39-dim**.

## 1. Visão geral da cadeia

```
int16 @16k ──► /32768 ──► passa-banda (filtfilt) ──► STFT 1024/256 ──► |·|²
  ──► mel128 ──► 10·log10 ──► DCT13 ──► Δ SG ──► Δ² SG ──► empilha 39 ──► média
```

Cada estágio abaixo traz a equação replicada e a decisão de implementação.

## 2. Pré-processamento

### 2.1 int16 → float

```c
x[i] = pcm[i] / 32768.0f;
```

Igual ao `soundfile` (`dtype="float32"`): PCM 16-bit escala cheia = ±1.0.

### 2.2 Passa-banda Butterworth + filtfilt

Filtro Butterworth passa-banda de ordem 5, cantos 80 Hz e 7920 Hz:

```python
b, a = signal.butter(5, [80/8000, 0.99], btype="bandpass")  # ordem total 10
y = signal.filtfilt(b, a, x)                                 # fase zero
```

`filtfilt` filtra ida **e** volta: a resposta em magnitude é elevada ao
quadrado e a fase é zerada — por isso o notebook o usa (não distorce a
posição temporal das formantes). O firmware replica o `sosfiltfilt` do
scipy exatamente:

1. **SOS:** 5 seções de 2ª ordem (`butter(..., output="sos")`), coeficientes
   embutidos em `MFCC_SOS` (`mfcc_tables.h`, gerado).
2. **Extensão ímpar** (`padtype="odd"`, `edge = 33`):
   `ntaps = 2·5+1 = 11`, `edge = 3·ntaps = 33`, e

   ```c
   ext[i]       = 2·x[0]   - x[edge-i]      // i = 0..edge-1
   ext[n+e+i]   = 2·x[n-1] - x[n-2-i]
   ```
3. **Condições iniciais de regime** (`sosfilt_zi`, embutidas em
   `MFCC_SOS_ZI`), escaladas pela 1ª amostra de cada passada:
   `zi·ext[0]`, como o scipy faz.
4. Filtragem **DF-II transposta** por seção e amostra:

   ```c
   out    = b0·v + d0
   d0     = b1·v − a1·out + d1
   d1     = b2·v − a2·out
   ```
5. Inverte o sinal, repete 2–4, inverte de novo, remove a extensão.

Paridade medida: **erro abs máx 1.6e-06** (ruído float32 vs float64).

## 3. STFT

### 3.1 Enquadramento (center + zero-pad)

Como `center=True, pad_mode="constant"` (padrão do `melspectrogram` no
librosa 1.x): 512 zeros de cada lado. Frame `t` centrado em `t·256`:

```
frame t = x[t·256 − 512 … t·256 + 511],  fora de [0, N) vale 0
```

Para N = 32000: `1 + 32000/256 = 126 frames`.

### 3.2 Janela de Hann periódica

$$w[n] = 0.5 - 0.5\cos(2\pi n/1024), \quad n = 0..1023$$

Periódica (`fftbins=True`, denominador N e não N−1), igual a
`get_window("hann", 1024)`. Ganho coerente 0.5.

### 3.3 FFT sem escala + potência

FFT radix-2 própria (`fft_fwd`, iterativa com bit-reversal), **sem
normalização 1/N** — igual a `scipy.fft.rfft` (norma `"backward"`):

$$X[k] = \sum_n x[n]\cdot w[n]\cdot e^{-j2\pi kn/1024}, \quad P[k] = Re^2 + Im^2, \quad k = 0..512$$

Verificado com senoide no bin 50: `|X| = 256 = N/4` (Hann 0.5 × N/2),
igual ao librosa. Motivo de FFT própria em vez de `esp-dsp`: o **mesmo**
`.c` compila no host (gcc) e no S3 — o teste de paridade exige isso.

## 4. Banco mel Slaney

Escala Slaney (`htk=False`): linear abaixo de 1000 Hz, log acima:

$$\mathrm{mel}(f) = \begin{cases} 3f/200 & f < 1000 \\ 15 + \dfrac{\ln(f/1000)}{\ln(6.4)/27} & f \geq 1000 \end{cases}$$

128 triângulos entre 0 e 8000 Hz com **normalização de área**
($\mathrm{enorm} = 2/(f[m+1] - f[m-1])$). Detalhe que quebrou a 1ª versão:
os triângulos amostrados **não zeram nos bins de borda**
(ex: filtro 7 vale `0.01499` no bin 11, não 0) — por isso embutimos os
**pesos exatos** em CSR (`MFCC_MEL_START/LEN/W`, maior filtro com
24 bins, tabela ≈ 3 KB) em vez de reconstruir rampas. Erro após a
correção: **5e-05**.

## 5. Log + piso

$$L[m] = 10\log_{10}(\max(E[m], 10^{-10})), \quad L[m] = \max(L[m], \max(L) - 80)$$

`ref=1.0` (dB absoluto) e `top_db=80` (padrão `power_to_db`).

## 6. DCT-II ortonormal → 13 MFCC

$$s[k] = \sum_{n=0}^{127} L[n]\cos\left(\frac{\pi(2n+1)k}{256}\right), \quad \mathrm{MFCC}[k] = s[k]\cdot\begin{cases} \sqrt{1/128} & k = 0 \\ \sqrt{2/128} & k \geq 1 \end{cases}$$

Equivale a `scipy.fft.dct(type=2, norm="ortho")`. Atenção ao c0:
como o log é absoluto, qualquer ganho global desloca **só o c0**
($\Delta c_0 = \Delta\mathrm{dB}\cdot\sqrt{128}$) — a FFT sem escala (§3.3) existe por causa dele.

## 7. Deltas Savitzky-Golay (largura 9)

> ⚠️ Armadilha de versão: no librosa 0.10 o `delta` era regressão linear;
> no **1.x é `savgol_filter(data, 9, deriv=order)`**. O notebook e o
> firmware usam 1.x.

Para largura 9 (`half = 4`), a derivada do ajuste polinomial é
**constante na janela** (reta → derivada 1ª constante; parábola →
derivada 2ª constante), logo as bordas usam os mesmos pesos nas
9 1ªs/últimas janelas:

$$\Delta^1[t] = \sum_{i=-4}^{4} \frac{i}{60}\cdot C[t+i], \qquad \Delta^2[t] = \sum_{i=-4}^{4} \mathrm{SG2}[i]\cdot C[t+i]$$

(denominador $2\sum i^2 = 60$; $\mathrm{SG2} = \mathrm{savgol}(9, \mathrm{poly2}, \mathrm{deriv2})$).

`SG2 = [0.0606, 0.0152, −0.0173, −0.0368, −0.0433, −0.0368, −0.0173, 0.0152, 0.0606]`
(simétrico; embutido em `MFCC_SG2`). Erros medidos: Δ 0.16, Δ² 0.15
(resíduo do c0, irrelevante p/ cosseno).

## 8. Vetor médio + decisão (ponte p/ F2)

$$v[d] = \frac{1}{126}\sum_t M[d,t], \quad d = 0..38, \qquad \mathrm{dist}(a,b) = 1 - \frac{a\cdot b}{|a||b|} \quad \text{(cosseno, igual ao sklearn)}$$

k-NN `k=3, metric="cosine", weights="distance"` vive em `speaker_db`.

## 9. Paridade medida (gate da F1)

`make -C tools/mfcc_hosttest run` — 2 s de voz do fixture `test_pcm.bin` (caminho int16
idêntico ao device):

| Estágio | Erro abs máx | Veredito |
|---|---|---|
| passa-banda | 1.6e-06 | cravado |
| potência (frame 37) | 9.8e-04 (pico 20.5) | cravado |
| mel | 5e-05 | cravado |
| vetor médio 39 | 0.05 | ok |
| **cos(vetor C, vetor ref)** | **dist 6e-08** | **gate real: PASS** |
| MFCC frame-a-frame | < 2.1 (1 frame) | explicado abaixo |

O outlier (frame 14, d=1..3, −2.08) é **piso de ruído float32 em
quase-silêncio**: potência máx 0.02 (contra 20 no frame 37) → a banda 0
difere 16 dB (−95.9 vs −79.6) → a DCT espalha `16.4·cos ≈ 2.08` nos
coefs. A média/cosseno — o que o k-NN usa — não sente (6e-08).
Limiares do teste documentados em `tools/mfcc_hosttest/main.c`.

## 10. Manutenção

- `mfcc_tables.h` é **gerado**: ` .venv/bin/python tools/export_mfcc.py`.
  Regenere **só** se mudar parâmetro do §0 — e aí re-treine a base (F2).
- Memória: ~135 KB transientes (malloc/free por janela) + buffers do
  chamador (64 KB PCM + 20 KB MFCC). S3 tem ~512 KB: ok.
- Detalhe 125 vs 126: o notebook fatia a concatenação em **blocos de 125**
  frames; o device processa a janela de 2 s com **126** (center). A média
  é robusta à diferença — paridade validada pelo cosseno, não pela
  contagem.
- `mfcc_debug_frame()` existe p/ depurar um frame estágio a estágio
  (potência/mel/dB); uso só em teste.
