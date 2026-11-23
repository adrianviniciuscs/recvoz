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

$$s[k] = \sum_{n=0}^{127} \mathrm{DCT}[k,n]\cdot L[n], \quad \mathrm{DCT} = \text{tabela 13×128 em } \texttt{mfcc\_tables.h}$$

Tabela gerada = `scipy.fft.dct(type=2, norm="ortho")` (fator $\sqrt{1/128}$ no
k=0, $\sqrt{2/128}$ nos demais). Dot em float na FPU em vez de 209k `cos()`
em double emulado — isso cortava ~4 s por janela no S3. E atenção ao c0:
como o log é absoluto, qualquer ganho global desloca **só o c0**
($\Delta c_0 = \Delta\mathrm{dB}\cdot\sqrt{128}$) — a FFT sem escala (§3.3) existe por causa dele.

## 7. Deltas Savitzky-Golay (largura 9)

> A referência está fixada em librosa 1.0.0 e usa Savitzky-Golay de largura 9.
> Mudanças de versão devem passar pelo teste de paridade antes de atualizar as tabelas.

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

k-NN `k=3`, cosseno sem c0 e voto ponderado vive em `knn`; `speaker_db` delega a decisão.

## 9. Paridade numérica

Na raiz: `make -C tests/host/mfcc run`. A referência em
`src/recvoz/features.py` usa SOS e piso de log por frame, como o firmware.
No fixture atual: erro máximo MFCC ≈ 0,0159, vetor médio ≈ 0,000465 e
 distância cosseno reportada 0 na precisão float32. Não há igualdade bit a bit.

O teste exige 126 frames, valores finitos, erro máximo por coeficiente ≤ 3,0
e distância cosseno do vetor médio ≤ 1e-4. O limite por coeficiente mantém
margem para sinais sintéticos de baixa energia; não é uma meta de erro típica.
A paridade também é exercitada com uma fixture sintética gerada durante os
testes Python. Esses testes não medem acurácia de identificação.

## 10. Manutenção

- `mfcc_tables.h` é **gerado**: ` .venv/bin/python tools/export_mfcc.py --help`.
  Regenere **só** se mudar parâmetro do §0 — e aí re-treine a base (F2).
- Memória: ~135 KB transientes (malloc/free por janela) + buffers do
  chamador (64 KB PCM + 20 KB MFCC). S3 tem ~512 KB: ok.
- O pipeline atual calcula cada janela de 32.000 amostras separadamente:
  Python e firmware produzem 126 frames. A base histórica de fábrica ainda
  preserva a segmentação antiga de 125 frames; veja `TREINO.md`.
- `mfcc_debug_frame()` existe p/ depurar um frame estágio a estágio
  (potência/mel/dB); uso só em teste. De propósito NÃO aplica supressão
  de ruído (§12) — depura os estágios crus.

## 12. Supressão de ruído (opcional, fora da paridade)

Subtração espectral em potência (Berouti 1979), em `noise_sup.c` — na mão,
só libc+libm, sem esp-dsp. Entra entre o §3.3 e o §4:

$$P_{\text{limpo}}[k] = \max(P[k] - \alpha\cdot N[k],\ \mathrm{floor}\cdot P[k]), \quad k = 0..512$$

- **Perfil** $N[k]$: média do espectro em 0.5 s de **silêncio** (8192
  amostras). A calibração (`mfcc_noise_calibrate()`) passa pelo MESMO
  `filtfilt` da voz — sem isso o perfil descasaria do espectro visto no
  `frame_mfcc()`. 1ª calibração inicializa; demais adaptam por EMA
  ($N \mathrel{+}= \beta(A - N)$, $\beta = 0.3$) + suavização 3-pt em
  frequência $[0.25, 0.5, 0.25]$ contra ruído musical.
- **Defaults**: $\alpha = 1.2$ (oversubtração), $\mathrm{floor} = 0.05$
  (−13 dB de piso proporcional — bin nunca zera de tutto, não pisca).
- **Custo**: perfil 513 floats (~2 KB estáticos) + 513 sub+max por frame;
  transiente da calibração ~65 KB (heap, na tarefa principal).
- **Paridade**: engine nasce DESLIGADA e sem perfil `ns_apply()` é no-op
  bit-exato — `mfcc_hosttest` continua PASS sem mudar (provado também pelo
  T5 do `ns_hosttest`: calibrar em zeros + processar = bit-idêntico).
- **Validação funcional** (`make -C tests/host/ns run`, sintético
  determinístico): seno 440+880 Hz + ruído branco ±3000, perfil do ruído
  puro → distância cosseno ao limpo **0.0307 → 0.0223** (−27%).
- **Limites**: só ruído ESTACIONÁRIO (hum/hiss/ventilador). Voz concorrente,
  batida e reverb não resolve. O VAD que alimenta o perfil usa limiar
  conservador (2/3 do `vad`) — vazar fala no $N[k]$ subtrai o locutor.
  `ns_snr_db()` (10·log10 ΣP/ΣN) serve p/ conferir a calibração em sala.

## 11. Diagrama

[ÁUDIO BRUTO (PCM)]
    │  (16.000 amostras por segundo)
    ▼
1. FILTRAGEM & JANELAMENTO
    │  - Limpa ruídos extremos (Passa-banda ~80-7920 Hz)
    │  - Corta o som em quadros de 64 ms (1024 amostras), hop de 16 ms
    ▼
2. FFT (Transformada Rápida de Fourier)
    │  - Converte o tempo para frequência
    │  - Obtém a Magnitude/Potência de 513 bins de frequência linear
    ▼
3. BANCO DE FILTROS MEL
    │  - Multiplica os 513 bins pelo banco de triângulos
    │  - Agrupa e reduz para 128 "Bandas Mel"
    ▼
4. LOGARITMO (Em escala dB)
    │  - Aplica o Log na energia de cada banda
    │  - Separa a convolução (voz = fonte * filtro) em SOMA (log(voz) = log(fonte) + log(filtro))
    ▼
5. DCT (Transformada Discreta de Cosseno)
    │  - Descorrelaciona as bandas (como uma PCA "de graça")
    │  - Concentra a forma da boca nos primeiros coeficientes
    │  - Trunca: descarta o resto e GUARDA SÓ OS 13 PRIMEIRA COEFICIENTES (Estáticos)
    ▼
6. DELTAS E DELTA-DELTAS (Derivadas Temporais)
    │  - Calcula a Velocidade (Δ): variação dos 13 números entre quadros
    │  - Calcula a Aceleração (ΔΔ): variação da velocidade entre quadros
    ▼
[VETOR FINAL POR FRAME]
    │  13 Estáticos + 13 Deltas + 13 Delta-Deltas = 39 DIMENSÕES
    ▼
7. MÉDIA TEMPORAL
    │  - Tira a média dos 39 números ao longo dos 2 segundos de áudio
    ▼
[MÉDIA DE 39 NÚMEROS = VETOR FINAL PARA CLASSIFICAÇÃO]
