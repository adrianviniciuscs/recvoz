# Classificação — pipeline completo (decisão + enroll)

> Da onda sonora à carinha no display. O DSP (MFCC) está em `docs/MFCC.md`;
> aqui vai **o que acontece com os vetores**: VAD, k-NN, limiar e enroll.

## 0. Mapa em 30 segundos

```
mic 16 kHz ──► VAD (RMS/100 ms, 3 quentes) ──► janela 2 s (32000) ──► MFCC 39×126
  ──► média 39 ──► k-NN k=3 cosseno ──► dmin > τ? ──► 😟 DESCONHECIDO
                                                └─► 🙂 NOME
```

Enroll (botão GPIO6 ou `enroll NOME`): 7 janelas de 2 s → 7 médias →
NVS como locutor novo. A partir daí o k-NN já o conhece.

## 1. VAD — porta de voz (`main/recvoz-esp.c`)

O firmware lê o mic em blocos de 100 ms (1600 amostras) e calcula o RMS:

```
RMS = √( (1/1600)·Σ x[i]² ),   x em int16
```

Se `RMS > vad` (default **300** — voz a ~10 cm dá `std > 300`) por
**3 blocos seguidos** (~300 ms de voz contínua), captura a janela de
decisão de 2 s. Ruído de fundo (`std < 50`) nunca abre a porta.
`vad` ajustável via serial, vale por boot (ver `docs/CALIBRACAO.md`).

## 2. Janela → vetor médio

A janela de 32000 amostras passa por `mfcc_process_window` (detalhes em
`docs/MFCC.md`) e sai uma matriz **39×126** (13 MFCC + 13 Δ + 13 Δ²).
O classificador usa **1 vetor por janela**, a média temporal:

```
v[d] = (1/126)·Σₜ M[d,t],   d = 0..38
```

Por que a média? A identidade do locutor está no **timbre médio**
(envelope espectral), não na fonética instantânea — e a média cancela
variação fonêmica. É exatamente o `mean_vecs` do notebook.

## 3. k-NN — quem fala? (`components/speaker_db/`)

Base: **ADRIAN + PEDRO de fábrica** (36 vetores em flash,
`enroll_default.h`) + até **3 extras em NVS** (8 vetores cada).
Idêntico ao sklearn do notebook (`k=3`, cosseno, peso por distância):

1. **Distância** a cada vetor cadastrado (cosseno = 1 − similaridade):

   ```
   dist(a,b) = 1 − (a·b)/(|a||b|)
   ```

   Invariante a ganho: falar mais alto ou mais baixo **não muda** a
   distância (só a direção do vetor importa).
2. **3 vizinhos mais próximos.** Se o mais próximo tem `d ≈ 0`
   (idêntico), voto direto nele.
3. **Voto ponderado** `peso = 1/d` somado por locutor; vence o maior peso.
4. **Limiar de desconhecido:** se `dmin > τ` → `-1` = DESCONHECIDO,
   mesmo que o k-NN apontasse alguém. Default **τ = 0.0030**.

Números offline (leave-one-out nos 36 vetores): τ=0.003 → **36/36**,
0 rejeições; τ=0.002 rejeita 3 legítimos. Na base, mesma-voz chega a
0.0026 e outra-voz começa em 0.0016 — **overlap fino**, por isso o τ
se calibra em sala (`tau` persiste em NVS; ver `docs/CALIBRACAO.md`).

## 4. Enroll — cadastrando gente nova

```
botão GPIO6 (ou `enroll NOME`) ──► display "FALE 14S…2S"
  ──► 7× [janela 2 s → MFCC → média] ──► NVS ──► 🙂 NOME
```

- São **~15 s de fala contínua** (7 janelas; o 15º segundo é folga).
- 7 vetores cobrem variação fonêmica suficiente p/ o k-NN (a base de
  fábrica tem 12–24 por locutor; 7 basta p/ demo).
- Nome: botão cadastra `VISITANTE`; serial aceita `enroll MARIA`
  (maiúsculas, sem acento na tela, máx 15 letras).
- Limite: 3 extras (`sem slot extra livre` → `reset` e recadastre).
- `reset` apaga os extras e volta p/ só fábrica.

Recalcular "os MFCC e salvar p/ classificar com kNN depois" — exatamente
como você descreveu: o enroll **não treina modelo nenhum**, só guarda
vetores médios; o k-NN os usa na próxima decisão. k-NN é *lazy*: não há
fase de treino, cadastrar = armazenar.

## 5. Limitações honestas (p/ a banca perguntar)

- **Overlap fino** (§3): com 2 locutores o gap mesma-voz/outra-voz é
  estreito; sala ruidosa ou mic longe achata tudo → calibre τ no local.
- **Canal:** treino e teste devem usar o **mesmo mic à mesma distância**
  (~15 cm); trocar de microfone desloca os vetores (por isso o notebook
  previa CMN p/ o caminho GMM — o k-NN vetorial não usa).
- **Latência:** decisão ≈ 2 s de fala + ~0.5 s de DSP; resultado segurado
  3 s no display.
- **Janela única:** cada decisão usa 1 janela de 2 s (sem voto majoritário
  — extensão natural se a banca pedir robustez).
- **Texto independente:** funciona com qualquer frase (não é senha de voz).
