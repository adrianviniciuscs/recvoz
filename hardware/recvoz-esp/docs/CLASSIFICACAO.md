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

## 1. VAD + janela só-com-voz (`main/recvoz-esp.c`)

O firmware lê o mic em blocos de 100 ms (1600 amostras) e calcula o RMS:

$$\mathrm{RMS} = \sqrt{\frac{1}{1600}\sum x[i]^2}, \quad x \text{ em int16}$$

- **Disparo:** 3 blocos quentes seguidos (`RMS > vad`, default **300**).
- **Janela sem silêncio:** após o disparo, acumula blocos **só acima do
  `vad`** até 32000 amostras (timeout 10 s). Reação lenta, pausa entre
  frases e rabo de silêncio ficam de fora — sem isso a média diluía
  (medido: metade da janela em silêncio deslocava `d` p/ 0.01+).
  Ruído de fundo (`std < 50`) nunca entra na janela.

## 2. Janela → vetor médio

A janela de 32000 amostras passa por `mfcc_process_window` (detalhes em
`docs/MFCC.md`) e sai uma matriz **39×126** (13 MFCC + 13 Δ + 13 Δ²).
O classificador usa **1 vetor por janela**, a média temporal:

$$v[d] = \frac{1}{126}\sum_t M[d,t], \quad d = 0..38$$

Por que a média? A identidade do locutor está no **timbre médio**
(envelope espectral), não na fonética instantânea — e a média cancela
variação fonêmica. É exatamente o `mean_vecs` do notebook.

## 3. k-NN — quem fala? (`components/speaker_db/`)

Base: **ADRIAN + PEDRO de fábrica** (24 vetores em flash,
`enroll_default.h`) + até **3 extras em NVS** (8 vetores cada).
Idêntico ao sklearn do notebook (`k=3`, cosseno, peso por distância):

1. **Distância** a cada vetor cadastrado (cosseno = 1 − similaridade),
   calculada nos dims **1..38 — sem o c0**:

   $$\mathrm{dist}(a,b) = 1 - \frac{a\cdot b}{|a||b|} \quad \text{(somando } i = 1..38\text{)}$$

   O c0 (energia log média) depende do ganho do mic e da distância da
   boca — não da identidade — e era ele que causava o overlap
   mesma-voz/outra-voz (com c0: gap −0.001; sem c0: gap limpo
   0.0185–0.0264). Invariante restante: falar alto/baixo não muda a
   direção do vetor.
2. **3 vizinhos mais próximos.** Se o mais próximo tem `d ≈ 0`
   (idêntico), voto direto nele.
3. **Voto ponderado** $\mathrm{peso} = 1/d$ somado por locutor; vence o maior peso.
4. **Limiar de desconhecido:** se $d_{\min} > \tau$ → `-1` = DESCONHECIDO,
   mesmo que o k-NN apontasse alguém. Default **τ = 0.017**
   (meio do gap 0.0119–0.0219; leave-one-out 24/24).

Números offline (leave-one-out nos 24 vetores com-voz, sem c0): τ=0.017 →
**24/24**, 0 rejeições; mesma-voz até 0.0119, outra-voz desde 0.0219
(**gap limpo**). A base de fábrica usa só blocos acima do VAD
(igual às janelas ao vivo — ver export_mfcc.py). Mesmo assim o τ se valida em sala
(`tau` persiste em NVS; ver `docs/CALIBRACAO.md`), porque sala real tem
ruído e vetores ao vivo.

## 3b. Didática: top-2 + margem (placar no display)

Cada decisão expõe `spk_result_t`: vencedor + $d_{\min}$, **vice + $d_{\mathrm{vice}}$**,
$\mathrm{margem} = d_{\mathrm{vice}} - d_{\min}$. O display alterna carinha (3/5 do `hold`) com
o **placar em fonte grande**: nome (×2), `dmin` + veredito `:-)`/`:-(` (×2),
vice em pequeno e **barra da distância com o traço do τ** (valor exato
do τ no log serial). Na banca: margem grande = decisão folgada; barra
encostando no traço = caso duvidoso (bom gancho p/ falar de limiar
open-set). O monitor imprime o placar em ASCII espelhando o display.

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
- **Override:** enroll com nome de fábrica (`enroll ADRIAN`) guarda os
  vetores ao vivo (mesmo mic/sala/ganho) e tira os de fábrica da votação
  — essencial quando a base veio de outro microfone. `reset` reverte.
- Limite: 3 extras (`sem slot extra livre` → `reset` e recadastre).
- `reset` apaga os extras e volta p/ só fábrica.

Recalcular "os MFCC e salvar p/ classificar com kNN depois" — exatamente
como você descreveu: o enroll **não treina modelo nenhum**, só guarda
vetores médios; o k-NN os usa na próxima decisão. k-NN é *lazy*: não há
fase de treino, cadastrar = armazenar.

## 5. Limitações honestas (p/ a banca perguntar)

- **Overlap fino** (§3): com 2 locutores o gap mesma-voz/outra-voz é
  estreito; sala ruidosa ou mic longe achata tudo → calibre τ no local.
- **Canal/ganho:** o c0 saiu da decisão, mas treino e teste ainda devem
  usar o **mesmo mic à mesma distância** (~15 cm). Vetores de outro
  microfone se resolvem com enroll ao vivo (§4, override).
- **Latência:** decisão ≈ 2 s de fala + ~0.5 s de DSP; resultado segurado
  3 s no display. Falar pausado ajuda a encher a janela mais rápido
  (timeout de 10 s por janela).
- **Janela única:** cada decisão usa 1 janela de 2 s (sem voto majoritário
  — extensão natural se a banca pedir robustez).
- **Texto independente:** funciona com qualquer frase (não é senha de voz).
