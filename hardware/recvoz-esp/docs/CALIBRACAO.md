# Calibração em sala + roteiro da demo

> Tudo aqui usa só o monitor serial (115200) + o display. Nenhum parâmetro
> exige recompilar: `tau` persiste em NVS; `vad` vale por boot
> (ajuste e anote o valor bom p/ repetir).

## 0. Pré-voo (fiação)

1. Conferir `docs/PINOUT.md`: display SDA5/SCL4, mic 14/15/16 (+L/R→GND,
   CHIPEN→3V3, 100 nF no VDD), **botão GPIO6→GND**, tudo em 3V3 com GND comum.
2. Sanidade do mic (opcional, no outro repo): `tools/live_audio.py` —
   voz a ~10 cm deve dar `std > 300`, `max > 2000`; `std < 50` = silêncio.

## 1. Flash + boot esperado

```bash
idf.py -p /dev/ttyACM0 flash monitor
```

No log deve aparecer, nesta ordem:

```
recvoz_display: SSD1306 em 0x3C        # display achado no I2C
audio_in: INMP441 @16000 Hz (BCLK=14 WS=15 SD=16)
speaker_db: pronto: 2 locutores (tau=0.0170)
recvoz: recvoz pronto (2 locutores). Fale p/ classificar, GPIO6 p/ enroll.
```

Display mostra o robozinho `OUVINDO` (olhos passeando + piscada).

## 2. Calibrar o VAD (porta de voz)

O VAD mede RMS em blocos de 100 ms; voz precisa de **3 blocos quentes**
seguidos (`> vad`) p/ disparar a janela de 2 s. Default `vad=300`.

| Sintoma | Ação |
|---|---|
| Fala a 15 cm e nada dispara | `vad 200` (repete até disparar sempre) |
| Dispara sozinho no silêncio/ar-condicionado | `vad 400` (sobe de 100 em 100) |
| Corta o começo da frase | normal: chegue mais perto (~15 cm) e fale contínuo 2–3 s |

```text
recvoz> vad        # lê o atual
recvoz> vad 250    # ajusta (vale até reboot; anote o final!)
```

## 3. Calibrar o τ (desconhecido)

`tau` = distância cosseno máxima p/ aceitar como conhecido (k-NN sem c0).
Default `0.0170`: na base com-voz (10 locutor A + 14 locutor B), mesma-voz até
0.0119 e outra-voz desde 0.0219 (leave-one-out 24/24, gap limpo). Valide em sala:

1. `list` → deve mostrar `0: LOCUTOR_A`, `1: LOCUTOR_B`.
2. Locutor A fala 3× de ~15 cm e de ~1 m → tem que dar `LOCUTOR_A` sempre.
   Anote as distâncias do log (`d=...`).
3. Locutor B fala 2× → `LOCUTOR_B` sempre.
4. Uma 3ª pessoa fala 2× → tem que dar `DESCONHECIDO`.
5. Regra:
   - Conhecido rejeitado → `tau` **sobe** (ex `tau 0.020`).
   - Desconhecido aceito → `tau` **desce** (ex `tau 0.014`).
   - Se nem assim separar: recadastre com `enroll NOME` no mesmo mic/sala
     (vetores ao vivo valem mais que τ — ver CLASSIFICACAO §4).
   - `tau` persiste em NVS (sobrevive a reboot).

```text
recvoz> tau          # lê o atual
recvoz> tau 0.004    # ajusta e salva
```

Se não houver ponto que separe bem na sala, aproxime o mic (15 cm) e
fale 3 s contínuos por decisão — a média de 126 frames estabiliza.

## 4. Ensaio do enroll

0. Antes: `diag` falando contínuo 3 s — mostra `+RMS` dos blocos
   aproveitados (`.rms` dos pulados) e `dists:` por locutor. Se a sua
   distância ao seu nome não for a menor, recadastre (passo 1).
1. Aperte o botão GPIO6 (ou `enroll MARIA`): display mostra `FALE 14S…2S`.
2. A pessoa fala **contínuo por ~15 s** (frases normais, mesma distância).
3. Display abre sorriso + nome; `list` mostra o novo locutor.
4. A pessoa fala de novo → reconhecida pelo nome.
5. Errou o enroll? `reset` apaga os extras (volta p/ só fábrica) e repete.

Nome do botão é sempre `VISITANTE`; pelo serial dá p/ nomear
(`enroll MARIA` — máx 15 letras, sem acento exibido em maiúsculas).

## 5. Roteiro da demo (~2 min)

| # | Ação | Esperado no display |
|---|---|---|
| 1 | Locutor A fala 3 s | 🙂 `LOCUTOR_A` |
| 2 | Locutor B fala 3 s | 🙂 `LOCUTOR_B` |
| 3 | 3ª pessoa fala 3 s | 🙁 `DESCONHECIDO` |
| 4 | Aperta GPIO6, pessoa fala 15 s | `FALE 14S…` → 🙂 `VISITANTE` |
| 5 | Pessoa fala de novo | 🙂 `VISITANTE` |

Fale a ~15 cm do mic, ambiente o mais quieto possível.

## 6. Troubleshooting

| Sintoma | Causa provável | Ação |
|---|---|---|
| Display apagado | fiação/VCC | ver scan I2C no boot (`I2C: 0x3C`); confira 3V3/GND/SDA/SCL |
| `SSD1306 não encontrado` | endereço/fio | `0x3D` é tentado sozinho; se scan não lista nada, confira pull-ups e solda |
| Nunca dispara (sempre OUVINDO) | `vad` alto ou mic mudo | §2; confira BCLK/WS/SD e CHIPEN→3V3, L/R→GND |
| Sempre DESCONHECIDO | `tau` baixo ou longe do mic | §3; chegue a 15 cm |
| Troca LOCUTOR_A↔LOCUTOR_B | vozes parecidas na sala / `tau` alto | baixe `tau` um pouco; fale mais longo |
| Enroll não termina | soltou o botão tarde? | botão só dispara no **aperto** (espera soltar); via serial use `enroll NOME` |
| `sem slot extra livre` | 3 extras cheios | `reset` e recadastre |
| Boot em loop | NVS corrompida (raro) | apague flash: `idf.py erase-flash && idf.py flash` |

## 7. Fallback total (se o botão falhar na hora)

Tudo pelo serial: `enroll VISITANTE` → fala 15 s → `list` confere →
fala de novo. O botão é só um atalho.
