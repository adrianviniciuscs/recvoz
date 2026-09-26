# recvoz-esp — firmware (ESP32-S3, ESP-IDF 6.1)

Reconhecimento de locutor **on-board**: ouve 2 s → MFCC 39 → k-NN →
mostra carinha no OLED. Sem laptop na demo.

```
INMP441 (I2S) → audio_in → mfcc → speaker_db → recvoz_display (SSD1306)
                                    ↑ botão GPIO6 / console serial
```

## Componentes

| Componente | Papel | Detalhe em |
|---|---|---|
| `audio_in` | INMP441 16 kHz mono + DC blocker | seu `README.md` |
| `mfcc` | MFCC 39-dim bit-igual ao notebook | `docs/MFCC.md` |
| `speaker_db` | k-NN k=3 + NVS (fábrica + enroll) | seu `README.md` |
| `recvoz_display` | robozinho: OUVINDO / nome / DESCONHECIDO | seu `README.md` |
| `main` | VAD + máquina de estados + enroll + console | `docs/CALIBRACAO.md` |

## Documentos

- `docs/PINOUT.md` — **leia primeiro**: mapa de GPIOs, fiação, pinos proibidos.
- `docs/TREINO.md` — coleta → notebook → export → validação: como a base nasce.
- `docs/MFCC.md` — pipeline DSP com equações + paridade numérica.
- `docs/CLASSIFICACAO.md` — VAD → k-NN → limiar → enroll, de ponta a ponta.
- `docs/CALIBRACAO.md` — calibrar VAD/τ em sala + roteiro da demo + troubleshooting.

## Quickstart

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash monitor   # 115200, comandos: enroll/list/reset/tau/vad
make -C ../../tools/mfcc_hosttest run  # teste de paridade MFCC (gate da F1)
.venv/bin/python ../../tools/export_mfcc.py  # regenera tabelas (só se mudar o DSP)
```

Base de fábrica: **ADRIAN + PEDRO** (36 vetores, `speaker_db/enroll_default.h`).
Enroll (15 s, botão GPIO6→GND ou `enroll NOME`) salva até 3 extras em NVS
como `VISITANTE` (ou o nome dado).
