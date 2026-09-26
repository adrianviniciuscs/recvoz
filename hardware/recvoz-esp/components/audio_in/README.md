# audio_in — microfone INMP441 p/ ESP-IDF

Captura **16 kHz mono 16-bit** com DC blocker, pronta p/ a cadeia MFCC do
recvoz (mesmo `TARGET_SR` dos notebooks).

## Fiação

Ver mapa completo em [`docs/PINOUT.md`](../../docs/PINOUT.md). Resumo:

| INMP441 | ESP32-S3 |
|---|---|
| SCK → GPIO14, WS → GPIO15, SD → GPIO16 | I2S Philips, slot LEFT |
| L/R → GND, CHIPEN → 3V3 (nunca flutuante) | — |
| VDD → 3V3 + 100 nF, GND → GND | — |

> Sem VU LED (era GPIO4 no projeto original — conflita com o SCL do
> display, ver PINOUT §4). Sem streamer UART: o PCM fica no device.

## Uso

```c
#include "audio_in.h"

audio_in_config_t acfg = AUDIO_IN_CONFIG_DEFAULT();
ESP_ERROR_CHECK(audio_in_init(&acfg));

static int16_t janela[AUDIO_IN_WINDOW_SAMPLES]; // 2 s = 64 KB (estático!)
ESP_ERROR_CHECK(audio_in_read(janela, AUDIO_IN_WINDOW_SAMPLES));
```

Origem: portado de `/home/adrian/projetos/inmp441_mic/main/main.c`.
P/ gravar WAV no host p/ debug, usar aquele projeto (`tools/grab_raw.py`).
