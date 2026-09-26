# recvoz_display — robozinho OLED p/ ESP-IDF

Componente ESP-IDF (sem Arduino/LVGL) que mostra um robozinho no
**SSD1306 128×64 (I2C)** com 3 carinhas + texto de status na faixa amarela:

| Carinha | Quando usar | Texto padrão |
|---|---|---|
| `RECVOZ_FACE_LISTENING` | ouvindo / aguardando voz | `OUVINDO` |
| `RECVOZ_FACE_HAPPY` | reconheceu o locutor | nome (ex `ADRIAN`) |
| `RECVOZ_FACE_SAD` | desconhecido | `DESCONHECIDO` |

Olhos paramétricos inspirados nos presets do
[esp32-eyes](https://github.com/playfultechnology/esp32-eyes)
(playfultechnology / Luis Llamas): Normal / Happy / Sad, com
transição suave (*morph*), piscada e olhar pros lados no modo escuta.
Fonte 5×7 clássica Adafruit embutida (`font5x7.h`).

## Fiação

Ver mapa completo em [`docs/PINOUT.md`](../../docs/PINOUT.md). Resumo:

| OLED | ESP32-S3 |
|---|---|
| VCC → 3V3, GND → GND, SDA → **GPIO5**, SCL → **GPIO4** | I2C0 400 kHz, endereço auto 0x3C/0x3D |

## Uso

```c
#include "recvoz_display.h"

recvoz_display_config_t cfg = RECVOZ_DISPLAY_CONFIG_DEFAULT();
// cfg.sda = 5; cfg.scl = 4; cfg.addr = 0; // 0 = auto 0x3C/0x3D
ESP_ERROR_CHECK(recvoz_display_init(&cfg));

recvoz_display_listening();     // OUVINDO
recvoz_display_happy("MARIA");  // reconheceu (aceita minúsculas e acentos: "João" -> "JOAO")
recvoz_display_unknown();       // DESCONHECIDO
// ou genérico: recvoz_display_show(RECVOZ_FACE_HAPPY, "PEDRO");
```

- `show()` é **não-bloqueante**: uma task interna anima (morph ~300 ms,
  blink ~150 ms a cada ~3–5 s, olhar aleatório) e empurra o frame a ~30 fps.
- Texto: nome curto (≤ 8 letras) sai **grande (×2)** na faixa amarela;
  texto longo sai ×1 (trunca em 21). Créditos da fonte no cabeçalho de `font5x7.h`.
