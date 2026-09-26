# Mapa de GPIOs — recvoz-esp (ESP32-S3, ESP-IDF 6.1)

> **Regra de ouro:** antes de ligar qualquer fio novo, confira nesta tabela se o
> GPIO já está em uso e se ele não é pino de *strapping* ou USB nativo.
> Última revisão: 2026-09-26 · target `esp32s3`.

## Resumo (tudo que está ligado)

| GPIO | Função | Periférico | Direção | Observação |
|------|--------|------------|---------|------------|
| 5 | SDA | Display OLED (I2C0) | I/O OD + pull-up | 400 kHz, pull-up interno ativado |
| 4 | SCL | Display OLED (I2C0) | O OD + pull-up | 400 kHz, pull-up interno ativado |
| 14 | BCLK/SCK | Mic INMP441 (I2S) | O | ~1,024 MHz |
| 15 | WS/LRCLK | Mic INMP441 (I2S) | O | 16 kHz |
| 16 | SD/DOUT | Mic INMP441 (I2S) | I | L/R = GND (slot LEFT) |
| 0 | BOOT (botão) | Reservado / não usado | I + pull-up interno | Recvoz usa botão **externo no GPIO6** (ver §3) |
| 6 | Enroll (botão externo) | Cadastro de locutor | I + pull-up interno | Botão momentâneo p/ GND, ativo em baixo |
| 48 | LED RGB built-in (WS2812) | **Forçado apagado** | O, nível 0 no boot | Pino flutuante acendia sozinho; firmware trava em 0 |
| — | VU LED | **DESATIVADO** | — | Era GPIO4 no projeto do mic → **conflito com SCL**, removido |

**GPIOs livres (seguros p/ expansão):** 1, 2, 7–13, 17, 18, 21, 33–48 (evitar 19/20, 0, 3, 45, 46 — ver §4).

## 1. Display OLED SSD1306 128×64 (I2C)

| Pino display | ESP32-S3 | Config |
|---|---|---|
| VCC | 3V3 | — |
| GND | GND | — |
| SDA | **GPIO5** | I2C0, 400 kHz, pull-up interno `enable_internal_pullup` |
| SCL | **GPIO4** | I2C0, 400 kHz, pull-up interno `enable_internal_pullup` |

- Endereço: auto-detecção **0x3C → 0x3D** (`addr = 0`); fixo via `recvoz_display_config_t.addr`.
- 4 pinos (sem RESET); `reset_gpio_num = -1`.
- Para trocar pinos: `cfg.sda / cfg.scl` em `RECVOZ_DISPLAY_CONFIG_DEFAULT()`.
- Faixa amarela (linhas 0–15) = barra de status com o texto (`OUVINDO` / nome / `DESCONHECIDO`).

## 2. Microfone INMP441 (I2S)

| Pino INMP441 | ESP32-S3 | Config |
|---|---|---|
| SCK (BCLK) | **GPIO14** | I2S STD Philips master, 16 kHz |
| WS (LRCLK) | **GPIO15** | 16 kHz |
| SD (DOUT) | **GPIO16** | 32-bit stereo, lê **slot LEFT**, `>> 16` → int16 mono |
| L/R | GND | seleciona slot LEFT |
| CHIPEN | 3V3 | **nunca deixar flutuando** |
| VDD | 3V3 + 100 nF p/ GND | capacitor junto ao módulo |
| GND | GND | — |

- Taxa: **16 kHz mono 16-bit** (igual ao `TARGET_SR` dos notebooks).
- DC blocker Q15 (`32630/32768 ≈ 0,9957`) remove offset CC.
- Janela de decisão: **2 s = 32000 amostras = 64 KB**.
- Referência original: `/home/adrian/projetos/inmp441_mic` (firmware + `tools/live_audio.py`).

## 3. Botão de enroll (cadastro)

| Função | Pino | Comportamento |
|---|---|---|
| Enroll | **GPIO6 (botão externo p/ GND)** | Press curto → inicia enroll: display mostra contagem regressiva de **15 s** (7 janelas de 2 s), recalcula MFCC e salva em NVS como `VISITANTE` (ou nome via serial) |

- Entrada com pull-up interno, ativo em baixo. Debounce 50 ms no firmware
  (espera soltar antes de começar).
- Fiação: um lado do botão no **GPIO6**, outro no **GND**. Sem resistor externo.
- Por que não o BOOT/GPIO0? É pino de *strapping* (nível no boot muda o modo
  de boot) — botão externo no GPIO6 evita qualquer risco.
- Fallback sem botão: comando serial `enroll [NOME]` (ver `main/recvoz-esp.c`).
- Comandos seriais (115200): `enroll [NOME]` | `list` | `reset` |
  `tau [V]` (limiar desconhecido) | `vad [V]` (limiar de voz, rms).

## 4. Pinos proibidos / cuidados (ESP32-S3)

| Pino | Motivo |
|---|---|
| GPIO0 | *Strapping* (boot mode) + botão BOOT — **não ligar nada que force nível no boot**; botão momentâneo p/ GND é OK |
| GPIO3 | *Strapping* — evitar |
| GPIO45, GPIO46 | *Strapping* — evitar |
| GPIO19, GPIO20 | USB nativo D−/D+ (flash + monitor) — **não usar** |
| GPIO4 (LED) | **Conflito histórico**: o projeto do mic usava GPIO4 p/ VU LED e o display usa GPIO4 p/ SCL — LED foi **desativado** por decisão de 2026-09-26 |

## 5. Alimentação

- Display + INMP441 em **3V3** (nunca 5 V nos sinais: SDA/SCL/SD/WS/SCK são 3V3).
- GND comum único entre ESP32, display e mic.
- Capacitor 100 nF junto ao VDD do INMP441 (ver §2).

## 6. Como alterar pinos

1. Display: `recvoz_display_config_t cfg = RECVOZ_DISPLAY_CONFIG_DEFAULT(); cfg.sda = X; cfg.scl = Y;`
2. Mic: `audio_in_config_t` (`INMP441_PIN_BCLK/WS/SD` em `components/audio_in/`).
3. Atualize **esta tabela** no mesmo commit (documentação anda junto do código).

## Histórico de decisões

| Data | Decisão |
|---|---|
| 2026-09-26 | Target migrado ESP32-C3 → **ESP32-S3** |
| 2026-09-26 | Display em **5/4** (SDA/SCL), I2C0 400 kHz |
| 2026-09-26 | Mic em **14/15/16** (BCLK/WS/SD), 16 kHz mono |
| 2026-09-26 | **VU LED (GPIO4) desativado** — conflito com SCL do display |
| 2026-09-26 | Enroll no **botão externo GPIO6** (p/ GND), nome fixo `VISITANTE`, 15 s |
