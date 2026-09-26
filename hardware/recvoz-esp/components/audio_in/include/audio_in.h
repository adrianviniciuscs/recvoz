#pragma once
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#include "driver/gpio.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Taxa do projeto (== TARGET_SR dos notebooks). Não trocar sem re-treinar. */
#define AUDIO_IN_SAMPLE_RATE 16000
/** Janela de decisão: 2 s == SEG_DUR do notebook (32000 amostras, 64 KB). */
#define AUDIO_IN_WINDOW_SAMPLES (AUDIO_IN_SAMPLE_RATE * 2)

typedef struct {
    gpio_num_t bclk;   ///< SCK do INMP441 (padrão 14)
    gpio_num_t ws;     ///< WS/LRCLK (padrão 15)
    gpio_num_t sd;     ///< SD/DOUT (padrão 16). L/R do módulo em GND (slot LEFT).
    int sample_rate;   ///< padrão 16000
} audio_in_config_t;

#define AUDIO_IN_CONFIG_DEFAULT() \
    (audio_in_config_t){ .bclk = 14, .ws = 15, .sd = 16, .sample_rate = AUDIO_IN_SAMPLE_RATE }

/**
 * @brief Inicializa I2S (Philips 32-bit stereo, slot LEFT) + warmup do DMA.
 *
 * Fiação e detalhes em `docs/PINOUT.md`. Sem LED e sem streamer UART
 * (diferenças em relação ao projeto original `inmp441_mic`).
 */
esp_err_t audio_in_init(const audio_in_config_t *cfg);

/**
 * @brief Lê `n` amostras int16 mono com DC blocker aplicado (bloqueante).
 *
 * O filtro CC é contínuo entre chamadas — não chamar com gaps longos
 * no meio de uma frase (o warmup só ocorre no init).
 */
esp_err_t audio_in_read(int16_t *pcm_out, size_t n_samples);

/** @brief Atalho: grava N segundos (ex 2 p/ classificar, 15 p/ enroll). */
static inline esp_err_t audio_in_record_seconds(int16_t *out, int seconds)
{
    return audio_in_read(out, (size_t)AUDIO_IN_SAMPLE_RATE * seconds);
}

#ifdef __cplusplus
}
#endif
