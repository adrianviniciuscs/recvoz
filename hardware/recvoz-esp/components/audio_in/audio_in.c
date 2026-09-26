// audio_in — captura INMP441 (I2S) p/ o recvoz.
//
// Portado de /home/adrian/projetos/inmp441_mic/main/main.c com duas
// diferenças decididas em 2026-09-26 (ver docs/PINOUT.md):
//   1. Sem VU LED (era GPIO4, conflita com o SCL do display).
//   2. Sem streamer UART: o PCM fica no device (classificação on-board).
//      P/ debug gravando WAV no host, usar o projeto inmp441_mic original.
//
// Cadeia: I2S Philips 32-bit stereo @16 kHz -> slot LEFT -> >>16 ->
// DC blocker y[n] = x[n] - x[n-1] + 0.9957*y[n-1] (Q15 32630) -> int16.
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_check.h"
#include "driver/i2s_std.h"
#include "audio_in.h"

static const char *TAG = "audio_in";

#define FRAME_SAMPLES 256
#define DC_BLOCKER_COEF_Q15 32630 // 32630/32768 ~= 0.9957

static i2s_chan_handle_t s_rx;
static int32_t s_x_prev, s_y_prev;

esp_err_t audio_in_init(const audio_in_config_t *cfg)
{
    if (s_rx) return ESP_OK;
    audio_in_config_t c = cfg ? *cfg : AUDIO_IN_CONFIG_DEFAULT();

    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan.dma_desc_num = 4;
    chan.dma_frame_num = FRAME_SAMPLES;
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan, NULL, &s_rx), TAG, "new_channel");

    const i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(c.sample_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = c.bclk,
            .ws = c.ws,
            .dout = I2S_GPIO_UNUSED,
            .din = c.sd,
            .invert_flags = {false, false, false},
        },
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_rx, &std), TAG, "std_mode");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_rx), TAG, "enable");

    // Warmup: descarta DMA velho + assenta o passa-altas.
    int32_t *trash = malloc(FRAME_SAMPLES * 2 * sizeof(int32_t));
    ESP_RETURN_ON_FALSE(trash, ESP_ERR_NO_MEM, TAG, "oom");
    size_t n = 0;
    for (int i = 0; i < 5; i++)
        i2s_channel_read(s_rx, trash, FRAME_SAMPLES * 2 * sizeof(int32_t), &n, portMAX_DELAY);
    free(trash);
    s_x_prev = s_y_prev = 0;
    ESP_LOGI(TAG, "INMP441 @%d Hz (BCLK=%d WS=%d SD=%d)", c.sample_rate, c.bclk, c.ws, c.sd);
    return ESP_OK;
}

esp_err_t audio_in_read(int16_t *pcm_out, size_t n_samples)
{
    ESP_RETURN_ON_FALSE(s_rx && pcm_out, ESP_ERR_INVALID_STATE, TAG, "init primeiro");
    int32_t *raw = malloc(FRAME_SAMPLES * 2 * sizeof(int32_t));
    ESP_RETURN_ON_FALSE(raw, ESP_ERR_NO_MEM, TAG, "oom");

    size_t got = 0;
    while (got < n_samples) {
        size_t want = n_samples - got;
        if (want > FRAME_SAMPLES) want = FRAME_SAMPLES;
        size_t nbytes = 0;
        esp_err_t r = i2s_channel_read(s_rx, raw, FRAME_SAMPLES * 2 * sizeof(int32_t), &nbytes, portMAX_DELAY);
        if (r != ESP_OK) { free(raw); return r; }
        int words = nbytes / sizeof(int32_t), m = 0;
        for (int i = 0; i < words && m < (int)want; i += 2) { // só slot LEFT
            int32_t s = raw[i] >> 16;
            int32_t y = s - s_x_prev + (int32_t)((s_y_prev * DC_BLOCKER_COEF_Q15) / 32768);
            s_x_prev = s;
            s_y_prev = y;
            if (y > 32767) y = 32767;
            if (y < -32768) y = -32768;
            pcm_out[got + m++] = (int16_t)y;
        }
        got += m;
    }
    free(raw);
    return ESP_OK;
}
