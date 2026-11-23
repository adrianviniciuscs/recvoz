#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include "driver/i2s_std.h"
#include "audio_in.h"
static int sample, mode;
esp_err_t i2s_new_channel(const i2s_chan_config_t *c, void *tx, i2s_chan_handle_t *rx) { (void)c; (void)tx; *rx = (void *)1; return ESP_OK; }
esp_err_t i2s_channel_init_std_mode(i2s_chan_handle_t h, const i2s_std_config_t *c) { (void)h; (void)c; return ESP_OK; }
esp_err_t i2s_channel_enable(i2s_chan_handle_t h) { (void)h; return ESP_OK; }
esp_err_t i2s_channel_read(i2s_chan_handle_t h, void *out, size_t size, size_t *got, unsigned timeout) {
    (void)h; (void)timeout;
    assert(size % 8 == 0);
    if (mode == 1) { *got = 0; return ESP_OK; }
    if (mode == 2 && size > 8) size = (size / 16) * 8;
    int32_t *p = out;
    for (size_t i = 0; i < size / 8; i++) { p[2*i] = (sample++ % 1000) * 65536; p[2*i+1] = 0; }
    *got = size;
    return ESP_OK;
}
int main(void) {
    assert(audio_in_init(NULL) == ESP_OK);
    sample = 0;
    int16_t pcm[1600];
    assert(audio_in_read(pcm, 1600) == ESP_OK);
    assert(sample == 1600); // Regressão: antes consumia 1792 e descartava 192.
    mode = 2;
    assert(audio_in_read(pcm, 1600) == ESP_OK);
    assert(sample == 3200);
    mode = 1;
    assert(audio_in_read(pcm, 1600) == ESP_FAIL);
    puts("PASS audio continuous consumption / partial / empty reads");
}
