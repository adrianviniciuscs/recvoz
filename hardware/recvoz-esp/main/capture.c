#include <math.h>
#include <stdio.h>
#include <string.h>
#include "esp_check.h"
#include "driver/gpio.h"
#include "audio_in.h"
#include "mfcc.h"
#include "noise_sup.h"
#include "capture.h"
#define CAP_ITERS 100
int capture_effective_vad(int base_thr)
{
    if (!ns_is_ready()) return base_thr;
    int floor_vad = ns_get_noise_rms() * 2;
    return floor_vad > base_thr ? floor_vad : base_thr;
}
int capture_rms16(const int16_t *p, int n)
{
    if (!p || n <= 0) return 0;
    if (n <= VAD_CHUNK) {
        static float xf[VAD_CHUNK];
        for (int i = 0; i < n; i++) xf[i] = p[i] / 32768.0f;
        mfcc_bandpass(xf, n);
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double s = (double)xf[i] * 32768.0;
            sum += s * s;
        }
        return (int)sqrt(sum / n);
    }
    long long sum = 0;
    for (int i = 0; i < n; i++) sum += (int)p[i] * p[i];
    return (int)sqrt((double)sum / n);
}
// Junta 2 s SÓ de blocos acima do VAD (pula silêncio/reação), p/ a janela
// não diluir a média com quietude. 1 = ok, 0 = timeout, -1 = botão.
int capture_voiced(int16_t *out, int threshold, int trace, void (*prog)(int, int))
{
    static int16_t chunk[VAD_CHUNK];
    int got = 0, reported = -1;
    for (int it = 0; it < CAP_ITERS && got < AUDIO_IN_WINDOW_SAMPLES; it++) {
        if (gpio_get_level(BTN_GPIO) == 0) return -1;
        ESP_ERROR_CHECK(audio_in_read(chunk, VAD_CHUNK));
        int r = capture_rms16(chunk, VAD_CHUNK);
        if (trace) printf(r > threshold ? "+%d " : ".%d ", r);
        if (r > threshold) {
            memcpy(out + got, chunk, sizeof(chunk));
            got += VAD_CHUNK;
            if (prog && got - reported >= 6400) { reported = got; prog(got, AUDIO_IN_WINDOW_SAMPLES); }
        }
    }
    if (trace) printf("\n  voz: %d/%d blocos\n", got / VAD_CHUNK, AUDIO_IN_WINDOW_SAMPLES / VAD_CHUNK);
    if (prog) prog(got, AUDIO_IN_WINDOW_SAMPLES);
    return got >= AUDIO_IN_WINDOW_SAMPLES ? 1 : 0;
}

