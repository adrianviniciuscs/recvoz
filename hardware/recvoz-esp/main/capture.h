#pragma once
#include <stdint.h>
#define BTN_GPIO 6
#define VAD_CHUNK 1600
int capture_rms16(const int16_t *pcm, int n);
int capture_effective_vad(int base_thr);
int capture_voiced(int16_t *out, int threshold, int trace, void (*progress)(int, int));
