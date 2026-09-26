#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Pipeline idêntica a eda_speaker_recognition.ipynb (librosa 1.x):
 *  16 kHz -> passa-banda 80-7920 (Butter 5a, filtfilt) -> STFT 1024/256
 *  Hann periódica, center + zero-pad -> |.|^2 -> mel128 slaney ->
 *  10*log10 (ref 1.0, top_db 80) -> DCT-II ortho 13 -> deltas
 *  Savitzky-Golay largura 9 -> 39-dim/frame.
 *
 *  Código 100% portátil (só libc + libm): o mesmo .c é testado no host
 *  contra o librosa (tools/mfcc_hosttest) e compilado no firmware.
 */
#define MFCC_SR 16000
#define MFCC_N_FFT 1024
#define MFCC_HOP 256
#define MFCC_N_MFCC 13
#define MFCC_DIM 39
#define MFCC_N_MELS 128
#define MFCC_WIN_SAMPLES 32000         ///< janela de decisão: 2 s
#define MFCC_N_FRAMES (1 + MFCC_WIN_SAMPLES / MFCC_HOP)  ///< 126 (center + zero-pad)

/** @brief Butter 80-7920Hz ida-e-volta (filtfilt), in-place. n = nº de amostras. */
void mfcc_bandpass(float *x, int n);

/**
 * @brief Janela int16 -> MFCC 39xT (row-major [39][T]).
 * @param pcm  janela de MFCC_WIN_SAMPLES (filtro aplicado fora ou não? ver abaixo)
 * @param out  saída [39][MFCC_N_FRAMES]
 * @return nº de frames (126) ou <0 em erro.
 *
 * Aplica o passa-banda internamente (cópia p/ work float). Aloca ~135 KB
 * transientes (malloc/free dentro da chamada).
 */
int mfcc_process_window(const int16_t *pcm, float *out);

/** @brief Variante genérica (n múltiplo de 256; retorna 1 + n/256 frames). */
int mfcc_process_n(const int16_t *pcm, int n, float *out);

/** @brief Vetor médio do segmento (o que o k-NN classifica). */
void mfcc_mean_vector(const float *m39xT, int n_frames, float *mean39);

/** @brief Debug/teste: um frame (x já filtrado) -> potência, mel, dB. */
void mfcc_debug_frame(const float *x, int n, int t,
                      float *power513, float *mel128, float *db128);

/** @brief Cosseno entre vetores 39-dim (p/ o k-NN da F2). */
float mfcc_cosine(const float *a, const float *b);

#ifdef __cplusplus
}
#endif
