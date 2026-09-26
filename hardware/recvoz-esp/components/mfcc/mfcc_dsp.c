// mfcc_dsp — MFCC 39-dim bit a bit compatível com o notebook (librosa 1.x).
// Portátil: só libc + libm. Sem dependências do ESP-IDF.
//
// Convenções replicadas (verificadas no fonte do librosa/scipy instalados):
//  - STFT: center=True, zero-pad de n_fft/2, Hann PERIÓDICA (fftbins),
//    rfft SEM escala (scipy.fft default 'backward').
//  - Mel: 128 bandas slaney, fmax = sr/2 (tabela exata em mfcc_tables.h).
//  - Log: 10*log10(max(e,1e-10)), clip em max-80 (ref=1.0, top_db=80).
//  - DCT-II ortonormal (scipy), 13 primeiros.
//  - Deltas: Savitzky-Golay largura 9 (derivada 1a poly1 / 2a poly2,
//    bordas = ajuste na 1a/última janela — constante p/ estes graus,
//    logo mesmas janelas de 9 com pesos do interior).
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mfcc.h"
#include "mfcc_tables.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PAD (MFCC_N_FFT / 2)  // 512 zeros de cada lado (center)

// ---------- FFT radix-2 complexa, in-place, SEM escala ----------
static void fft_fwd(float *re, float *im, int n)
{
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cwr = 1.0f, cwi = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                float ur = re[i + k], ui = im[i + k];
                float vr = re[i + k + len / 2] * cwr - im[i + k + len / 2] * cwi;
                float vi = re[i + k + len / 2] * cwi + im[i + k + len / 2] * cwr;
                re[i + k] = ur + vr; im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                float nwr = cwr * wr - cwi * wi;
                cwi = cwr * wi + cwi * wr; cwr = nwr;
            }
        }
    }
}

// ---------- passa-banda Butter via SOS, ida-e-volta (filtfilt) ----------
// Extensão ímpar (scipy 'odd'): [2x0-x[e-1..x1], x, 2x[n-1]-x[n-2..]],
// edge = 3 * (2*n_sec+1 - correções) = 33 p/ nosso SOS(5) sem zeros.
#define SOS_EDGE 33

static void sosfilt_df2t(const float sos[MFCC_N_SOS][6], float *x, int n,
                         const float zi[MFCC_N_SOS][2], float x0_scale)
{
    float d[MFCC_N_SOS][2];
    for (int s = 0; s < MFCC_N_SOS; s++) {
        d[s][0] = zi[s][0] * x0_scale;
        d[s][1] = zi[s][1] * x0_scale;
    }
    for (int i = 0; i < n; i++) {
        float v = x[i];
        for (int s = 0; s < MFCC_N_SOS; s++) {
            const float *c = sos[s];
            float out = c[0] * v + d[s][0];
            d[s][0] = c[1] * v - c[4] * out + d[s][1];
            d[s][1] = c[2] * v - c[5] * out;
            v = out;
        }
        x[i] = v;
    }
}

void mfcc_bandpass(float *x, int n)
{
    int e = SOS_EDGE;
    if (n <= e) return;
    float *ext = (float *)malloc((n + 2 * e) * sizeof(float));
    if (!ext) return;
    for (int i = 0; i < e; i++) {
        ext[i] = 2 * x[0] - x[e - i];
        ext[n + e + i] = 2 * x[n - 1] - x[n - 2 - i];
    }
    memcpy(ext + e, x, n * sizeof(float));
    int m = n + 2 * e;
    sosfilt_df2t(MFCC_SOS, ext, m, MFCC_SOS_ZI, ext[0]);
    for (int i = 0; i < m / 2; i++) { float t = ext[i]; ext[i] = ext[m - 1 - i]; ext[m - 1 - i] = t; }
    sosfilt_df2t(MFCC_SOS, ext, m, MFCC_SOS_ZI, ext[0]);
    for (int i = 0; i < n; i++) x[i] = ext[m - 1 - (e + i)];
    free(ext);
}

// ---------- núcleo MFCC ----------
static float s_hann[MFCC_N_FFT];
static int s_hann_ok;

static void hann_init(void)
{
    if (s_hann_ok) return;
    for (int n = 0; n < MFCC_N_FFT; n++)
        s_hann[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / MFCC_N_FFT); // periódica
    s_hann_ok = 1;
}

// Um frame -> potência[513] (rfft sem escala, Hann periódica, zero-pad).
static void frame_power(const float *x, int n, int t, float *power)
{
    static float re[MFCC_N_FFT], im[MFCC_N_FFT];
    hann_init();
    int start = t * MFCC_HOP - PAD; // zero-pad fora de [0, n)
    for (int i = 0; i < MFCC_N_FFT; i++) {
        int idx = start + i;
        float s = (idx < 0 || idx >= n) ? 0.0f : x[idx];
        re[i] = s * s_hann[i];
        im[i] = 0.0f;
    }
    fft_fwd(re, im, MFCC_N_FFT);
    for (int b = 0; b <= MFCC_N_FFT / 2; b++)
        power[b] = re[b] * re[b] + im[b] * im[b];
}

static void mel_apply(const float *power, float *mel)
{
    for (int m = 0; m < MFCC_N_MELS; m++) {
        const float *w = MFCC_MEL_W[m];
        int b = MFCC_MEL_START[m];
        float acc = 0.0f;
        for (int i = 0; i < MFCC_MEL_LEN[m]; i++)
            acc += power[b + i] * w[i];
        mel[m] = acc;
    }
}

static void mel_to_db(float *mel)
{
    float mx = -1e30f;
    for (int m = 0; m < MFCC_N_MELS; m++) {
        if (mel[m] < 1e-10f) mel[m] = 1e-10f;
        mel[m] = 10.0f * log10f(mel[m]);
        if (mel[m] > mx) mx = mel[m];
    }
    float floor = mx - 80.0f;
    for (int m = 0; m < MFCC_N_MELS; m++)
        if (mel[m] < floor) mel[m] = floor;
}

// Um frame -> 13 MFCC (usa buffers temporários do chamador).
static void frame_mfcc(const float *x, int n, int t, float *out13)
{
    float power[MFCC_N_FFT / 2 + 1], mel[MFCC_N_MELS];
    frame_power(x, n, t, power);
    mel_apply(power, mel);
    mel_to_db(mel);

    // DCT-II ortho, 13 primeiros
    for (int k = 0; k < MFCC_N_MFCC; k++) {
        double s = 0.0;
        for (int nn = 0; nn < MFCC_N_MELS; nn++)
            s += mel[nn] * cos(M_PI * (2 * nn + 1) * k / (2.0 * MFCC_N_MELS));
        out13[k] = (float)(s * (k ? sqrt(2.0 / MFCC_N_MELS) : sqrt(1.0 / MFCC_N_MELS)));
    }
}

/** Debug/teste: reexecuta UM frame já filtrado e devolve os estágios. */
void mfcc_debug_frame(const float *x, int n, int t,
                      float *power513, float *mel128, float *db128)
{
    frame_power(x, n, t, power513);
    mel_apply(power513, mel128);
    memcpy(db128, mel128, MFCC_N_MELS * sizeof(float));
    mel_to_db(db128);
}

// Savitzky-Golay largura 9: janela = 9 1os / 9 ultimos / centrada.
static void sg_delta(const float *in13xT, int nfr, float *out13xT, int order2)
{
    static const float W1[9] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
    for (int d = 0; d < MFCC_N_MFCC; d++) {
        for (int t = 0; t < nfr; t++) {
            int base = t < 4 ? 0 : (t > nfr - 5 ? nfr - 9 : t - 4);
            double acc = 0.0;
            for (int i = 0; i < 9; i++) {
                float w = order2 ? MFCC_SG2[i] : W1[i] / 60.0f;
                acc += w * in13xT[d * nfr + base + i];
            }
            out13xT[d * nfr + t] = (float)acc;
        }
    }
}

int mfcc_process_window(const int16_t *pcm, float *out)
{
    return mfcc_process_n(pcm, MFCC_WIN_SAMPLES, out);
}

int mfcc_process_n(const int16_t *pcm, int n, float *out)
{
    if (!pcm || !out || n % MFCC_HOP) return -1;
    int nfr = 1 + n / MFCC_HOP;
    float *x = (float *)malloc(n * sizeof(float));
    float *base = (float *)malloc(MFCC_N_MFCC * nfr * sizeof(float));
    float *d1 = (float *)malloc(MFCC_N_MFCC * nfr * sizeof(float));
    float *d2 = (float *)malloc(MFCC_N_MFCC * nfr * sizeof(float));
    if (!x || !base || !d1 || !d2) { free(x); free(base); free(d1); free(d2); return -1; }
    for (int i = 0; i < n; i++) x[i] = pcm[i] / 32768.0f;
    mfcc_bandpass(x, n);
    float fr[MFCC_N_MFCC];
    for (int t = 0; t < nfr; t++) {
        frame_mfcc(x, n, t, fr);
        for (int d = 0; d < MFCC_N_MFCC; d++) base[d * nfr + t] = fr[d];
    }
    sg_delta(base, nfr, d1, 0);
    sg_delta(base, nfr, d2, 1);
    memcpy(out, base, MFCC_N_MFCC * nfr * sizeof(float));
    memcpy(out + MFCC_N_MFCC * nfr, d1, MFCC_N_MFCC * nfr * sizeof(float));
    memcpy(out + 2 * MFCC_N_MFCC * nfr, d2, MFCC_N_MFCC * nfr * sizeof(float));
    free(x); free(base); free(d1); free(d2);
    return nfr;
}

void mfcc_mean_vector(const float *m39xT, int n_frames, float *mean39)
{
    for (int d = 0; d < MFCC_DIM; d++) {
        double s = 0.0;
        for (int t = 0; t < n_frames; t++) s += m39xT[d * n_frames + t];
        mean39[d] = (float)(s / n_frames);
    }
}

float mfcc_cosine(const float *a, const float *b)
{
    double d = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < MFCC_DIM; i++) {
        d += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0f;
    return (float)(d / (sqrt(na) * sqrt(nb)));
}
