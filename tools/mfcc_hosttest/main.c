// Teste de paridade: DSP em C vs librosa (esperados de tools/export_mfcc.py).
// Uso: make run   (lê ../mfcc_test/*.bin, compara, imprime erros)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mfcc.h"

#define N 32000
#define NF 126
#define DIM 39

static void *load(const char *p, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f) { printf("FALTA %s (rode tools/export_mfcc.py)\n", p); exit(2); }
    void *b = malloc(n);
    if (fread(b, 1, n, f) != n) { printf("SHORT %s\n", p); exit(2); }
    fclose(f);
    return b;
}
static float maxabs_f(const float *a, const float *b, int n, int *at) {
    float m = 0; int ai = 0;
    for (int i = 0; i < n; i++) { float e = fabsf(a[i]-b[i]); if (e > m) { m = e; ai = i; } }
    if (at) *at = ai;
    return m;
}

int main(void) {
    const char *d = "../mfcc_test/";
    char p[256];
    snprintf(p, sizeof p, "%stest_pcm.bin", d);
    int16_t *pcm = load(p, N * 2);
    snprintf(p, sizeof p, "%sexpected_bp.bin", d);
    float *exp_bp = load(p, N * 4);
    snprintf(p, sizeof p, "%sexpected_mfcc.bin", d);
    float *exp_m = load(p, DIM * NF * 4);
    snprintf(p, sizeof p, "%sexpected_mean.bin", d);
    float *exp_mean = load(p, DIM * 4);

    // 1. bandpass isolado
    float *x = malloc(N * 4);
    for (int i = 0; i < N; i++) x[i] = pcm[i] / 32768.0f;
    mfcc_bandpass(x, N);
    int at;
    printf("bandpass  maxabs=%.3g @%d\n", maxabs_f(x, exp_bp, N, &at), at);

    // 1b. estágios do frame 37
    snprintf(p, sizeof p, "%sexpected_pow37.bin", d);
    float *e_pow = load(p, 513 * 4);
    snprintf(p, sizeof p, "%sexpected_mel37.bin", d);
    float *e_mel = load(p, 128 * 4);
    snprintf(p, sizeof p, "%sexpected_db37.bin", d);
    float *e_db = load(p, 128 * 4);
    float pow513[513], mel128[128], db128[128];
    mfcc_debug_frame(x, N, 37, pow513, mel128, db128);
    // razão (fator de escala) e erro relativo
    double num = 0, den = 0;
    for (int i = 0; i < 513; i++) { num += (double)e_pow[i] * pow513[i]; den += (double)pow513[i] * pow513[i]; }
    printf("pow37 ratio(lib/c)=%.4f  maxabs=%.3g\n", num / den, maxabs_f(pow513, e_pow, 513, &at));
    printf("pow37 c[1:6]: %.3g %.3g %.3g %.3g %.3g\n", pow513[1], pow513[2], pow513[3], pow513[4], pow513[5]);
    printf("mel37 maxabs=%.3g  db37 maxabs=%.3g\n",
           maxabs_f(mel128, e_mel, 128, NULL), maxabs_f(db128, e_db, 128, NULL));

    // 2. pipeline completa
    float *out = malloc(DIM * NF * 4);
    int nf = mfcc_process_window(pcm, out);
    printf("frames=%d (esperado %d)\n", nf, NF);
    if (nf != NF) return 1;
    float e = maxabs_f(out, exp_m, DIM * NF, &at);
    printf("mfcc39x126 maxabs=%.3g @%d (d=%d t=%d)\n", e, at, at / NF, at % NF);
    // top-3 outliers (sem mutar out: bloco/mean abaixo usam o original)
    float te[3] = {0, 0, 0}; int td[3] = {0, 0, 0}, tt[3] = {0, 0, 0};
    for (int dd = 0; dd < DIM; dd++) for (int tti = 0; tti < NF; tti++) {
        float ee = fabsf(out[dd * NF + tti] - exp_m[dd * NF + tti]);
        for (int q = 0; q < 3; q++) if (ee > te[q]) {
            for (int r = 2; r > q; r--) { te[r] = te[r-1]; td[r] = td[r-1]; tt[r] = tt[r-1]; }
            te[q] = ee; td[q] = dd; tt[q] = tti; break;
        }
    }
    for (int q = 0; q < 3; q++)
        printf("  outlier d=%d t=%d err=%.3g mine=%.6g ref=%.6g\n",
               td[q], tt[q], te[q], out[td[q] * NF + tt[q]], exp_m[td[q] * NF + tt[q]]);
    // por bloco
    for (int b = 0; b < 3; b++) {
        const char *n = b == 0 ? "mfcc " : b == 1 ? "delta " : "delta2";
        printf("  %s maxabs=%.3g\n", n, maxabs_f(out + b*13*NF, exp_m + b*13*NF, 13*NF, NULL));
    }

    // 3. vetor médio + cosseno (o que o k-NN usa)
    float mean[DIM], c;
    mfcc_mean_vector(out, NF, mean);
    printf("mean39  maxabs=%.3g\n", maxabs_f(mean, exp_mean, DIM, &at));
    c = mfcc_cosine(mean, exp_mean);
    printf("cos(mean_c, mean_ref)=%.8f  dist=%.3g\n", c, 1 - c);

    int fail = !isfinite(e) || e > 3.0f || (1 - c) > 1e-4f;
    // Limiares: mfcc frame-a-frame < 3.0 (float32 tem piso de ruído em
    // frames de quase-silêncio: 16 dB numa banda -> ~2.1 nos coefs;
    // inofensivo p/ o k-NN). O que importa — cosseno do vetor médio —
    // exige dist < 1e-4 (típico 6e-8).
    printf(fail ? "FAIL\n" : "PASS\n");
    return fail;
}
