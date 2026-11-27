// Teste do supressor de ruído (noise_sup.c + gancho no mfcc_dsp.c).
// 100% sintético e determinístico — sem fixtures, exceto o T5 que reusa
// ../../fixtures/test_pcm.bin p/ provar paridade com ruído zero.
// Uso: make run
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mfcc.h"
#include "noise_sup.h"
#include "voice_prior.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N 32000
#define NF 126
#define DIM MFCC_DIM
#define NCAL 8192

static int fails;

static void check(int ok, const char *name, const char *detail)
{
    printf("%s %s %s\n", ok ? "ok  " : "FAIL", name, detail ? detail : "");
    if (!ok) fails++;
}

static float maxabs_f(const float *a, const float *b, int n)
{
    float m = 0;
    for (int i = 0; i < n; i++) { float e = fabsf(a[i] - b[i]); if (e > m) m = e; }
    return m;
}

// Gerador determinístico (LCG) — nada de rand()/srand() do host.
static uint32_t s_lcg = 12345;
static float uni01(void)
{
    s_lcg = 1664525u * s_lcg + 1013904223u;
    return (float)((s_lcg >> 16) & 0x7FFF) / 32767.0f;
}

int main(void)
{
    // ---- T1: desligado ou sem perfil = passthrough bit-exato ----
    {
        float p[NS_BINS], ref[NS_BINS];
        for (int k = 0; k < NS_BINS; k++) p[k] = ref[k] = (float)(k * 1.5 + 1.0);
        ns_reset(); ns_set_enabled(0);
        ns_apply(p);
        check(maxabs_f(p, ref, NS_BINS) == 0.0f, "T1a", "desligado pass-through");
        ns_set_enabled(1); // ligado mas sem perfil: ainda no-op
        ns_apply(p);
        check(maxabs_f(p, ref, NS_BINS) == 0.0f, "T1b", "sem perfil pass-through");
    }

    // ---- T2: matemática do Wiener casado à voz (alpha=1, floor=0) ----
    {
        float prof[NS_BINS], p[NS_BINS], exp[NS_BINS];
        double e_tot = 0.0, e_n = 0.0;
        for (int k = 0; k < NS_BINS; k++) {
            prof[k] = (float)(k + 1) * 0.2f;
            p[k] = 1000.0f;
            e_tot += p[k];
            e_n += prof[k];
        }
        double e_sig = e_tot - e_n;
        for (int k = 0; k < NS_BINS; k++) {
            double sh = e_sig * (double)NS_VOICE_PRIOR[k];
            double g = sh / (sh + (double)prof[k]);
            exp[k] = (float)(g * g * (double)p[k]);
        }
        ns_reset(); ns_set_enabled(1);
        ns_set_params(1.0f, 0.0f);
        ns_update_avg(prof);
        ns_apply(p);
        char d[64];
        snprintf(d, sizeof d, "maxabs=%.3g", maxabs_f(p, exp, NS_BINS));
        check(maxabs_f(p, exp, NS_BINS) < 1e-3f, "T2", d);
    }

    // ---- T3: piso proporcional (ruído muito alto não zera o bin) ----
    {
        float prof[NS_BINS], p[NS_BINS];
        for (int k = 0; k < NS_BINS; k++) prof[k] = 5000.0f;
        for (int k = 0; k < NS_BINS; k++) p[k] = 100.0f;
        ns_reset(); ns_set_enabled(1);
        ns_set_params(1.2f, 0.05f);
        ns_update_avg(prof);
        ns_apply(p);
        check(fabsf(p[7] - 5.0f) < 1e-3f && fabsf(p[300] - 5.0f) < 1e-3f, "T3", "piso=0.05*P");
    }

    // ---- T4: EMA entre calibrações (beta=0.5: perfil cai de 100 -> 50) ----
    {
        float a[NS_BINS], b[NS_BINS];
        for (int k = 0; k < NS_BINS; k++) { a[k] = 100.0f; b[k] = 0.0f; }
        ns_reset(); ns_set_enabled(1);
        ns_set_beta(0.5f);
        ns_update_avg(a);
        ns_update_avg(b); // N = 50 no interior (suavização preserva patamar)
        const float *prof = ns_get_profile();
        char d[64];
        snprintf(d, sizeof d, "N[100]=%.2f (esp 50)", prof ? prof[100] : -1.0f);
        check(prof && fabsf(prof[100] - 50.0f) < 0.5f, "T4", d);
        ns_set_beta(NS_BETA_DEFAULT);
        ns_set_params(NS_ALPHA_DEFAULT, NS_FLOOR_DEFAULT);
    }

    // ---- T5: paridade fim-a-fim com ruído zero ----
    // Calibra em 8192 zeros (perfil N=0) e processa a fixture real: a saída
    // tem que ser BIT-IDENTICA à sem supressão (N=0 => P-alpha*0=P).
    {
        FILE *f = fopen("../../fixtures/test_pcm.bin", "rb");
        if (!f) { printf("FALTA ../../fixtures/test_pcm.bin (rode tools/export_mfcc.py)\n"); return 2; }
        int16_t *pcm = malloc(N * 2);
        if (fread(pcm, 1, N * 2, f) != N * 2) { printf("SHORT test_pcm.bin\n"); return 2; }
        fclose(f);
        float *o0 = malloc(DIM * NF * 4), *o1 = malloc(DIM * NF * 4);
        int16_t *zero = calloc(NCAL, 2);
        ns_reset(); ns_set_enabled(0);
        int nf0 = mfcc_process_window(pcm, o0);
        ns_set_enabled(1);
        int nf1 = mfcc_noise_calibrate(zero, NCAL);
        int nf2 = mfcc_process_window(pcm, o1);
        char d[64];
        snprintf(d, sizeof d, "nf=%d/%d maxabs=%.3g", nf0, nf2, maxabs_f(o0, o1, DIM * NF));
        check(nf0 == NF && nf1 > 0 && nf2 == NF && maxabs_f(o0, o1, DIM * NF) == 0.0f, "T5", d);
        free(pcm); free(o0); free(o1); free(zero);
    }

    // ---- T6: funcional — fala real (fixture) + ruído branco, perfil do ruído puro ----
    // Ruído estacionário conhecido: a distância ao limpo tem que DIMINUIR.
    {
        FILE *f = fopen("../../fixtures/test_pcm.bin", "rb");
        if (!f) { printf("FALTA ../../fixtures/test_pcm.bin (rode tools/export_mfcc.py)\n"); return 2; }
        int16_t *clean = malloc(N * 2), *noisy = malloc(N * 2), *nz = malloc(NCAL * 2);
        if (fread(clean, 1, N * 2, f) != N * 2) { printf("SHORT test_pcm.bin\n"); return 2; }
        fclose(f);
        for (int i = 0; i < NCAL; i++) {
            float w = (uni01() * 2.0f - 1.0f) * 3000.0f;
            nz[i] = (int16_t)w;
        }
        for (int i = 0; i < N; i++) {
            float w = (uni01() * 2.0f - 1.0f) * 3000.0f;
            double v = (double)clean[i] + w;
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            noisy[i] = (int16_t)v;
        }
        float *oc = malloc(DIM * NF * 4), *oraw = malloc(DIM * NF * 4), *ons = malloc(DIM * NF * 4);
        float mc[DIM], mr[DIM], mn[DIM];
        ns_reset(); ns_set_enabled(0);
        mfcc_process_n(clean, N, oc);
        mfcc_process_n(noisy, N, oraw);
        ns_set_enabled(1);
        ns_set_params(NS_ALPHA_DEFAULT, NS_FLOOR_DEFAULT);
        mfcc_noise_calibrate(nz, NCAL);
        mfcc_process_n(noisy, N, ons);
        mfcc_mean_vector(oc, NF, mc);
        mfcc_mean_vector(oraw, NF, mr);
        mfcc_mean_vector(ons, NF, mn);
        float draw = 1 - mfcc_cosine(mc, mr), dns = 1 - mfcc_cosine(mc, mn);
        char d[96];
        snprintf(d, sizeof d, "dist_raw=%.5f dist_ns=%.5f (ganho=%.5f)", draw, dns, draw - dns);
        check(dns <= draw, "T6", d);
        free(clean); free(noisy); free(nz); free(oc); free(oraw); free(ons);
    }

    // ---- T7: ns_snr_db — 10*log10(sumP/sumN), -99 sem perfil ----
    {
        float prof[NS_BINS], p[NS_BINS];
        for (int k = 0; k < NS_BINS; k++) { prof[k] = 10.0f; p[k] = 100.0f; }
        ns_reset();
        check(ns_snr_db(p) == -99.0f, "T7a", "sem perfil = -99");
        ns_update_avg(prof);
        float snr = ns_snr_db(p); // 512 bins a 100 vs 10 -> 10 dB
        char d[64];
        snprintf(d, sizeof d, "snr=%.3fdB (esp 10)", snr);
        check(fabsf(snr - 10.0f) < 0.05f, "T7b", d);
    }

    // ---- T8: âncora de ruído + modulação articulatória (rejeição de ruído puro) ----
    {
        int16_t *nz_cal = malloc(NCAL * 2), *nz_win = malloc(N * 2);
        for (int i = 0; i < NCAL; i++) nz_cal[i] = (int16_t)((uni01() * 2.0f - 1.0f) * 1500.0f);
        for (int i = 0; i < N; i++) nz_win[i] = (int16_t)((uni01() * 2.0f - 1.0f) * 1500.0f);
        float *on = malloc(DIM * NF * 4);
        float vn[DIM];
        ns_reset();
        ns_set_enabled(1);
        mfcc_noise_calibrate(nz_cal, NCAL);
        mfcc_process_n(nz_win, N, on);
        mfcc_mean_vector(on, NF, vn);
        float d_anchor = ns_noise_dist(vn);
        float mod = mfcc_frame_modulation(on, NF);
        int nrms = ns_get_noise_rms();
        char d[96];
        snprintf(d, sizeof d, "rms=%d d_anchor=%.4f mod=%.1f/<%.1f", nrms, d_anchor, mod, mfcc_min_voice_mod());
        check(nrms > 500 && d_anchor < 0.035f && mod < mfcc_min_voice_mod(), "T8", d);
        free(nz_cal); free(nz_win); free(on);
    }

    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
