#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "gmm_ubm.h"
#include "gmm_default.h"

#define GMM_FVAD_FRAC 0.70f
#define GMM_FVAD_MIN  10

int gmm_cmn24_window(const float *m39xT, int n_frames, float *out_Tx24)
{
    if (!m39xT || !out_Tx24 || n_frames <= 0) return 0;

    // Frame-level VAD: mantém os 70% quadros de maior energia (c0),
    // descartando micropausas internas e consoantes surdas antes da CMN-c1.
    int k_keep = (int)(n_frames * GMM_FVAD_FRAC + 0.5f);
    if (k_keep < GMM_FVAD_MIN) k_keep = GMM_FVAD_MIN;
    if (k_keep > n_frames) k_keep = n_frames;

    uint8_t keep[256];
    int use_mask = (n_frames <= (int)sizeof(keep) && k_keep < n_frames);
    if (use_mask) {
        int order[256];
        for (int t = 0; t < n_frames; t++) order[t] = t;
        for (int i = 1; i < n_frames; i++) {
            int idx = order[i];
            float key = m39xT[idx];
            int j = i - 1;
            while (j >= 0 && m39xT[order[j]] > key) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = idx;
        }
        memset(keep, 0, (size_t)n_frames);
        for (int i = n_frames - k_keep; i < n_frames; i++)
            keep[order[i]] = 1;
    }

    // CMN seletiva em c1 (d=0): remove viés de inclinação espectral/canal
    // preservando a assinatura do trato vocal em c2..c12 e dinâmicas Δc1..Δc12.
    double s_c1 = 0.0;
    int kept = 0;
    for (int t = 0; t < n_frames; t++) {
        if (use_mask && !keep[t]) continue;
        s_c1 += m39xT[1 * n_frames + t];
        kept++;
    }
    if (kept <= 0) return 0;
    float mu_c1 = (float)(s_c1 / kept);

    const int n_stat = GMM_UBM_D / 2;   // 16 (c1..c16)
    const int d1_off = n_stat + 1;      // 17 (início do bloco Δc0..Δc16)
    int out_i = 0;
    for (int t = 0; t < n_frames; t++) {
        if (use_mask && !keep[t]) continue;
        out_Tx24[out_i * GMM_UBM_D + 0] = m39xT[1 * n_frames + t] - mu_c1;
        out_Tx24[out_i * GMM_UBM_D + n_stat] = m39xT[(d1_off + 1) * n_frames + t];
        for (int d = 1; d < n_stat; d++) {
            out_Tx24[out_i * GMM_UBM_D + d] = m39xT[(1 + d) * n_frames + t];
            out_Tx24[out_i * GMM_UBM_D + n_stat + d] = m39xT[(d1_off + 1 + d) * n_frames + t];
        }
        out_i++;
    }
    return out_i;
}

static void frame_comp_logp(const float *x, const float *means_Mx24, float *out_logp)
{
    for (int m = 0; m < GMM_UBM_M; m++) {
        const float *mu = means_Mx24 ? (means_Mx24 + m * GMM_UBM_D) : GMM_UBM_MEANS[m];
        const float *iv = GMM_UBM_INVVAR[m];
        float maha = 0.0f;
        for (int d = 0; d < GMM_UBM_D; d++) {
            float diff = x[d] - mu[d];
            maha += diff * diff * iv[d];
        }
        out_logp[m] = GMM_UBM_LOGW[m] + GMM_UBM_LOGCONST[m] - 0.5f * maha;
    }
}

static float logsumexp_m(const float *logp)
{
    float mx = logp[0];
    for (int m = 1; m < GMM_UBM_M; m++)
        if (logp[m] > mx) mx = logp[m];
    float s = 0.0f;
    for (int m = 0; m < GMM_UBM_M; m++)
        s += expf(logp[m] - mx);
    return mx + logf(s);
}

float gmm_logpdf_cmn(const float *Tx24, int n_frames, const float *means_Mx24)
{
    if (!Tx24 || n_frames <= 0) return -100.0f;
    double total = 0.0;
    float logp[GMM_UBM_M];
    for (int t = 0; t < n_frames; t++) {
        frame_comp_logp(Tx24 + t * GMM_UBM_D, means_Mx24, logp);
        total += logsumexp_m(logp);
    }
    return (float)(total / n_frames);
}

float gmm_score_llr_cmn(const float *Tx24, int n_frames, const float *spk_means_Mx24, float ubm_ll)
{
    return gmm_logpdf_cmn(Tx24, n_frames, spk_means_Mx24) - ubm_ll;
}

float gmm_score_llr(const float *m39xT, int n_frames, const float *spk_means_Mx24)
{
    if (!m39xT || n_frames <= 0) return -100.0f;
    float *cmn = (float *)malloc((size_t)n_frames * GMM_UBM_D * sizeof(float));
    if (!cmn) return -100.0f;
    int nk = gmm_cmn24_window(m39xT, n_frames, cmn);
    float ubm_ll = gmm_logpdf_cmn(cmn, nk, NULL);
    float llr = gmm_logpdf_cmn(cmn, nk, spk_means_Mx24) - ubm_ll;
    free(cmn);
    return llr;
}

void gmm_adapt_init(gmm_adapt_ctx_t *ctx)
{
    if (ctx) memset(ctx, 0, sizeof(*ctx));
}

void gmm_adapt_add_window(gmm_adapt_ctx_t *ctx, const float *m39xT, int n_frames)
{
    if (!ctx || !m39xT || n_frames <= 0) return;
    float *cmn = (float *)malloc((size_t)n_frames * GMM_UBM_D * sizeof(float));
    if (!cmn) return;
    int nk = gmm_cmn24_window(m39xT, n_frames, cmn);
    float logp[GMM_UBM_M];
    for (int t = 0; t < nk; t++) {
        const float *xt = cmn + t * GMM_UBM_D;
        frame_comp_logp(xt, NULL, logp);
        float lse = logsumexp_m(logp);
        for (int m = 0; m < GMM_UBM_M; m++) {
            double gamma = exp((double)logp[m] - (double)lse);
            ctx->nk[m] += gamma;
            for (int d = 0; d < GMM_UBM_D; d++)
                ctx->sum[m][d] += gamma * (double)xt[d];
        }
    }
    free(cmn);
}

void gmm_adapt_finish(const gmm_adapt_ctx_t *ctx, float *out_means_Mx24)
{
    if (!ctx || !out_means_Mx24) return;
    for (int m = 0; m < GMM_UBM_M; m++) {
        double nk = ctx->nk[m] + 1e-12;
        double alpha = nk / (nk + (double)GMM_R);
        for (int d = 0; d < GMM_UBM_D; d++) {
            double ek = ctx->sum[m][d] / nk;
            out_means_Mx24[m * GMM_UBM_D + d] =
                (float)(alpha * ek + (1.0 - alpha) * (double)GMM_UBM_MEANS[m][d]);
        }
    }
}
