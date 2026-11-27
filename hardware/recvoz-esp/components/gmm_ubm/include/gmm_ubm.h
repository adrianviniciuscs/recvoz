#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define GMM_UBM_M 8
#define GMM_UBM_D 32

typedef struct {
    double nk[GMM_UBM_M];
    double sum[GMM_UBM_M][GMM_UBM_D];
} gmm_adapt_ctx_t;

/**
 * @brief Frame-VAD (top 70% de c0) + extração de 24 dims (c1..c12 + Δc1..Δc12) + CMN seletiva em c1.
 * @return Número de quadros mantidos em out_Tx24 (≤ n_frames).
 */
int gmm_cmn24_window(const float *m39xT, int n_frames, float *out_Tx24);

/** @brief Log-verossimilhança média por frame contra modelo diagonal (NULL = UBM). */
float gmm_logpdf_cmn(const float *Tx24, int n_frames, const float *means_Mx24);

/** @brief LLR = log p(X | spk) - log p(X | UBM) sobre m39xT (aplica CMN internamente). */
float gmm_score_llr(const float *m39xT, int n_frames, const float *spk_means_Mx24);

/** @brief LLR usando matriz já normalizada Tx24 (evita repetir CMN para cada locutor). */
float gmm_score_llr_cmn(const float *Tx24, int n_frames, const float *spk_means_Mx24, float ubm_ll);

/** @brief Inicia adaptação MAP incremental (1 passada, sem precisar guardar todos os frames). */
void gmm_adapt_init(gmm_adapt_ctx_t *ctx);

/** @brief Acumula estatísticas suficientes (nk, sum) de uma janela de 2 s. */
void gmm_adapt_add_window(gmm_adapt_ctx_t *ctx, const float *m39xT, int n_frames);

/** @brief Finaliza adaptação MAP -> out_means_Mx24 (192 floats). */
void gmm_adapt_finish(const gmm_adapt_ctx_t *ctx, float *out_means_Mx24);

#ifdef __cplusplus
}
#endif
