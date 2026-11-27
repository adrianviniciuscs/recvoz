#pragma once
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Base de locutores: persistência (fábrica em flash + extras em NVS).
 *
 *  O algoritmo k-NN (k=3, cosseno, voto 1/d — igual ao notebook) mora no
 *  componente `knn`; aqui montamos a base e delegamos a decisão a ele.
 *  Base = 2 locutores de fábrica (enroll_default.h) + até 3 extras em NVS.
 */
#define SPK_DIM 34
#define SPK_EXTRA_MAX 3
#define SPK_EXTRA_VECS 8   ///< 15 s de enroll = 7 janelas de 2 s (cabe)
#define SPK_NAME_LEN 16
#define SPK_TAU_DEFAULT 0.70f ///< limiar equivalente GMM-UBM FLEURS (LLR >= +1.50; d = 1 - LLR/5)

/** @brief nvs_flash_init + carrega extras + tau da NVS. Idempotente. */
esp_err_t speaker_db_init(void);

/** @brief Nº total de locutores (fábrica ativa + extras). */
int speaker_db_count(void);

/** @brief Nome do locutor idx (0..count-1). Ponteiro interno, não liberar. */
const char *speaker_db_name(int idx);

/**
 * @brief Cadastra/atualiza locutor extra (ex "VISITANTE").
 * @param nvec 1..SPK_EXTRA_VECS vetores médios 39-dim.
 *
 * Se o nome é de fábrica (ex "ADRIAN"), os vetores ao vivo passam a
 * valer e os de fábrica saem da votação (bit persistido; `reset` reverte).
 */
esp_err_t speaker_db_enroll(const char *name, const float *vecs, int nvec);

/** @brief Cadastra locutor extra com modelo GMM-UBM adaptado (192 floats) + vetores médios. */
esp_err_t speaker_db_enroll_gmm(const char *name, const float *vecs, int nvec, const float *gmm_means_Mx24);

/** @brief Apaga extras da NVS (volta p/ só fábrica). */
esp_err_t speaker_db_reset(void);

float speaker_db_get_tau(void);
esp_err_t speaker_db_set_tau(float tau);

/**
 * @brief Classifica vetor médio via k-NN (delega a knn_classify, sem o c0).
 * @return índice do locutor, ou -1 = DESCONHECIDO (d_min > tau).
 * @param out_dist recebe a menor distância (p/ log/calibração, pode ser NULL).
 */
int speaker_db_classify(const float *mean39, float *out_dist);

/** Resultado completo p/ didática (top-2 + margem no display). */
typedef struct {
    int winner;      ///< índice (ou -1 = desconhecido)
    float dmin;      ///< distância ao vencedor (ou ao mais próximo, se -1)
    int runner;      ///< vice (índice, -1 se só há 1 locutor)
    float drunner;   ///< distância ao vice
    int nearest;     ///< mais próximo por distância (mesmo se desconhecido)
} spk_result_t;

/** @brief Classifica e preenche top-1/top-2 (margem = drunner - dmin). */
int speaker_db_classify_full(const float *mean39, spk_result_t *out);

/** @brief Menor distância ao locutor idx (p/ diag/calibração). */
float speaker_db_dist_to(int idx, const float *mean39);

/** @brief Classifica janela MFCC 39xT via GMM-UBM + CMN; preenche out e LLRs top-2. */
int speaker_db_classify_gmm(const float *m39xT, int n_frames, spk_result_t *out,
                            float *out_llr1, float *out_llr2);

/** @brief LLR de uma janela MFCC 39xT para o locutor idx (para diag). */
float speaker_db_llr_to(int idx, const float *m39xT, int n_frames);

#ifdef __cplusplus
}
#endif
