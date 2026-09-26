#pragma once
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/** k-NN igual ao notebook: k=3, cosseno, voto ponderado por 1/d.
 *  Base = 2 locutores de fábrica (enroll_default.h) + até 3 extras em NVS.
 */
#define SPK_DIM 39
#define SPK_EXTRA_MAX 3
#define SPK_EXTRA_VECS 8   ///< 15 s de enroll = 7 janelas de 2 s (cabe)
#define SPK_NAME_LEN 16
#define SPK_TAU_DEFAULT 0.003f ///< limiar desconhecido (calibrar em sala; ver README)

/** @brief nvs_flash_init + carrega extras + tau da NVS. Idempotente. */
esp_err_t speaker_db_init(void);

/** @brief Nº total de locutores (fábrica + extras). */
int speaker_db_count(void);

/** @brief Nome do locutor idx (0..count-1). Ponteiro interno, não liberar. */
const char *speaker_db_name(int idx);

/**
 * @brief Cadastra/atualiza locutor extra (ex "VISITANTE").
 * @param nvec 1..SPK_EXTRA_VECS vetores médios 39-dim.
 */
esp_err_t speaker_db_enroll(const char *name, const float *vecs, int nvec);

/** @brief Apaga extras da NVS (volta p/ só fábrica). */
esp_err_t speaker_db_reset(void);

float speaker_db_get_tau(void);
esp_err_t speaker_db_set_tau(float tau);

/**
 * @brief Classifica vetor médio: k-NN k=3 cosseno, peso 1/d.
 * @return índice do locutor, ou -1 = DESCONHECIDO (d_min > tau).
 * @param out_dist recebe a menor distância (p/ log/calibração, pode ser NULL).
 */
int speaker_db_classify(const float *mean39, float *out_dist);

#ifdef __cplusplus
}
#endif
