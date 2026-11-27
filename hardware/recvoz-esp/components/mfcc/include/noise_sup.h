#pragma once
#include <stddef.h>
#include <stdint.h>

/** @brief Ponteiro interno p/ o perfil de ruído (513 potências) ou NULL se não calibrado. */
const float *ns_get_profile(void);

/** @brief Registra a âncora MFCC do ruído ambiente e o RMS na banda de voz. */
void ns_set_noise_anchor(const float *vec39, int rms16);

/** @brief Âncora MFCC 39-dim do ruído calibrado (ou NULL se não calibrado). */
const float *ns_get_noise_anchor(void);

/** @brief RMS int16 na banda de voz do silêncio calibrado (0 se não calibrado). */
int ns_get_noise_rms(void);

/** @brief Distância cosseno (sem c0) entre mean39 e a âncora de ruído (1.0 se indisponível). */
float ns_noise_dist(const float *mean39);

#ifdef __cplusplus
extern "C" {
#endif

/** Supressão espectral de ruído estacionário via Filtro de Wiener Casado à Voz.
 *
 *  Ideia: enquanto o VAD diz "silêncio", estima o espectro médio do ruído
 *  N[k]; na voz, aplica um ganho suave de Wiener casado ao prior espectral
 *  normalizado de fala humana S_voz[k] (FLEURS PT-BR, 513 bins em Flash):
 *    H[k] = max( (E_sig * S_voz[k]) / (E_sig * S_voz[k] + alpha * N[k]), sqrt(floor) )
 *    P_limpo[k] = H[k]^2 * P[k]
 *  Entra entre `frame_power()` e `mel_apply()` do mfcc, sem ruído musical.
 *
 *  Limites honestos: resolve ruído ESTACIONÁRIO (hum, hiss, ventilador).
 *  Não resolve voz concorrente, batida, nem reverb. Se o ruído dispara o
 *  VAD sozinho, o VAD quebra antes — aí o ganho vem de CMVN/VAD, não daqui.
 *
 *  Concorrência: SEM trava interna. No firmware, `ns_update_*` roda sob o
 *  mutex `s_dsp` (ver main) e `ns_apply()` roda dentro do `frame_mfcc()`
 *  (também sob `s_dsp`) — nunca concorrentes. Leitura de 2 KB sem escrita
 *  concorrente é segura.
 *
 *  Paridade: com engine DESLIGADA (padrão) ou perfil vazio, `ns_apply()` é
 *  no-op bit-exato — o teste `mfcc_hosttest` continua PASS sem mudar.
 */
#define NS_BINS 513              ///< bins de potência (rfft 1024)
#define NS_CAL_SAMPLES 8192      ///< lote de calibração: 0.512 s @16k (múltiplo de 256)
#define NS_ALPHA_DEFAULT 0.8f    ///< oversubtração moderada (preserva formantes)
#define NS_FLOOR_DEFAULT 0.05f   ///< piso proporcional (−13 dB)
#define NS_BETA_DEFAULT 0.3f     ///< EMA entre calibrações

/** @brief Zera o perfil (volta a passthrough até a 1ª calibração). */
void ns_reset(void);

/** @brief 1 se já houve ao menos uma calibração (perfil válido). */
int ns_is_ready(void);

/** @brief Liga/desliga o engine. Padrão: DESLIGADO (paridade com librosa). */
void ns_set_enabled(int en);
/** @brief Estado do engine (0/1). */
int ns_is_enabled(void);

/**
 * @brief Ajusta oversubtração e piso.
 * @param alpha  (0, 4]: >1 subtrai mais (agressivo), <1 conserva mais voz.
 * @param floor_ratio [0, 0.5]: piso = floor*P[k] por bin.
 * @return 0 ok, -1 se fora de faixa (mantém os antigos).
 */
int ns_set_params(float alpha, float floor_ratio);

/** @brief Lê os parâmetros atuais (ponteiros podem ser NULL). */
void ns_get_params(float *alpha, float *floor_ratio);

/**
 * @brief Adaptação entre calibrações (0, 1]: 1 = troca tudo, ~0.3 = suave.
 * Valores fora de (0,1] são ignorados (mantém o atual).
 */
void ns_set_beta(float beta);
/** @brief Beta atual. */
float ns_get_beta(void);

/**
 * @brief Incorpora UM espectro médio de ruído (513 potências, pós-filtragem).
 *
 * 1ª chamada inicializa (memcpy); demais fazem EMA por bin com `beta` +
 * suavização 3-pt em frequência [0.25, 0.5, 0.25]. Chamar só com SILÊNCIO
 * (VAD conservador) — vazar fala aqui subtrai o próprio locutor.
 */
void ns_update_avg(const float *avg_pow513);

/**
 * @brief Subtração por frame (chamada dentro do frame_mfcc).
 * No-op se engine desligada, perfil vazio ou `pow513 == NULL`.
 */
void ns_apply(float *pow513);

/**
 * @brief SNR estimada do frame contra o perfil, em dB (10*log10(sumP/sumN),
 * bins 1..512). Retorna −99 se perfil vazio/ponteiro NULL; +99 se N==0.
 * Útil p/ `diag` e p/ validar calibração em sala.
 */
float ns_snr_db(const float *pow513);

/** @brief Ponteiro interno p/ o perfil de ruído (513 potências) ou NULL se não calibrado. */
const float *ns_get_profile(void);

/** @brief Registra a âncora MFCC do ruído ambiente e o RMS na banda de voz. */
void ns_set_noise_anchor(const float *vec39, int rms16);

/** @brief Âncora MFCC 39-dim do ruído calibrado (ou NULL se não calibrado). */
const float *ns_get_noise_anchor(void);

/** @brief RMS int16 na banda de voz do silêncio calibrado (0 se não calibrado). */
int ns_get_noise_rms(void);

/** @brief Distância cosseno (sem c0) entre mean39 e a âncora de ruído (1.0 se indisponível). */
float ns_noise_dist(const float *mean39);

#ifdef __cplusplus
}
#endif
