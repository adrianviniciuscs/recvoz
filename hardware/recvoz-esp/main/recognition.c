#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_check.h"
#include "mfcc.h"
#include "noise_sup.h"
#include "speaker_db.h"
#include "recvoz_display.h"
#include "recognition.h"
static const char *TAG = "recognition";
void recognition_run(const int16_t *pcm, float *mfcc, int hold_ms)
{
    int nf = mfcc_process_window(pcm, mfcc);
    if (nf <= 0) { ESP_LOGE(TAG, "mfcc falhou"); return; }
    float mean[MFCC_DIM];
    mfcc_mean_vector(mfcc, nf, mean);
    float vmod = mfcc_frame_modulation(mfcc, nf);
    float min_mod = mfcc_min_voice_mod();
    if (vmod < min_mod) {
        mfcc_noise_calibrate(pcm, NS_CAL_SAMPLES);
    }
    float d_ns = ns_noise_dist(mean);
    float llr1 = 0.0f, llr2 = 0.0f;
    spk_result_t r;
    int idx = speaker_db_classify_gmm(mfcc, nf, &r, &llr1, &llr2);
    int is_noise = (vmod < min_mod) || (vmod < 48.0f && d_ns <= r.dmin);
    if (is_noise) idx = -1;
    const char *wn = idx >= 0 ? speaker_db_name(idx) : "?";
    const char *near = is_noise ? "RUIDO" : (r.nearest >= 0 ? speaker_db_name(r.nearest) : "?");
    const char *rn = r.runner >= 0 ? speaker_db_name(r.runner) : "?";
    float tau = speaker_db_get_tau();
    if (is_noise)
        ESP_LOGI(TAG, "RUIDO AMBIENTE rejeitado (mod=%.1f d_ns=%.4f d_min=%.4f tau=%.4f)",
                 vmod, d_ns, r.dmin, tau);
    else if (idx < 0)
        ESP_LOGI(TAG, "DESCONHECIDO (quase %s) llr=%.3f d=%.4f (2o %s llr=%.3f d=%.4f, tau=%.4f, mod=%.1f)",
                 near, llr1, r.dmin, rn, llr2, r.drunner, tau, vmod);
    else
        ESP_LOGI(TAG, "%s llr=%.3f d=%.4f (2o %s llr=%.3f d=%.4f, tau=%.4f, margem=%.3f, mod=%.1f)",
                 wn, llr1, r.dmin, rn, llr2, r.drunner, tau, llr1 - llr2, vmod);
    // Placar ASCII no monitor (espelho do display, 1:1).
    {
        float maxv = r.drunner > tau ? r.drunner : tau;
        maxv *= 1.15f;
        if (maxv < 0.05f) maxv = 0.05f;
        int fill = (int)(30 * r.dmin / maxv), tick = (int)(30 * tau / maxv);
        if (fill > 30) fill = 30;
        if (tick > 30) tick = 30;
        char bar[32];
        for (int i = 0; i < 30; i++) bar[i] = i < fill ? '#' : (i == tick ? '|' : '-');
        bar[30] = 0;
        printf("+------------------------------+\n");
        printf("| 1 %-10.10s %7.3f |\n", idx >= 0 ? wn : near, r.dmin);
        printf("| 2 %-10.10s %7.3f |\n", rn, r.runner >= 0 ? r.drunner : -1.0f);
        printf("| [%s] |\n", bar);
        printf("| TAU %.3f => %s |\n", tau, idx >= 0 ? "CONHECIDO :-)" : "DESCONHECIDO :-(");
        printf("+------------------------------+\n");
    }
    // Carinha 3/5 do tempo + placar top-2 no resto (didático).
    // No DESCONHECIDO o placar mostra quem ele QUASE foi (nearest).
    if (idx < 0) ESP_ERROR_CHECK(recvoz_display_unknown());
    else ESP_ERROR_CHECK(recvoz_display_happy(wn));
    vTaskDelay(pdMS_TO_TICKS(hold_ms * 3 / 5));
    ESP_ERROR_CHECK(recvoz_display_score(idx >= 0 ? wn : near, r.dmin, rn, r.runner >= 0 ? r.drunner : -1, tau));
    vTaskDelay(pdMS_TO_TICKS(hold_ms - hold_ms * 3 / 5));
}

