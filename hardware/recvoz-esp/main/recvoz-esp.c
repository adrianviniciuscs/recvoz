// recvoz-esp — demo viva: OUVINDO -> ADRIAN/PEDRO/VISITANTE/DESCONHECIDO.
//
// Fluxo: VAD por energia -> janela 2 s -> MFCC 39 -> vetor médio ->
// k-NN (speaker_db) -> carinha no display. Botão GPIO6 (ou `enroll`)
// cadastra locutor novo (7 janelas de 2 s ~= 15 s) em NVS.
//
// Comandos seriais (115200): enroll/list/reset/tau/vad/hold/diag/ns/nscal
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "console.h"
#include "capture.h"
#include "recognition.h"
#include "command_args.h"
#include "esp_log.h"
#include "esp_check.h"
#include "esp_console.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "audio_in.h"
#include "mfcc.h"
#include "noise_sup.h"
#include "speaker_db.h"
#include "gmm_ubm.h"
#include "recvoz_display.h"

static const char *TAG = "recvoz";

#define VAD_HOT_N 3           // 300 ms de voz p/ disparar
#define ENROLL_WINS 7         // 7 x 2 s ~= 15 s
#define RESULT_HOLD_MS 5000   // quanto tempo o resultado fica no display

static int s_hold_ms = RESULT_HOLD_MS;

static int s_vad_thr = 300;

static int16_t s_win[AUDIO_IN_WINDOW_SAMPLES]; // 64 KB
static float s_m[MFCC_DIM * MFCC_N_FRAMES];    // ~20 KB

// Perfil de ruído acumulado e aplicado apenas pela tarefa principal.
static int16_t s_ns_buf[NS_CAL_SAMPLES];
static int s_ns_fill;
static int s_ns_pending;
static int s_ns_auto = 1; // 1 = auto-calibra no silêncio; 0 = só via `nscal`

// 1 = voz, 0 = botão, 2 = comando ou perfil pendente.
static int listen_trigger(void)
{
    static int16_t chunk[VAD_CHUNK];
    static int quiet_streak;
    int hot = 0;
    ESP_ERROR_CHECK(recvoz_display_listening());
    while (1) {
        if (console_pending() || s_ns_pending) return 2;
        if (gpio_get_level(BTN_GPIO) == 0) { s_ns_fill = 0; quiet_streak = 0; return 0; } // enroll
        ESP_ERROR_CHECK(audio_in_read(chunk, VAD_CHUNK));
        int r = capture_rms16(chunk, VAD_CHUNK);
        int eff_vad = capture_effective_vad(s_vad_thr);
        if (r > eff_vad) {
            s_ns_fill = 0;
            quiet_streak = 0;
            if (++hot >= VAD_HOT_N) break;
        } else {
            hot = 0;
            // Silêncio p/ o perfil de ruído: na 1ª calibração aceita até 4/5 do VAD
            // para calibrar rápido no boot; nas adaptações exige 2/3 do VAD e 2,5 s
            // de silêncio contínuo (25 blocos) para não vazar fala fraca no N[k].
            int ready = ns_is_ready();
            int ceil = ready ? (eff_vad * 2 / 3) : (s_vad_thr * 4 / 5);
            if (r <= ceil) {
                if (quiet_streak < 25) quiet_streak++;
                if (s_ns_auto && !s_ns_pending && s_ns_fill < NS_CAL_SAMPLES && (!ready || quiet_streak >= 25)) {
                    int cp = NS_CAL_SAMPLES - s_ns_fill;
                    if (cp > VAD_CHUNK) cp = VAD_CHUNK;
                    memcpy(s_ns_buf + s_ns_fill, chunk, cp * sizeof(int16_t));
                    s_ns_fill += cp;
                    if (s_ns_fill >= NS_CAL_SAMPLES) s_ns_pending = 1;
                }
            } else {
                quiet_streak = 0;
                s_ns_fill = 0;
            }
        }
    }
    s_ns_fill = 0;
    quiet_streak = 0;
    return 1;
}

// Contagem regressiva ao vivo do enroll (segundos de VOZ restantes).
static int s_enroll_banked;
static void enroll_prog(int got, int need)
{
    (void)need;
    int rem = (ENROLL_WINS * 2) - (s_enroll_banked + got) / 16000;
    if (rem < 1) rem = 1;
    char lbl[16];
    snprintf(lbl, sizeof lbl, "FALE %dS", rem);
    recvoz_display_show(RECVOZ_FACE_LISTENING, lbl);
}

static void do_enroll(const char *name)
{
    char who[SPK_NAME_LEN];
    snprintf(who, sizeof who, "%s", (name && name[0]) ? name : "VISITANTE");
    ESP_LOGI(TAG, "enroll %s: fale por ~15 s", who);
    // Estáticos: a main task tem só 3584 B de stack; m[] tem ~20 KB.
    // (stack overflow aqui = DoubleException + reset WDT, visto em teste)
    static float evecs[ENROLL_WINS][MFCC_DIM];
    static gmm_adapt_ctx_t gctx;
    static float gmm_means[GMM_UBM_M * GMM_UBM_D];
    gmm_adapt_init(&gctx);
    for (int w = 0; w < ENROLL_WINS; w++) {
        s_enroll_banked = w * AUDIO_IN_WINDOW_SAMPLES;
        enroll_prog(0, AUDIO_IN_WINDOW_SAMPLES);
        int tries = 0, r = 0;
        while ((r = capture_voiced(s_win, capture_effective_vad(s_vad_thr), 0, enroll_prog)) == 0 && tries < 2) {
            tries++;
            ESP_LOGI(TAG, "janela %d fraca, repetindo (%d/2)...", w + 1, tries);
            ESP_ERROR_CHECK(recvoz_display_show(RECVOZ_FACE_LISTENING, "FALE ALTO"));
        }
        if (r < 0) { ESP_LOGI(TAG, "enroll cancelado"); return; }
        if (r == 0) { ESP_LOGI(TAG, "enroll abortado: pouca voz"); return; }
        int64_t t0 = esp_timer_get_time();
        int nf = mfcc_process_window(s_win, s_m);
        ESP_LOGI(TAG, "mfcc %d frames em %lld ms", nf, (esp_timer_get_time() - t0) / 1000);
        if (nf <= 0) { ESP_LOGE(TAG, "mfcc falhou no enroll"); return; }
        mfcc_mean_vector(s_m, nf, evecs[w]);
        float mod = mfcc_frame_modulation(s_m, nf);
        float d_ns = ns_noise_dist(evecs[w]);
        if (mod < mfcc_min_voice_mod() || d_ns <= 0.030f) {
            ESP_LOGW(TAG, "janela %d parece ruído da sala (mod=%.1f d_ns=%.4f), repetindo...", w + 1, mod, d_ns);
            if (tries < 2) {
                tries++;
                ESP_ERROR_CHECK(recvoz_display_show(RECVOZ_FACE_LISTENING, "FALE ALTO"));
                w--;
                continue;
            }
            ESP_LOGI(TAG, "enroll abortado: somente ruído detectado");
            return;
        }
        gmm_adapt_add_window(&gctx, s_m, nf);
        ESP_LOGI(TAG, "enroll %d/%d (mod=%.1f d_ns=%.4f)", w + 1, ENROLL_WINS, mod, d_ns);
    }
    gmm_adapt_finish(&gctx, gmm_means);
    if (speaker_db_enroll_gmm(who, (const float *)evecs, ENROLL_WINS, gmm_means) == ESP_OK)
        ESP_ERROR_CHECK(recvoz_display_happy(who));
    vTaskDelay(pdMS_TO_TICKS(3000));
}

// ---------- console ----------
static int cmd_enroll(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && strlen(argv[1]) >= SPK_NAME_LEN)) {
        puts("uso: enroll [NOME de ate 15 caracteres]"); return 1;
    }
    do_enroll(argc > 1 ? argv[1] : "VISITANTE");
    return 0;
}
static int cmd_list(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("%d locutores (tau=%.4f):\n", speaker_db_count(), speaker_db_get_tau());
    for (int i = 0; i < speaker_db_count(); i++) printf("  %d: %s\n", i, speaker_db_name(i));
    return 0;
}
static int cmd_reset(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("%s\n", speaker_db_reset() == ESP_OK ? "extras apagados" : "falhou");
    return 0;
}
static int cmd_tau(int argc, char **argv)
{
    if (argc > 1) printf("%s\n", speaker_db_set_tau(strtof(argv[1], NULL)) == ESP_OK ? "tau ok" : "tau invalido");
    else printf("tau=%.4f\n", speaker_db_get_tau());
    return 0;
}
static int cmd_vad(int argc, char **argv)
{
    if (argc > 1) { s_vad_thr = atoi(argv[1]); printf("vad=%d\n", s_vad_thr); }
    else printf("vad=%d\n", s_vad_thr);
    return 0;
}
static int cmd_hold(int argc, char **argv)
{
    if (argc > 1) { s_hold_ms = atoi(argv[1]); printf("hold=%d ms\n", s_hold_ms); }
    else printf("hold=%d ms\n", s_hold_ms);
    return 0;
}
// Supressão de ruído: ns | ns on|off | ns auto on|off | ns alpha F | ns floor F | ns beta F
static int cmd_ns(int argc, char **argv)
{
    float a, f;
    if (argc == 1) {
        ns_get_params(&a, &f);
        printf("ns %s auto=%s pronto=%s alpha=%.2f floor=%.3f beta=%.2f\n",
               ns_is_enabled() ? "on" : "off", s_ns_auto ? "on" : "off",
               ns_is_ready() ? "sim" : "nao", a, f, ns_get_beta());
        return 0;
    }
    if (!strcmp(argv[1], "on")) { ns_set_enabled(1); printf("ns on\n"); }
    else if (!strcmp(argv[1], "off")) { ns_set_enabled(0); printf("ns off\n"); }
    else if (!strcmp(argv[1], "auto") && argc > 2) {
        s_ns_auto = !strcmp(argv[2], "on") ? 1 : 0;
        if (!s_ns_auto) s_ns_fill = s_ns_pending = 0;
        printf("ns auto %s\n", s_ns_auto ? "on" : "off");
    } else if (!strcmp(argv[1], "alpha") && argc > 2) {
        ns_get_params(&a, &f);
        printf("%s\n", ns_set_params(strtof(argv[2], NULL), f) == 0 ? "alpha ok" : "alpha invalido (0,4]");
    } else if (!strcmp(argv[1], "floor") && argc > 2) {
        ns_get_params(&a, &f);
        printf("%s\n", ns_set_params(a, strtof(argv[2], NULL)) == 0 ? "floor ok" : "floor invalido [0,0.5]");
    } else if (!strcmp(argv[1], "beta") && argc > 2) {
        ns_set_beta(strtof(argv[2], NULL));
        printf("beta=%.2f\n", ns_get_beta());
    } else {
        printf("uso: ns [on|off|auto on|off|alpha F|floor F|beta F]\n");
        return 1;
    }
    return 0;
}
// Força calibração agora: junta 0.5 s de SILÊNCIO (fique quieto!) e calibra.
// Aborta se entrar voz no meio, p/ não contaminar o perfil.
static int cmd_nscal(int argc, char **argv)
{
    (void)argc; (void)argv;
    static int16_t cal[NS_CAL_SAMPLES]; // estático: a tarefa principal tem pouca stack
    static int16_t chunk[VAD_CHUNK];
    printf("juntando 0.5 s de silencio... FIQUE QUIETO\n");
    int got = 0, guard = 0;
    while (got < NS_CAL_SAMPLES && guard++ < 100) {
        if (gpio_get_level(BTN_GPIO) == 0) { printf("cancelado\n"); return 1; }
        ESP_ERROR_CHECK(audio_in_read(chunk, VAD_CHUNK));
        int r = capture_rms16(chunk, VAD_CHUNK);
        if (r > s_vad_thr * 2 / 3) { printf("voz no meio (rms=%d), abortado\n", r); return 1; }
        int cp = NS_CAL_SAMPLES - got;
        if (cp > VAD_CHUNK) cp = VAD_CHUNK;
        memcpy(cal + got, chunk, cp * sizeof(int16_t));
        got += cp;
    }
    if (got < NS_CAL_SAMPLES) { printf("timeout\n"); return 1; }
    int nf = mfcc_noise_calibrate(cal, NS_CAL_SAMPLES);
    s_ns_fill = s_ns_pending = 0; // descarta perfil pendente anterior
    // Reporta o resultado da calibração.
    if (nf > 0) printf("ns calibrado (%d frames)\n", nf);
    else printf("ns falhou\n");
    return nf > 0 ? 0 : 1;
}
// Diagnóstico: junta 1 janela (só voz) e mostra blocos + distância a cada um.
// Fale contínuo por ~3 s após o "JUNTE".
static int cmd_diag(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("juntando 2 s de voz... FALE agora (contínuo)\n");
    int r = capture_voiced(s_win, s_vad_thr, 1, NULL);
    if (r <= 0) { printf(r < 0 ? "cancelado\n" : "pouca voz\n"); return 1; }
    int nf = mfcc_process_window(s_win, s_m);
    if (nf <= 0) { printf("mfcc falhou\n"); return 1; }
    float mean[MFCC_DIM];
    mfcc_mean_vector(s_m, nf, mean);
    printf("c0=%.1f | llr:", mean[0]);
    for (int i = 0; i < speaker_db_count(); i++)
        printf(" %s=%.3f", speaker_db_name(i), speaker_db_llr_to(i, s_m, nf));
    printf(" (tau=%.4f)\n", speaker_db_get_tau());
    float na, nf2;
    ns_get_params(&na, &nf2);
    printf("ns: %s pronto=%s alpha=%.2f floor=%.3f beta=%.2f\n",
           ns_is_enabled() ? "on" : "off", ns_is_ready() ? "sim" : "nao",
           na, nf2, ns_get_beta());
    return 0;
}

static int cmd_grab(int argc, char **argv)
{
    static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int sec = (argc > 1) ? atoi(argv[1]) : 2;
    if (sec < 1) sec = 1;
    if (sec > 3) sec = 3;
    int total = sec * 16000;
    int16_t *buf = (int16_t *)malloc((size_t)total * sizeof(int16_t));
    if (!buf) { printf("sem memoria para %d s\n", sec); return 1; }
    ESP_ERROR_CHECK(recvoz_display_show(RECVOZ_FACE_LISTENING, "GRAVANDO"));
    for (int got = 0; got < total; got += VAD_CHUNK) {
        int n = total - got;
        if (n > VAD_CHUNK) n = VAD_CHUNK;
        if (audio_in_read(buf + got, n) != ESP_OK) { free(buf); return 1; }
    }
    const uint8_t *raw = (const uint8_t *)buf;
    int nbytes = total * (int)sizeof(int16_t);
    printf("RECVOZ_B64_START %d\n", total);
    char line[76];
    int pos = 0;
    for (int i = 0; i < nbytes; i += 3) {
        uint32_t v = ((uint32_t)raw[i] << 16)
                   | ((i + 1 < nbytes) ? ((uint32_t)raw[i + 1] << 8) : 0)
                   | ((i + 2 < nbytes) ? (uint32_t)raw[i + 2] : 0);
        line[pos++] = B64[(v >> 18) & 0x3F];
        line[pos++] = B64[(v >> 12) & 0x3F];
        line[pos++] = (i + 1 < nbytes) ? B64[(v >> 6) & 0x3F] : '=';
        line[pos++] = (i + 2 < nbytes) ? B64[v & 0x3F] : '=';
        if (pos >= 72) {
            line[pos] = '\0';
            puts(line);
            pos = 0;
        }
    }
    if (pos > 0) { line[pos] = '\0'; puts(line); }
    puts("RECVOZ_B64_END");
    fflush(stdout);
    free(buf);
    return 0;
}

static void bench_stats(const double *a, int n, double *mean, double *sd)
{
    double s = 0.0, s2 = 0.0;
    for (int i = 0; i < n; i++) s += a[i];
    *mean = s / n;
    for (int i = 0; i < n; i++) { double d = a[i] - *mean; s2 += d * d; }
    *sd = n > 1 ? sqrt(s2 / (n - 1)) : 0.0;
}

static int cmd_bench(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 20;
    if (n < 1) n = 1;
    if (n > 50) n = 50;
    for (int i = 0; i < AUDIO_IN_WINDOW_SAMPLES; i++) {
        float t = (float)i / 16000.0f;
        float env = 0.5f + 0.5f * sinf(2.0f * 3.14159265f * 4.0f * t);
        float s = env * (0.45f * sinf(2.0f * 3.14159265f * 150.0f * t)
                       + 0.30f * sinf(2.0f * 3.14159265f * 450.0f * t)
                       + 0.15f * sinf(2.0f * 3.14159265f * 1200.0f * t));
        s_win[i] = (int16_t)(s * 12000.0f);
    }
    double t_bp[50], t_stft[50], t_gmm[50], t_e2e[50], t_map[50];
    int nspk = speaker_db_count();
    if (nspk < 1) nspk = 1;
    // Aquecimento (1 janela)
    int nf = mfcc_process_window(s_win, s_m);
    if (nf <= 0) { printf("bench: mfcc falhou (nf=%d, heap=%lu)\n", nf, (unsigned long)esp_get_free_heap_size()); return 1; }

    // 1) Mede pipeline completo + GMM-UBM + MAP (sem buffer extra no heap)
    for (int it = 0; it < n; it++) {
        int64_t t0 = esp_timer_get_time();
        nf = mfcc_process_window(s_win, s_m);
        int64_t t2 = esp_timer_get_time();
        float mean[MFCC_DIM];
        mfcc_mean_vector(s_m, nf, mean);
        (void)mfcc_frame_modulation(s_m, nf);
        (void)ns_noise_dist(mean);
        int64_t t3 = esp_timer_get_time();
        spk_result_t r;
        float llr1 = 0.0f, llr2 = 0.0f;
        (void)speaker_db_classify_gmm(s_m, nf, &r, &llr1, &llr2);
        int64_t t4 = esp_timer_get_time();

        t_stft[it] = (double)(t2 - t0) / 1000.0; // mfcc_total provisorio
        t_gmm[it] = (double)(t4 - t3) / 1000.0;
        t_e2e[it] = (double)(t4 - t0) / 1000.0;

        static gmm_adapt_ctx_t gctx;
        static float gmeans[GMM_UBM_M * GMM_UBM_D];
        int64_t tm0 = esp_timer_get_time();
        gmm_adapt_init(&gctx);
        for (int w = 0; w < ENROLL_WINS; w++) gmm_adapt_add_window(&gctx, s_m, nf);
        gmm_adapt_finish(&gctx, gmeans);
        int64_t tm1 = esp_timer_get_time();
        t_map[it] = (double)(tm1 - tm0) / 1000.0;
        vTaskDelay(1);
    }
    unsigned long heap_free = (unsigned long)esp_get_free_heap_size();
    unsigned long heap_min = (unsigned long)esp_get_minimum_free_heap_size();
    unsigned long stack_hwm = (unsigned long)(uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t));

    // 2) Mede filtfilt isolado (em seguida libera xf)
    float *xf = (float *)malloc(AUDIO_IN_WINDOW_SAMPLES * sizeof(float));
    if (!xf) { puts("sem memoria para medir filtfilt"); return 1; }
    for (int it = 0; it < n; it++) {
        for (int i = 0; i < AUDIO_IN_WINDOW_SAMPLES; i++) xf[i] = (float)s_win[i] / 32768.0f;
        int64_t t0 = esp_timer_get_time();
        mfcc_bandpass(xf, AUDIO_IN_WINDOW_SAMPLES);
        int64_t t1 = esp_timer_get_time();
        t_bp[it] = (double)(t1 - t0) / 1000.0;
        t_stft[it] = t_stft[it] > t_bp[it] ? (t_stft[it] - t_bp[it]) : 0.0;
    }
    free(xf);
    double m_bp, s_bp, m_stft, s_stft, m_gmm, s_gmm, m_e2e, s_e2e, m_map, s_map;
    bench_stats(t_bp, n, &m_bp, &s_bp);
    bench_stats(t_stft, n, &m_stft, &s_stft);
    bench_stats(t_gmm, n, &m_gmm, &s_gmm);
    bench_stats(t_e2e, n, &m_e2e, &s_e2e);
    bench_stats(t_map, n, &m_map, &s_map);
    puts("RECVOZ_BENCH_START");
    printf("N=%d NS=%d SPEAKERS=%d FRAMES=%d\n", n, ns_is_enabled(), nspk, nf);
    printf("FILTFILT_MS=%.3f %.3f\n", m_bp, s_bp);
    printf("STFT_MFCC_MS=%.3f %.3f\n", m_stft, s_stft);
    printf("GMM_TOTAL_MS=%.3f %.3f\n", m_gmm, s_gmm);
    printf("GMM_PER_SPK_MS=%.3f %.3f\n", m_gmm / nspk, s_gmm / nspk);
    printf("MAP_ENROLL7_MS=%.3f %.3f\n", m_map, s_map);
    printf("E2E_MS=%.3f %.3f\n", m_e2e, s_e2e);
    printf("HEAP_FREE_B=%lu\n", heap_free);
    printf("HEAP_MIN_B=%lu\n", heap_min);
    printf("STACK_HWM_B=%lu\n", stack_hwm);
    puts("RECVOZ_BENCH_END");
    fflush(stdout);
    return 0;
}

static int execute_command(int argc, char **argv)
{
    if (!command_args_valid(argc, argv)) { puts("argumentos invalidos; consulte help e docs/CALIBRACAO.md"); return 1; }
    if (!strcmp(argv[0], "enroll")) return cmd_enroll(argc, argv);
    if (!strcmp(argv[0], "list")) return cmd_list(argc, argv);
    if (!strcmp(argv[0], "reset")) return cmd_reset(argc, argv);
    if (!strcmp(argv[0], "tau")) return cmd_tau(argc, argv);
    if (!strcmp(argv[0], "vad")) return cmd_vad(argc, argv);
    if (!strcmp(argv[0], "hold")) return cmd_hold(argc, argv);
    if (!strcmp(argv[0], "diag")) return cmd_diag(argc, argv);
    if (!strcmp(argv[0], "ns")) return cmd_ns(argc, argv);
    if (!strcmp(argv[0], "nscal")) return cmd_nscal(argc, argv);
    if (!strcmp(argv[0], "grab")) return cmd_grab(argc, argv);
    if (!strcmp(argv[0], "bench")) return cmd_bench(argc, argv);
    return 1;
}

// O botão inicia matrícula após soltar; durante matrícula cancela.

void app_main(void)
{
    // Apaga o LED RGB built-in (WS2812 no GPIO48 na DevKit S3): pino
    // flutuante capta ruído que parece dado e acende o LED sozinho.
    gpio_config_t led = {.pin_bit_mask = 1ULL << 48, .mode = GPIO_MODE_OUTPUT,
                         .pull_up_en = GPIO_PULLUP_DISABLE,
                         .pull_down_en = GPIO_PULLDOWN_DISABLE};
    if (gpio_config(&led) == ESP_OK) gpio_set_level(48, 0);

    gpio_config_t bc = {.pin_bit_mask = 1ULL << BTN_GPIO, .mode = GPIO_MODE_INPUT,
                        .pull_up_en = GPIO_PULLUP_ENABLE};
    ESP_ERROR_CHECK(gpio_config(&bc));

    recvoz_display_config_t dc = RECVOZ_DISPLAY_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(recvoz_display_init(&dc));
    audio_in_config_t ac = AUDIO_IN_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(audio_in_init(&ac));
    ESP_ERROR_CHECK(speaker_db_init());
    ns_set_enabled(1); // supressão ligada desde o boot
    if (audio_in_read(s_ns_buf, NS_CAL_SAMPLES) == ESP_OK) {
        int nf = mfcc_noise_calibrate(s_ns_buf, NS_CAL_SAMPLES);
        if (nf > 0) ESP_LOGI(TAG, "ns: perfil inicial calibrado no boot (%d frames, rms=%d)", nf, ns_get_noise_rms());
    }
    console_init();
    ESP_LOGI(TAG, "recvoz pronto (%d locutores). Fale p/ classificar, GPIO%d p/ enroll.",
             speaker_db_count(), BTN_GPIO);

    while (1) {
        if (s_ns_pending) { // silêncio acumulado no idle
            int nf = mfcc_noise_calibrate(s_ns_buf, NS_CAL_SAMPLES);
            s_ns_fill = 0;
            s_ns_pending = 0;
            // Formato precisa ser literal (a macro ESP_LOGI concatena a
            // string) — nada de ternário escolhendo o formato.
            if (nf > 0) ESP_LOGI(TAG, "ns: perfil calibrado (%d frames)", nf);
            else ESP_LOGI(TAG, "ns: calibracao falhou");
        }
        console_process(execute_command);
        int trigger = listen_trigger();
        if (trigger == 2) continue;
        if (trigger == 0) {
            while (gpio_get_level(BTN_GPIO) == 0) vTaskDelay(pdMS_TO_TICKS(20));
            s_ns_fill = s_ns_pending = 0;
            do_enroll("VISITANTE");
            continue;
        }
        ESP_LOGI(TAG, "voz! juntando 2 s de fala...");
        int r = capture_voiced(s_win, capture_effective_vad(s_vad_thr), 0, NULL);
        if (r < 0) { // botão no meio: enroll
            while (gpio_get_level(BTN_GPIO) == 0) vTaskDelay(pdMS_TO_TICKS(20));
            do_enroll("VISITANTE");
            continue;
        }
        if (r == 0) {
            ESP_LOGI(TAG, "pouca voz (fale mais alto e contínuo)");
            continue;
        }
        recognition_run(s_win, s_m, s_hold_ms);
    }
}
