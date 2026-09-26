// recvoz-esp — demo viva: OUVINDO -> ADRIAN/PEDRO/VISITANTE/DESCONHECIDO.
//
// Fluxo: VAD por energia -> janela 2 s -> MFCC 39 -> vetor médio ->
// k-NN (speaker_db) -> carinha no display. Botão GPIO6 (ou `enroll`)
// cadastra locutor novo (7 janelas de 2 s ~= 15 s) em NVS.
//
// Comandos seriais (115200): enroll/list/reset/tau/vad/diag
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_check.h"
#include "esp_console.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "audio_in.h"
#include "mfcc.h"
#include "speaker_db.h"
#include "recvoz_display.h"

static const char *TAG = "recvoz";

#define BTN_GPIO 6            // botão enroll p/ GND (pull-up interno)
#define VAD_CHUNK 1600        // 100 ms @16k
#define VAD_HOT_N 3           // 300 ms de voz p/ disparar
#define ENROLL_WINS 7         // 7 x 2 s ~= 15 s
#define CAP_ITERS 100         // 10 s máx p/ juntar 2 s de voz
#define RESULT_HOLD_MS 5000   // quanto tempo o resultado fica no display

static int s_hold_ms = RESULT_HOLD_MS;

static int s_vad_thr = 300;
static volatile int s_enroll_req; // 1 = console pediu enroll
static char s_enroll_name[16];

static int16_t s_win[AUDIO_IN_WINDOW_SAMPLES]; // 64 KB
static float s_m[MFCC_DIM * MFCC_N_FRAMES];    // ~20 KB

// Trava o DSP: mfcc_process_window aloca ~150 KB transientes e usa os
// buffers estáticos acima — diag (task console) x loop principal não
// podem rodar DSP juntos (causa "mfcc falhou" por OOM + corrupção).
static SemaphoreHandle_t s_dsp;

static int rms16(const int16_t *p, int n)
{
    long long s = 0;
    for (int i = 0; i < n; i++) s += (int)p[i] * p[i];
    return (int)sqrt((double)s / n);
}

// Junta 2 s SÓ de blocos acima do VAD (protótipo; definição abaixo).
// prog(got, need) é chamado a cada ~0.4 s p/ barra de progresso (pode ser NULL).
static int capture_voiced(int16_t *out, int trace, void (*prog)(int, int));

// Aguarda voz (VAD, buffer local — seguro sem trava). 1 = voz, 0 = enroll.
static int listen_trigger(void)
{
    static int16_t chunk[VAD_CHUNK];
    int hot = 0;
    ESP_ERROR_CHECK(recvoz_display_listening());
    while (1) {
        if (gpio_get_level(BTN_GPIO) == 0 || s_enroll_req) return 0; // enroll
        ESP_ERROR_CHECK(audio_in_read(chunk, VAD_CHUNK));
        if (rms16(chunk, VAD_CHUNK) > s_vad_thr) {
            if (++hot >= VAD_HOT_N) break;
        } else hot = 0;
    }
    return 1;
}

// Junta 2 s SÓ de blocos acima do VAD (pula silêncio/reação), p/ a janela
// não diluir a média com quietude. 1 = ok, 0 = timeout, -1 = botão.
static int capture_voiced(int16_t *out, int trace, void (*prog)(int, int))
{
    static int16_t chunk[VAD_CHUNK];
    int got = 0, reported = -1;
    for (int it = 0; it < CAP_ITERS && got < AUDIO_IN_WINDOW_SAMPLES; it++) {
        if (gpio_get_level(BTN_GPIO) == 0) return -1;
        ESP_ERROR_CHECK(audio_in_read(chunk, VAD_CHUNK));
        int r = rms16(chunk, VAD_CHUNK);
        if (trace) printf(r > s_vad_thr ? "+%d " : ".%d ", r);
        if (r > s_vad_thr) {
            memcpy(out + got, chunk, sizeof(chunk));
            got += VAD_CHUNK;
            if (prog && got - reported >= 6400) { reported = got; prog(got, AUDIO_IN_WINDOW_SAMPLES); }
        }
    }
    if (trace) printf("\n  voz: %d/%d blocos\n", got / VAD_CHUNK, AUDIO_IN_WINDOW_SAMPLES / VAD_CHUNK);
    if (prog) prog(got, AUDIO_IN_WINDOW_SAMPLES);
    return got >= AUDIO_IN_WINDOW_SAMPLES ? 1 : 0;
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

static void do_classify(void) // s_dsp travado pelo chamador
{
    int nf = mfcc_process_window(s_win, s_m);
    if (nf <= 0) { ESP_LOGE(TAG, "mfcc falhou"); return; }
    float mean[MFCC_DIM];
    mfcc_mean_vector(s_m, nf, mean);
    spk_result_t r;
    int idx = speaker_db_classify_full(mean, &r);
    const char *wn = idx >= 0 ? speaker_db_name(idx) : "?";
    const char *near = r.nearest >= 0 ? speaker_db_name(r.nearest) : "?";
    const char *rn = r.runner >= 0 ? speaker_db_name(r.runner) : "?";
    float tau = speaker_db_get_tau();
    if (idx < 0)
        ESP_LOGI(TAG, "DESCONHECIDO (quase %s) d=%.4f (2o %s %.4f, tau=%.4f)",
                 near, r.dmin, rn, r.drunner, tau);
    else
        ESP_LOGI(TAG, "%s d=%.4f (2o %s %.4f, tau=%.4f, margem=%.4f)",
                 wn, r.dmin, rn, r.drunner, tau, r.drunner - r.dmin);
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
    vTaskDelay(pdMS_TO_TICKS(s_hold_ms * 3 / 5));
    ESP_ERROR_CHECK(recvoz_display_score(idx >= 0 ? wn : near, r.dmin, rn, r.runner >= 0 ? r.drunner : -1, tau));
    vTaskDelay(pdMS_TO_TICKS(s_hold_ms - s_hold_ms * 3 / 5));
}

static void do_enroll(const char *name)
{
    char who[SPK_NAME_LEN];
    snprintf(who, sizeof who, "%s", (name && name[0]) ? name : "VISITANTE");
    ESP_LOGI(TAG, "enroll %s: fale por ~15 s", who);
    // Estáticos: a main task tem só 3584 B de stack; m[] tem ~20 KB.
    // (stack overflow aqui = DoubleException + reset WDT, visto em teste)
    static float evecs[ENROLL_WINS][MFCC_DIM];
    xSemaphoreTake(s_dsp, portMAX_DELAY);
    for (int w = 0; w < ENROLL_WINS; w++) {
        s_enroll_banked = w * AUDIO_IN_WINDOW_SAMPLES;
        enroll_prog(0, AUDIO_IN_WINDOW_SAMPLES);
        int tries = 0, r = 0;
        while ((r = capture_voiced(s_win, 0, enroll_prog)) == 0 && tries < 2) {
            tries++;
            ESP_LOGI(TAG, "janela %d fraca, repetindo (%d/2)...", w + 1, tries);
            ESP_ERROR_CHECK(recvoz_display_show(RECVOZ_FACE_LISTENING, "FALE ALTO"));
        }
        if (r < 0) { ESP_LOGI(TAG, "enroll cancelado"); xSemaphoreGive(s_dsp); return; }
        if (r == 0) { ESP_LOGI(TAG, "enroll abortado: pouca voz"); xSemaphoreGive(s_dsp); return; }
        int64_t t0 = esp_timer_get_time();
        int nf = mfcc_process_window(s_win, s_m); // s_m: sob trava, seguro
        ESP_LOGI(TAG, "mfcc %d frames em %lld ms", nf, (esp_timer_get_time() - t0) / 1000);
        if (nf <= 0) { ESP_LOGE(TAG, "mfcc falhou no enroll"); xSemaphoreGive(s_dsp); return; }
        mfcc_mean_vector(s_m, nf, evecs[w]);
        ESP_LOGI(TAG, "enroll %d/%d", w + 1, ENROLL_WINS);
    }
    xSemaphoreGive(s_dsp);
    if (speaker_db_enroll(who, (const float *)evecs, ENROLL_WINS) == ESP_OK)
        ESP_ERROR_CHECK(recvoz_display_happy(who));
    vTaskDelay(pdMS_TO_TICKS(3000));
}

// ---------- console ----------
static int cmd_enroll(int argc, char **argv)
{
    snprintf(s_enroll_name, sizeof s_enroll_name, "%s", argc > 1 ? argv[1] : "VISITANTE");
    s_enroll_req = 1;
    printf("enroll %s agendado (aperte o botão ou aguarde o loop)\n", s_enroll_name);
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
// Diagnóstico: junta 1 janela (só voz) e mostra blocos + distância a cada um.
// Fale contínuo por ~3 s após o "JUNTE".
static int cmd_diag(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("juntando 2 s de voz... FALE agora (contínuo)\n");
    xSemaphoreTake(s_dsp, portMAX_DELAY);
    int r = capture_voiced(s_win, 1, NULL);
    if (r <= 0) { printf(r < 0 ? "cancelado\n" : "pouca voz\n"); xSemaphoreGive(s_dsp); return 1; }
    int nf = mfcc_process_window(s_win, s_m);
    if (nf <= 0) { printf("mfcc falhou\n"); xSemaphoreGive(s_dsp); return 1; }
    float mean[MFCC_DIM];
    mfcc_mean_vector(s_m, nf, mean);
    printf("c0=%.1f | dists:", mean[0]);
    for (int i = 0; i < speaker_db_count(); i++)
        printf(" %s=%.4f", speaker_db_name(i), speaker_db_dist_to(i, mean));
    printf(" (tau=%.4f)\n", speaker_db_get_tau());
    xSemaphoreGive(s_dsp);
    return 0;
}

static void console_init(void)
{
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t rc = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    rc.prompt = "recvoz>";
    esp_console_dev_uart_config_t uc = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&uc, &rc, &repl));
    const esp_console_cmd_t cmds[] = {
        {.command = "enroll", .help = "cadastra locutor: enroll [NOME]", .hint = NULL, .func = cmd_enroll, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "list", .help = "lista locutores", .hint = NULL, .func = cmd_list, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "reset", .help = "apaga extras (volta p/ fabrica)", .hint = NULL, .func = cmd_reset, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "tau", .help = "limiar desconhecido: tau [V]", .hint = NULL, .func = cmd_tau, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "vad", .help = "limiar de voz (rms): vad [V]", .hint = NULL, .func = cmd_vad, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "hold", .help = "tempo do resultado no display (ms): hold [MS]", .hint = NULL, .func = cmd_hold, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "diag", .help = "captura 2 s e mostra rms + distancias", .hint = NULL, .func = cmd_diag, .argtable = NULL, .func_w_context = NULL, .context = NULL},
    };
    for (int i = 0; i < 7; i++) ESP_ERROR_CHECK(esp_console_cmd_register(&cmds[i]));
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}

// (botão lido por nível em listen_trigger/capture_voiced; sem espera soltar
//  p/ não travar o loop — o debounce é o poll de 100 ms dos chunks)

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
    s_dsp = xSemaphoreCreateMutex();
    assert(s_dsp);
    console_init();
    ESP_LOGI(TAG, "recvoz pronto (%d locutores). Fale p/ classificar, GPIO%d p/ enroll.",
             speaker_db_count(), BTN_GPIO);

    while (1) {
        if (!listen_trigger()) { // enroll pedido (botão ou console)
            if (s_enroll_req) {
                char nm[16];
                snprintf(nm, sizeof nm, "%s", s_enroll_name);
                s_enroll_req = 0;
                do_enroll(nm);
            } else {
                do_enroll("VISITANTE");
            }
            continue;
        }
        // Voz! Captura + DSP sob trava (diag não entra no meio).
        ESP_LOGI(TAG, "voz! juntando 2 s de fala...");
        xSemaphoreTake(s_dsp, portMAX_DELAY);
        if (s_enroll_req) { // pedido chegou durante o gatilho
            xSemaphoreGive(s_dsp);
            continue; // volta ao topo, que desvia p/ enroll
        }
        int r = capture_voiced(s_win, 0, NULL);
        if (r < 0) { // botão no meio: enroll
            xSemaphoreGive(s_dsp);
            do_enroll("VISITANTE");
            continue;
        }
        if (r == 0) {
            xSemaphoreGive(s_dsp);
            ESP_LOGI(TAG, "pouca voz (fale mais alto e contínuo)");
            continue;
        }
        do_classify(); // trava ainda presa; solta abaixo
        xSemaphoreGive(s_dsp);
    }
}
