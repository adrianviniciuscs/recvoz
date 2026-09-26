// recvoz-esp — demo viva: OUVINDO -> ADRIAN/PEDRO/VISITANTE/DESCONHECIDO.
//
// Fluxo: VAD por energia -> janela 2 s -> MFCC 39 -> vetor médio ->
// k-NN (speaker_db) -> carinha no display. Botão GPIO6 (ou `enroll`)
// cadastra locutor novo (7 janelas de 2 s ~= 15 s) em NVS.
//
// Comandos seriais (115200): enroll [NOME] | list | reset | tau [V] | vad [V]
#include <ctype.h>
#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_check.h"
#include "esp_console.h"
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

static int s_vad_thr = 300;
static volatile int s_enroll_req; // 1 = console pediu enroll
static char s_enroll_name[16];

static int16_t s_win[AUDIO_IN_WINDOW_SAMPLES]; // 64 KB
static float s_m[MFCC_DIM * MFCC_N_FRAMES];    // ~20 KB

static int rms16(const int16_t *p, int n)
{
    long long s = 0;
    for (int i = 0; i < n; i++) s += (int)p[i] * p[i];
    return (int)sqrt((double)s / n);
}

// Aguarda voz e captura 1 janela de 2 s. Retorna 1 se botão foi apertado.
static int listen_window(void)
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
    ESP_LOGI(TAG, "voz! capturando 2 s...");
    ESP_ERROR_CHECK(audio_in_read(s_win, AUDIO_IN_WINDOW_SAMPLES));
    return 1;
}

static void do_classify(void)
{
    int nf = mfcc_process_window(s_win, s_m);
    if (nf <= 0) { ESP_LOGE(TAG, "mfcc falhou"); return; }
    float mean[MFCC_DIM], dist = 0;
    mfcc_mean_vector(s_m, nf, mean);
    int idx = speaker_db_classify(mean, &dist);
    if (idx < 0) {
        ESP_LOGI(TAG, "DESCONHECIDO (d=%.4f)", dist);
        ESP_ERROR_CHECK(recvoz_display_unknown());
    } else {
        ESP_LOGI(TAG, "reconhecido: %s (d=%.4f)", speaker_db_name(idx), dist);
        ESP_ERROR_CHECK(recvoz_display_happy(speaker_db_name(idx)));
    }
    vTaskDelay(pdMS_TO_TICKS(3000)); // segura o resultado no display
}

static void do_enroll(const char *name)
{
    char who[SPK_NAME_LEN];
    snprintf(who, sizeof who, "%s", (name && name[0]) ? name : "VISITANTE");
    ESP_LOGI(TAG, "enroll %s: fale por ~15 s", who);
    static float vecs[ENROLL_WINS][MFCC_DIM];
    float m[MFCC_DIM * MFCC_N_FRAMES];
    for (int w = 0; w < ENROLL_WINS; w++) {
        char lbl[16];
        snprintf(lbl, sizeof lbl, "FALE %dS", (ENROLL_WINS - w) * 2);
        ESP_ERROR_CHECK(recvoz_display_show(RECVOZ_FACE_LISTENING, lbl));
        ESP_ERROR_CHECK(audio_in_read(s_win, AUDIO_IN_WINDOW_SAMPLES));
        int nf = mfcc_process_window(s_win, m);
        if (nf <= 0) { ESP_LOGE(TAG, "mfcc falhou no enroll"); return; }
        mfcc_mean_vector(m, nf, vecs[w]);
        ESP_LOGI(TAG, "enroll %d/%d", w + 1, ENROLL_WINS);
    }
    if (speaker_db_enroll(who, (const float *)vecs, ENROLL_WINS) == ESP_OK)
        ESP_ERROR_CHECK(recvoz_display_happy(who));
    vTaskDelay(pdMS_TO_TICKS(3000));
}

// ---------- console ----------
static int cmd_enroll(int argc, char **argv)
{
    snprintf(s_enroll_name, sizeof s_enroll_name, "%s", argc > 1 ? argv[1] : "VISITANTE");
    s_enroll_req = 1;
    printf("enroll %s agendado (pira o botão ou aguarde o loop)\n", s_enroll_name);
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
    };
    for (int i = 0; i < 5; i++) ESP_ERROR_CHECK(esp_console_cmd_register(&cmds[i]));
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}

// Botão: nível baixo = apertado (debounce simples).
static int btn_pressed(void)
{
    if (gpio_get_level(BTN_GPIO) != 0) return 0;
    vTaskDelay(pdMS_TO_TICKS(50));
    if (gpio_get_level(BTN_GPIO) != 0) return 0;
    while (gpio_get_level(BTN_GPIO) == 0) vTaskDelay(pdMS_TO_TICKS(20)); // espera soltar
    return 1;
}

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
    console_init();
    ESP_LOGI(TAG, "recvoz pronto (%d locutores). Fale p/ classificar, GPIO%d p/ enroll.",
             speaker_db_count(), BTN_GPIO);

    while (1) {
        if (btn_pressed()) { do_enroll("VISITANTE"); continue; }
        if (s_enroll_req) {
            char nm[16];
            snprintf(nm, sizeof nm, "%s", s_enroll_name);
            s_enroll_req = 0;
            do_enroll(nm);
            continue;
        }
        if (listen_window()) {
            do_classify();
        } else if (s_enroll_req) { // pedido via console durante a escuta
            char nm[16];
            snprintf(nm, sizeof nm, "%s", s_enroll_name);
            s_enroll_req = 0;
            do_enroll(nm);
        } else { // botão apertado durante a escuta
            do_enroll("VISITANTE");
        }
    }
}
