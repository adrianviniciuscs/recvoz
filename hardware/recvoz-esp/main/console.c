#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "esp_console.h"
#include "esp_check.h"
#include "console.h"

#define MAX_ARGS 4
#define ARG_LEN 32
typedef struct { int argc; char args[MAX_ARGS][ARG_LEN]; } request_t;
static QueueHandle_t requests, replies;

static int submit(int argc, char **argv)
{
    request_t request = {.argc = argc};
    if (argc > MAX_ARGS) { puts("argumentos demais"); return 1; }
    for (int i = 0; i < argc; i++) {
        if (strlen(argv[i]) >= ARG_LEN) { puts("argumento muito longo"); return 1; }
        strcpy(request.args[i], argv[i]);
    }
    if (xQueueSend(requests, &request, 0) != pdTRUE) { puts("console ocupado"); return 1; }
    int result;
    xQueueReceive(replies, &result, portMAX_DELAY);
    return result;
}

bool console_pending(void) { return uxQueueMessagesWaiting(requests) > 0; }

void console_process(recvoz_command_fn execute)
{
    request_t request;
    if (xQueueReceive(requests, &request, 0) != pdTRUE) return;
    char *argv[MAX_ARGS];
    for (int i = 0; i < request.argc; i++) argv[i] = request.args[i];
    int result = execute(request.argc, argv);
    xQueueSend(replies, &result, portMAX_DELAY);
}

void console_init(void)
{
    requests = xQueueCreate(1, sizeof(request_t));
    replies = xQueueCreate(1, sizeof(int));
    assert(requests && replies);
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t rc = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    rc.prompt = "recvoz>";
    esp_console_dev_uart_config_t uc = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&uc, &rc, &repl));
    const esp_console_cmd_t cmds[] = {
        {.command = "enroll", .help = "cadastra locutor: enroll [NOME]", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "list", .help = "lista locutores", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "reset", .help = "apaga extras (volta p/ fabrica)", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "tau", .help = "limiar desconhecido: tau [V]", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "vad", .help = "limiar de voz (rms): vad [V]", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "hold", .help = "tempo do resultado no display (ms): hold [MS]", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "diag", .help = "captura 2 s e mostra rms + distancias", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "ns", .help = "supressao de ruido: ns [on|off|auto on|off|alpha F|floor F|beta F]", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "nscal", .help = "calibra o perfil de ruido agora (0.5 s de silencio)", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "grab", .help = "envia PCM cru em Base64 pela serial: grab [SEG=2 (1..3)]", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
        {.command = "bench", .help = "mede latencia (ms) e heap no ESP32-S3: bench [N=20 (1..50)]", .hint = NULL, .func = submit, .argtable = NULL, .func_w_context = NULL, .context = NULL},
    };
    for (size_t i = 0; i < sizeof(cmds) / sizeof(cmds[0]); i++)
        ESP_ERROR_CHECK(esp_console_cmd_register(&cmds[i]));
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}
