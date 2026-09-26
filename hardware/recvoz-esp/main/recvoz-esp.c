// Demo do componente recvoz_display.
// O uso real é só: recvoz_display_init() + recvoz_display_show_*().
// Ex.: no loop do reconhecedor -> recvoz_display_happy("MARIA").
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "recvoz_display.h"

static const char *TAG = "recvoz-esp";

void app_main(void)
{
    recvoz_display_config_t cfg = RECVOZ_DISPLAY_CONFIG_DEFAULT();
    // cfg.sda = 5; cfg.scl = 4; // ajuste conforme sua fiação
    ESP_ERROR_CHECK(recvoz_display_init(&cfg));

    while (1) {
        ESP_LOGI(TAG, "OUVINDO");
        ESP_ERROR_CHECK(recvoz_display_listening());
        vTaskDelay(pdMS_TO_TICKS(4000));

        ESP_LOGI(TAG, "RECONHECIDO: ADRIAN");
        ESP_ERROR_CHECK(recvoz_display_happy("ADRIAN"));
        vTaskDelay(pdMS_TO_TICKS(3000));

        ESP_LOGI(TAG, "DESCONHECIDO");
        ESP_ERROR_CHECK(recvoz_display_unknown());
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}
