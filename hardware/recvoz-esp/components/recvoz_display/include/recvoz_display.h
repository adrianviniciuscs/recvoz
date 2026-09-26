#pragma once
#include "esp_err.h"
#include "driver/gpio.h"
#include "hal/i2c_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Carinhas do robozinho. */
typedef enum {
    RECVOZ_FACE_LISTENING, ///< neutro, escutando (olho Normal do Cozmo)
    RECVOZ_FACE_HAPPY,     ///< reconheceu :) (olho Happy/Glee do Cozmo)
    RECVOZ_FACE_SAD,       ///< desconhecido :( (olho Sad do Cozmo)
} recvoz_face_t;

typedef struct {
    i2c_port_t port;      ///< porta I2C (padrão 0)
    gpio_num_t sda;       ///< pino SDA
    gpio_num_t scl;       ///< pino SCL
    uint8_t addr;         ///< 0 = auto (tenta 0x3C, 0x3D)
    int pixel_clock_hz;   ///< clock I2C (padrão 400000)
} recvoz_display_config_t;

#define RECVOZ_DISPLAY_CONFIG_DEFAULT() \
    (recvoz_display_config_t){ .port = 0, .sda = 5, .scl = 4, .addr = 0, .pixel_clock_hz = 400 * 1000 }

/**
 * @brief Inicializa barramento I2C + SSD1306 128x64 e sobe a task de animação.
 *
 * Chamar uma vez no boot. Depois é só usar recvoz_display_show_*().
 */
esp_err_t recvoz_display_init(const recvoz_display_config_t *cfg);

/**
 * @brief Mostra uma carinha com texto (não-bloqueante; a transição é animada).
 *
 * @param face  qual carinha
 * @param label texto da faixa amarela (NULL = padrão de cada face).
 *              Aceita minúsculas e acentos PT ("Adrián", "João" -> "ADRIAN", "JOAO").
 *              Nome curto (<=8) sai grande x2; longo sai x1 (trunca em 21).
 */
esp_err_t recvoz_display_show(recvoz_face_t face, const char *label);

/** @brief Atalhos: escutando / reconheceu(nome) / desconhecido. */
esp_err_t recvoz_display_listening(void);
esp_err_t recvoz_display_happy(const char *name);
esp_err_t recvoz_display_unknown(void);

#ifdef __cplusplus
}
#endif
