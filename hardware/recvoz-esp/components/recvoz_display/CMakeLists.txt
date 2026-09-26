idf_component_register(
    SRCS "recvoz_display.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_lcd esp_driver_i2c esp_driver_gpio freertos esp_timer
)
