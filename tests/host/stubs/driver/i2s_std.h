#pragma once
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
typedef void *i2s_chan_handle_t;
typedef struct { int dma_desc_num, dma_frame_num; } i2s_chan_config_t;
typedef struct {
    int clk_cfg, slot_cfg;
    struct { int mclk,bclk,ws,dout,din; int invert_flags[3]; } gpio_cfg;
} i2s_std_config_t;
#define I2S_CHANNEL_DEFAULT_CONFIG(a,b) (i2s_chan_config_t){0}
#define I2S_STD_CLK_DEFAULT_CONFIG(a) (a)
#define I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(a,b) 0
#define I2S_GPIO_UNUSED -1
#define false 0
esp_err_t i2s_new_channel(const i2s_chan_config_t *, void *, i2s_chan_handle_t *);
esp_err_t i2s_channel_init_std_mode(i2s_chan_handle_t, const i2s_std_config_t *);
esp_err_t i2s_channel_enable(i2s_chan_handle_t);
esp_err_t i2s_channel_read(i2s_chan_handle_t, void *, size_t, size_t *, unsigned);
