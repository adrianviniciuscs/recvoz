#pragma once
#include <assert.h>
#include <stdio.h>
#include "esp_err.h"
#define ESP_RETURN_ON_ERROR(call, tag, ...) do { (void)(tag); esp_err_t e_ = (call); if (e_) return e_; } while (0)
#define ESP_RETURN_ON_FALSE(cond, err, tag, ...) do { (void)(tag); if (!(cond)) return err; } while (0)
#define ESP_ERROR_CHECK(call) assert((call) == ESP_OK)
