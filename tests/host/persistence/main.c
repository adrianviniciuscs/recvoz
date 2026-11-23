#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "nvs.h"
#include "speaker_db.h"
static int fail_commit, fail_write;
esp_err_t nvs_flash_init(void) { return ESP_OK; }
esp_err_t nvs_flash_erase(void) { return ESP_OK; }
esp_err_t nvs_open(const char *s, int mode, nvs_handle_t *h) { (void)s; (void)mode; *h=1; return ESP_OK; }
void nvs_close(nvs_handle_t h) { (void)h; }
esp_err_t nvs_get_u8(nvs_handle_t h,const char*k,uint8_t*v) { (void)h;(void)k;(void)v;return ESP_ERR_NVS_NOT_FOUND; }
esp_err_t nvs_get_u32(nvs_handle_t h,const char*k,uint32_t*v) { (void)h;(void)k;(void)v;return ESP_ERR_NVS_NOT_FOUND; }
esp_err_t nvs_get_str(nvs_handle_t h,const char*k,char*v,size_t*n) { (void)h;(void)k;(void)v;(void)n;return ESP_ERR_NVS_NOT_FOUND; }
esp_err_t nvs_get_blob(nvs_handle_t h,const char*k,void*v,size_t*n) { (void)h;(void)k;(void)v;(void)n;return ESP_ERR_NVS_NOT_FOUND; }
esp_err_t nvs_set_u8(nvs_handle_t h,const char*k,uint8_t v) { (void)h;(void)k;(void)v;return fail_write ? ESP_FAIL : ESP_OK; }
esp_err_t nvs_set_u32(nvs_handle_t h,const char*k,uint32_t v) { (void)h;(void)k;(void)v;return fail_write ? ESP_FAIL : ESP_OK; }
esp_err_t nvs_set_str(nvs_handle_t h,const char*k,const char*v) { (void)h;(void)k;(void)v;return fail_write ? ESP_FAIL : ESP_OK; }
esp_err_t nvs_set_blob(nvs_handle_t h,const char*k,const void*v,size_t n) { (void)h;(void)k;(void)v;(void)n;return fail_write ? ESP_FAIL : ESP_OK; }
esp_err_t nvs_erase_key(nvs_handle_t h,const char*k) { (void)h;(void)k;return fail_write ? ESP_FAIL : ESP_OK; }
esp_err_t nvs_commit(nvs_handle_t h) { (void)h;return fail_commit ? ESP_FAIL : ESP_OK; }
int main(void) {
    assert(speaker_db_init() == ESP_OK);
    int initial = speaker_db_count();
    float v[SPK_DIM] = {1, 2, 3};
    fail_commit = 1;
    assert(speaker_db_enroll("NEW", v, 1) != ESP_OK);
    assert(speaker_db_count() == initial);
    fail_commit = 0; fail_write = 1;
    assert(speaker_db_enroll("NEW", v, 1) != ESP_OK);
    assert(speaker_db_count() == initial);
    fail_write = 0;
    assert(speaker_db_enroll("NEW", v, 1) == ESP_OK);
    assert(speaker_db_count() == initial + 1);
    fail_commit = 1;
    assert(speaker_db_reset() != ESP_OK);
    assert(speaker_db_count() == initial + 1);
    float tau = speaker_db_get_tau();
    assert(speaker_db_set_tau(.1f) != ESP_OK);
    assert(speaker_db_get_tau() == tau);
    fail_commit = 0; v[1] = NAN;
    assert(speaker_db_enroll("BAD", v, 1) != ESP_OK);
    assert(speaker_db_reset() == ESP_OK);
    assert(speaker_db_count() == initial);
    puts("PASS persistence failures preserve RAM state");
}
