// recvoz_display — robozinho OLED SSD1306 128x64 (ESP-IDF, sem Arduino/LVGL).
//
// Olhos paramétricos inspirados nos presets do esp32-eyes
// (playfultechnology / Luis Llamas, GPL-3.0): Normal 40x40, Happy 40x10,
// Sad 40x15 com slope. Fonte 5x7 clássica Adafruit (vide font5x7.h).
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_check.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_random.h"
#include "driver/i2c_master.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "recvoz_display.h"
#include "font5x7.h"

static const char *TAG = "recvoz_display";

#define LCD_H_RES 128
#define LCD_V_RES 64

// ---------------- estado ----------------
static esp_lcd_panel_handle_t s_panel;
static uint8_t s_fb[LCD_H_RES * LCD_V_RES / 8];
static SemaphoreHandle_t s_lock;
static recvoz_face_t s_face = RECVOZ_FACE_LISTENING;
static char s_label[32] = "OUVINDO";
static int s_mode; // 0 = carinha, 1 = placar top-2
static char s_w[13], s_r[13];
static float s_dmin, s_drunner, s_tau;

typedef struct { float w, h, r_top, r_bot; } eye_t;
static const eye_t EYE_NORMAL = {34, 30, 8, 8};
static const eye_t EYE_HAPPY  = {36, 10, 9, 0};
static const eye_t EYE_SAD    = {34, 14, 1, 9};
static eye_t s_eye_l = {34, 30, 8, 8}, s_eye_r = {34, 30, 8, 8};
static float s_slope_l, s_slope_r;

// ---------------- primitivas 1bpp (byte = 8px verticais) ----------------
static inline void px(int x, int y, int on)
{
    if (x < 0 || x >= LCD_H_RES || y < 0 || y >= LCD_V_RES) return;
    uint8_t *b = &s_fb[x + LCD_H_RES * (y / 8)];
    if (on) *b |= (1 << (y % 8));
    else *b &= ~(1 << (y % 8));
}
static void line(int x0, int y0, int x1, int y1)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1, err = dx + dy;
    while (1) {
        px(x0, y0, 1);
        if (x0 == x1 && y0 == y1) break;
        int e = 2 * err;
        if (e >= dy) { err += dy; x0 += sx; }
        if (e <= dx) { err += dx; y0 += sy; }
    }
}
static void rect(int x0, int y0, int x1, int y1)
{
    for (int x = x0; x <= x1; x++) { px(x, y0, 1); px(x, y1, 1); }
    for (int y = y0; y <= y1; y++) { px(x0, y, 1); px(x1, y, 1); }
}
static void circle_fill(int cx, int cy, int r)
{
    for (int y = -r; y <= r; y++)
        for (int x = -r; x <= r; x++)
            if (x * x + y * y <= r * r) px(cx + x, cy + y, 1);
}
static void arc(int cx, int cy, int r, int a0, int a1) // 0=dir, 90=baixo (y p/ baixo)
{
    for (int a = a0; a <= a1; a += 2) {
        float rad = a * 3.14159f / 180.0f;
        px(cx + (int)(r * cosf(rad)), cy + (int)(r * sinf(rad)), 1);
    }
}

// ---------------- texto 5x7 ----------------
static void draw_char(int x, int y, char c, int scale)
{
    if (c < 0x20 || c > 0x7E) c = '?';
    const uint8_t *g = &font5x7[(c - 0x20) * 5];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 7; j++)
            if (g[i] & (1 << j))
                for (int sy = 0; sy < scale; sy++)
                    for (int sx = 0; sx < scale; sx++)
                        px(x + i * scale + sx, y + j * scale + sy, 1);
}
static char fold_accent(uint8_t b2)
{
    if (b2 >= 0x80 && b2 <= 0x85) return 'A';
    if (b2 == 0x87) return 'C';
    if (b2 >= 0x88 && b2 <= 0x8B) return 'E';
    if (b2 >= 0x8C && b2 <= 0x8F) return 'I';
    if (b2 == 0x91) return 'N';
    if (b2 >= 0x92 && b2 <= 0x96) return 'O';
    if (b2 >= 0x99 && b2 <= 0x9C) return 'U';
    if (b2 >= 0xA0 && b2 <= 0xA5) return 'A';
    if (b2 == 0xA7) return 'C';
    if (b2 >= 0xA8 && b2 <= 0xAB) return 'E';
    if (b2 >= 0xAC && b2 <= 0xAF) return 'I';
    if (b2 == 0xB1) return 'N';
    if (b2 >= 0xB2 && b2 <= 0xB6) return 'O';
    if (b2 >= 0xB9 && b2 <= 0xBC) return 'U';
    return '?';
}
static void draw_status(const char *s)
{
    char tmp[32]; int n = 0;
    for (int i = 0; s[i] && n < 31; i++) {
        uint8_t b = s[i];
        if (b < 0x80) tmp[n++] = (b >= 'a' && b <= 'z') ? b - 32 : b;
        else if (b == 0xC3 && s[i + 1]) { tmp[n++] = fold_accent(s[i + 1]); i++; }
    }
    tmp[n] = 0;
    int scale = (n <= 8) ? 2 : 1;
    if (strlen(tmp) * 6 * scale - scale > LCD_H_RES) { tmp[21] = 0; scale = 1; }
    int w = strlen(tmp) * 6 * scale - scale;
    int x = (LCD_H_RES - w) / 2, y = (scale == 2) ? 1 : 3;
    const char *p = s;
    while (*p) {
        uint8_t b = *p;
        if (b < 0x80) {
            char c = b;
            if (c >= 'a' && c <= 'z') c -= 32;
            draw_char(x, y, c, scale); x += 6 * scale; p++;
        } else if (b == 0xC3 && p[1]) {
            draw_char(x, y, fold_accent(p[1]), scale); x += 6 * scale; p += 2;
        } else p++;
    }
}

// Texto genérico alinhado (placar usa este; status usa o centralizado acima).
static void draw_text(int x, int y, const char *s, int scale)
{
    while (*s) {
        uint8_t b = (uint8_t)*s;
        if (b < 0x80) {
            char c = b;
            if (c >= 'a' && c <= 'z') c -= 32;
            draw_char(x, y, c, scale); x += 6 * scale; s++;
        } else if (b == 0xC3 && s[1]) {
            draw_char(x, y, fold_accent(s[1]), scale); x += 6 * scale; s += 2;
        } else s++;
    }
}
// Placar didático: nome e veredito em x2 (legível na demo), vice em x1,
// barra dmin vs TAU embaixo. Traço = TAU (valor exato no log serial).
static void render_score(const char *w, float dmin, const char *r, float drunner, float tau)
{
    char b[16], w10[11];
    memset(s_fb, 0, sizeof(s_fb));
    // nome do vencedor em grande (máx 10 chars = 118 px)
    strncpy(w10, w, 10);
    w10[10] = 0;
    draw_text(2, 1, w10, 2);
    // dmin + veredito em grande: "0.018 OK" / "0.030 NAO"
    snprintf(b, sizeof b, "%.3f %s", dmin, dmin > tau ? "NAO" : "OK");
    draw_text(2, 17, b, 2);
    // vice pequeno (cabe folgado em x1)
    char rline[24];
    if (drunner < 0) snprintf(rline, sizeof rline, "2 %s ---", r);
    else snprintf(rline, sizeof rline, "2 %s %.3f", r, drunner);
    rline[21] = 0;
    draw_text(2, 35, rline, 1);
    // barra: eixo x4..123, preenchido até dmin, traço no TAU
    float maxv = drunner > tau ? drunner : tau;
    maxv *= 1.15f;
    if (maxv < 0.05f) maxv = 0.05f;
    for (int x = 4; x <= 123; x++) px(x, 50, 1);
    int fill = 4 + (int)(119 * dmin / maxv);
    if (fill > 123) fill = 123;
    for (int x = 4; x <= fill; x++)
        for (int y = 47; y <= 49; y++) px(x, y, 1);
    int tick = 4 + (int)(119 * tau / maxv);
    if (tick > 123) tick = 123;
    for (int y = 44; y <= 52; y++) px(tick, y, 1);
}

// ---------------- olhos estilo Cozmo ----------------
static int eye_hit(float dx, float dy, const eye_t *e, float slope)
{
    float top = -e->h / 2 + slope * dx * 0.7f, bot = e->h / 2;
    if (dy < top || dy > bot) return 0;
    float l = -e->w / 2, r = e->w / 2;
    if (dx < l || dx > r) return 0;
    if (e->r_top > 0.5f && dy < top + e->r_top) {
        float cy = top + e->r_top, ex, ey;
        if (dx < l + e->r_top) {
            ex = dx - (l + e->r_top); ey = dy - cy;
            if (ex * ex + ey * ey > e->r_top * e->r_top) return 0;
        }
        if (dx > r - e->r_top) {
            ex = dx - (r - e->r_top); ey = dy - cy;
            if (ex * ex + ey * ey > e->r_top * e->r_top) return 0;
        }
    }
    if (e->r_bot > 0.5f && dy > bot - e->r_bot) {
        float cy = bot - e->r_bot, ex, ey;
        if (dx < l + e->r_bot) {
            ex = dx - (l + e->r_bot); ey = dy - cy;
            if (ex * ex + ey * ey > e->r_bot * e->r_bot) return 0;
        }
        if (dx > r - e->r_bot) {
            ex = dx - (r - e->r_bot); ey = dy - cy;
            if (ex * ex + ey * ey > e->r_bot * e->r_bot) return 0;
        }
    }
    return 1;
}
static void draw_eye(int cx, int cy, const eye_t *e, float slope)
{
    for (int y = cy - e->h / 2 - 8; y <= cy + e->h / 2 + 8; y++)
        for (int x = cx - e->w / 2 - 2; x <= cx + e->w / 2 + 2; x++)
            if (eye_hit(x - cx, y - cy, e, slope)) px(x, y, 1);
}

static void render(recvoz_face_t f, const char *label, int look_x, int blink)
{
    memset(s_fb, 0, sizeof(s_fb));
    if (label) draw_status(label);
    rect(14, 17, 114, 63);   // cabeça (faixa amarela y0..15 livre)
    rect(6, 30, 13, 46);     // orelhas
    rect(115, 30, 122, 46);

    eye_t el = s_eye_l, er = s_eye_r;
    float sl = s_slope_l, sr = s_slope_r;
    if (blink) { el.h = 2; er.h = 2; el.r_top = el.r_bot = er.r_top = er.r_bot = 0; }
    int cy = (f == RECVOZ_FACE_HAPPY) ? 34 : 36;
    draw_eye(38 + look_x, cy, &el, sl);
    draw_eye(90 + look_x, cy, &er, sr);

    if (f == RECVOZ_FACE_LISTENING) {
        for (int x = 48; x <= 80; x++) { px(x, 53, 1); px(x, 54, 1); }
        line(40, 49, 44, 53); line(40, 57, 44, 53);
        line(88, 49, 84, 53); line(88, 57, 84, 53);
    } else if (f == RECVOZ_FACE_HAPPY) {
        arc(64, 45, 14, 25, 155); arc(64, 45, 13, 25, 155);
        circle_fill(30, 47, 2); circle_fill(98, 47, 2);
    } else {
        arc(64, 61, 12, 205, 335); arc(64, 61, 11, 205, 335);
    }
}

// ---------------- task de animação ----------------
static void target_for(recvoz_face_t f, eye_t *l, eye_t *r, float *sl, float *sr)
{
    if (f == RECVOZ_FACE_HAPPY) { *l = EYE_HAPPY; *r = EYE_HAPPY; *sl = *sr = 0; }
    else if (f == RECVOZ_FACE_SAD) { *l = EYE_SAD; *r = EYE_SAD; *sl = -0.5f; *sr = 0.5f; }
    else { *l = EYE_NORMAL; *r = EYE_NORMAL; *sl = *sr = 0; }
}
static void anim_task(void *arg)
{
    int look_x = 0, look_step = 0;
    int64_t next_blink = 0, next_look = 0, blink_end = 0;
    while (1) {
        int64_t now = esp_timer_get_time() / 1000;
        recvoz_face_t f; char label[32];
        int mode; char nm_w[13], nm_r[13]; float dmin, drunner, tau;
        xSemaphoreTake(s_lock, portMAX_DELAY);
        f = s_face; strncpy(label, s_label, sizeof(label));
        label[sizeof(label) - 1] = 0;
        mode = s_mode;
        strncpy(nm_w, s_w, sizeof(nm_w)); strncpy(nm_r, s_r, sizeof(nm_r));
        dmin = s_dmin; drunner = s_drunner; tau = s_tau;
        xSemaphoreGive(s_lock);

        if (mode == 1) {
            render_score(nm_w, dmin, nm_r, drunner, tau);
            esp_lcd_panel_draw_bitmap(s_panel, 0, 0, LCD_H_RES, LCD_V_RES, s_fb);
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        eye_t tl, tr; float sl, sr;
        target_for(f, &tl, &tr, &sl, &sr);
        // morph suave
        s_eye_l.w += (tl.w - s_eye_l.w) * 0.25f; s_eye_l.h += (tl.h - s_eye_l.h) * 0.25f;
        s_eye_l.r_top += (tl.r_top - s_eye_l.r_top) * 0.25f; s_eye_l.r_bot += (tl.r_bot - s_eye_l.r_bot) * 0.25f;
        s_eye_r.w += (tr.w - s_eye_r.w) * 0.25f; s_eye_r.h += (tr.h - s_eye_r.h) * 0.25f;
        s_eye_r.r_top += (tr.r_top - s_eye_r.r_top) * 0.25f; s_eye_r.r_bot += (tr.r_bot - s_eye_r.r_bot) * 0.25f;
        s_slope_l += (sl - s_slope_l) * 0.25f; s_slope_r += (sr - s_slope_r) * 0.25f;

        int blink = 0;
        if (f == RECVOZ_FACE_LISTENING) {
            if (now >= next_blink) { blink_end = now + 150; next_blink = now + 2500 + (esp_random() % 2000); }
            if (now < blink_end) blink = 1;
            if (now >= next_look) {
                const int opts[] = {-5, 0, 0, 5, 0};
                look_x = opts[esp_random() % 5]; look_step = 0;
                next_look = now + 1800 + (esp_random() % 1500);
            }
            if (look_step < 4) look_step++;
        } else { look_x = 0; }

        render(f, label, look_x, blink);
        esp_lcd_panel_draw_bitmap(s_panel, 0, 0, LCD_H_RES, LCD_V_RES, s_fb);
        vTaskDelay(pdMS_TO_TICKS(33));
    }
}

// ---------------- API pública ----------------
esp_err_t recvoz_display_init(const recvoz_display_config_t *cfg)
{
    if (s_panel) return ESP_OK;
    recvoz_display_config_t c = cfg ? *cfg : RECVOZ_DISPLAY_CONFIG_DEFAULT();

    i2c_master_bus_handle_t bus;
    i2c_master_bus_config_t bus_cfg = {
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .i2c_port = c.port,
        .sda_io_num = c.sda,
        .scl_io_num = c.scl,
        .flags.enable_internal_pullup = true,
    };
    ESP_RETURN_ON_ERROR(i2c_new_master_bus(&bus_cfg, &bus), TAG, "I2C");

    uint8_t addrs[2]; int n = 0;
    if (c.addr) addrs[n++] = c.addr;
    else { addrs[n++] = 0x3C; addrs[n++] = 0x3D; }

    esp_lcd_panel_io_handle_t io = NULL;
    for (int i = 0; i < n; i++) {
        if (i2c_master_probe(bus, addrs[i], 50) != ESP_OK) {
            ESP_LOGI(TAG, "0x%02X sem resposta", addrs[i]);
            continue;
        }
        esp_lcd_panel_io_i2c_config_t io_cfg = {
            .dev_addr = addrs[i], .scl_speed_hz = c.pixel_clock_hz,
            .control_phase_bytes = 1, .lcd_cmd_bits = 8, .lcd_param_bits = 8, .dc_bit_offset = 6,
        };
        ESP_RETURN_ON_ERROR(esp_lcd_new_panel_io_i2c(bus, &io_cfg, &io), TAG, "IO 0x%02X", addrs[i]);
        esp_lcd_panel_ssd1306_config_t ssd = {.height = LCD_V_RES};
        esp_lcd_panel_dev_config_t pcfg = {.bits_per_pixel = 1, .reset_gpio_num = -1, .vendor_config = &ssd};
        if (esp_lcd_new_panel_ssd1306(io, &pcfg, &s_panel) == ESP_OK) {
            ESP_LOGI(TAG, "SSD1306 em 0x%02X", addrs[i]);
            break;
        }
        esp_lcd_panel_io_del(io); io = NULL;
    }
    ESP_RETURN_ON_FALSE(s_panel, ESP_ERR_NOT_FOUND, TAG, "SSD1306 não encontrado");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_reset(s_panel), TAG, "reset");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_init(s_panel), TAG, "init");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_disp_on_off(s_panel, true), TAG, "on");

    s_lock = xSemaphoreCreateMutex();
    ESP_RETURN_ON_FALSE(s_lock, ESP_ERR_NO_MEM, TAG, "mutex");
    xTaskCreate(anim_task, "recvoz_disp", 4096, NULL, 5, NULL);
    return ESP_OK;
}

esp_err_t recvoz_display_show(recvoz_face_t face, const char *label)
{
    if (!s_panel || !s_lock) return ESP_ERR_INVALID_STATE;
    const char *def = face == RECVOZ_FACE_HAPPY ? "OLA"
                    : face == RECVOZ_FACE_SAD ? "DESCONHECIDO" : "OUVINDO";
    xSemaphoreTake(s_lock, portMAX_DELAY);
    s_mode = 0;
    s_face = face;
    strncpy(s_label, label && *label ? label : def, sizeof(s_label) - 1);
    xSemaphoreGive(s_lock);
    return ESP_OK;
}

/** Placar top-2 (didático). Volta p/ carinha no próximo show(). */
esp_err_t recvoz_display_score(const char *wname, float dmin, const char *rname, float drunner, float tau)
{
    if (!s_panel || !s_lock) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(s_lock, portMAX_DELAY);
    s_mode = 1;
    snprintf(s_w, sizeof s_w, "%s", (wname && wname[0]) ? wname : "?");
    snprintf(s_r, sizeof s_r, "%s", (rname && rname[0]) ? rname : "?");
    s_dmin = dmin; s_drunner = drunner; s_tau = tau;
    xSemaphoreGive(s_lock);
    return ESP_OK;
}

esp_err_t recvoz_display_listening(void) { return recvoz_display_show(RECVOZ_FACE_LISTENING, "OUVINDO"); }
esp_err_t recvoz_display_happy(const char *name) { return recvoz_display_show(RECVOZ_FACE_HAPPY, name); }
esp_err_t recvoz_display_unknown(void) { return recvoz_display_show(RECVOZ_FACE_SAD, "DESCONHECIDO"); }
