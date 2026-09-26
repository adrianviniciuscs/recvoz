// speaker_db — base de locutores: fábrica (flash) + extras (NVS) + k-NN.
//
// Fábrica = enroll_default.h (ADRIAN + PEDRO, 36 vetores), sempre disponível.
// Extras (ex enroll "VISITANTE") vão p/ NVS "recvoz": no máx 3 locutores x
// 8 vetores (blob de 1249 B cada — cabe folgado no limite de ~2 KB/entrada).
// k-NN replica o sklearn do notebook: k=3, distância cosseno, peso 1/d.
#include <math.h>
#include <string.h>
#include "esp_log.h"
#include "esp_check.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "speaker_db.h"
#include "enroll_default.h"

static const char *TAG = "speaker_db";
#define NS "recvoz"

typedef struct {
    char name[SPK_NAME_LEN];
    uint8_t nvec;
    float vecs[SPK_EXTRA_VECS][SPK_DIM];
    uint8_t used;
} extra_t;

static extra_t s_extra[SPK_EXTRA_MAX];
static float s_tau = SPK_TAU_DEFAULT;
static int s_ok;

static float cos_dist(const float *a, const float *b)
{
    double d = 0, na = 0, nb = 0;
    for (int i = 0; i < SPK_DIM; i++) {
        d += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na <= 0 || nb <= 0) return 1.0f;
    float c = (float)(d / (sqrt(na) * sqrt(nb)));
    if (c > 1) c = 1;
    if (c < -1) c = -1;
    return 1.0f - c;
}

// Nº de vetores de fábrica por locutor (conta via ENROLL_DEFAULT_LABELS).
static int factory_nvec(int spk)
{
    int n = 0;
    for (int i = 0; i < ENROLL_DEFAULT_NVEC; i++)
        if (ENROLL_DEFAULT_LABELS[i] == spk) n++;
    return n;
}

// Vetor j do locutor de fábrica spk.
static const float *factory_vec(int spk, int j)
{
    int k = 0;
    for (int i = 0; i < ENROLL_DEFAULT_NVEC; i++)
        if (ENROLL_DEFAULT_LABELS[i] == spk && k++ == j)
            return ENROLL_DEFAULT_VECS[i];
    return NULL;
}

static esp_err_t load_all(void)
{
    nvs_handle_t h;
    ESP_RETURN_ON_ERROR(nvs_open(NS, NVS_READONLY, &h), TAG, "nvs open");
    uint32_t bits = 0;
    size_t len = sizeof(bits);
    if (nvs_get_u32(h, "tau", &bits) == ESP_OK) {
        float t;
        memcpy(&t, &bits, 4);
        if (isfinite(t) && t > 0 && t < 1) s_tau = t;
    }
    memset(s_extra, 0, sizeof(s_extra));
    for (int i = 0; i < SPK_EXTRA_MAX; i++) {
        char kname[8], kvec[8];
        snprintf(kname, sizeof kname, "x%dn", i);
        snprintf(kvec, sizeof kvec, "x%dv", i);
        size_t nl = SPK_NAME_LEN;
        uint8_t blob[1 + SPK_EXTRA_VECS * SPK_DIM * sizeof(float)];
        size_t bl = sizeof(blob);
        if (nvs_get_str(h, kname, s_extra[i].name, &nl) != ESP_OK) continue;
        if (nvs_get_blob(h, kvec, blob, &bl) != ESP_OK) { s_extra[i].name[0] = 0; continue; }
        uint8_t n = blob[0];
        if (n == 0 || n > SPK_EXTRA_VECS || bl < 1 + n * SPK_DIM * sizeof(float)) {
            s_extra[i].name[0] = 0;
            continue;
        }
        s_extra[i].nvec = n;
        memcpy(s_extra[i].vecs, blob + 1, n * SPK_DIM * sizeof(float));
        s_extra[i].used = 1;
        ESP_LOGI(TAG, "extra[%d] %s (%d vetores)", i, s_extra[i].name, n);
    }
    nvs_close(h);
    return ESP_OK;
}

esp_err_t speaker_db_init(void)
{
    if (s_ok) return ESP_OK;
    esp_err_t r = nvs_flash_init();
    if (r == ESP_ERR_NVS_NO_FREE_PAGES || r == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        r = nvs_flash_init();
    }
    ESP_RETURN_ON_ERROR(r, TAG, "nvs_flash_init");
    ESP_RETURN_ON_ERROR(load_all(), TAG, "load");
    s_ok = 1;
    ESP_LOGI(TAG, "pronto: %d locutores (tau=%.4f)", speaker_db_count(), s_tau);
    return ESP_OK;
}

int speaker_db_count(void)
{
    int n = ENROLL_DEFAULT_NSPK;
    for (int i = 0; i < SPK_EXTRA_MAX; i++)
        if (s_extra[i].used) n++;
    return n;
}

const char *speaker_db_name(int idx)
{
    if (idx < ENROLL_DEFAULT_NSPK) return ENROLL_DEFAULT_NAMES[idx];
    int e = idx - ENROLL_DEFAULT_NSPK;
    if (e < SPK_EXTRA_MAX && s_extra[e].used) return s_extra[e].name;
    return "?";
}

esp_err_t speaker_db_enroll(const char *name, const float *vecs, int nvec)
{
    ESP_RETURN_ON_FALSE(s_ok && name && name[0] && vecs, ESP_ERR_INVALID_ARG, TAG, "arg");
    ESP_RETURN_ON_FALSE(nvec >= 1 && nvec <= SPK_EXTRA_VECS, ESP_ERR_INVALID_ARG, TAG, "nvec=%d", nvec);
    char clean[SPK_NAME_LEN];
    snprintf(clean, sizeof clean, "%s", name);
    for (char *p = clean; *p; p++) {
        if (*p >= 'a' && *p <= 'z') *p -= 32;
        if (!((*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9') || *p == ' ' || *p == '_')) *p = '_';
    }
    int slot = -1;
    for (int i = 0; i < SPK_EXTRA_MAX; i++)
        if (s_extra[i].used && strcmp(s_extra[i].name, clean) == 0) { slot = i; break; }
    if (slot < 0)
        for (int i = 0; i < SPK_EXTRA_MAX; i++)
            if (!s_extra[i].used) { slot = i; break; }
    ESP_RETURN_ON_FALSE(slot >= 0, ESP_ERR_NO_MEM, TAG, "sem slot extra livre");

    memcpy(s_extra[slot].name, clean, SPK_NAME_LEN);
    s_extra[slot].nvec = nvec;
    memcpy(s_extra[slot].vecs, vecs, nvec * SPK_DIM * sizeof(float));
    s_extra[slot].used = 1;

    nvs_handle_t h;
    ESP_RETURN_ON_ERROR(nvs_open(NS, NVS_READWRITE, &h), TAG, "nvs open");
    char kname[8], kvec[8];
    snprintf(kname, sizeof kname, "x%dn", slot);
    snprintf(kvec, sizeof kvec, "x%dv", slot);
    uint8_t blob[1 + SPK_EXTRA_VECS * SPK_DIM * sizeof(float)];
    blob[0] = nvec;
    memcpy(blob + 1, vecs, nvec * SPK_DIM * sizeof(float));
    esp_err_t r = nvs_set_str(h, kname, clean);
    if (r == ESP_OK) r = nvs_set_blob(h, kvec, blob, 1 + nvec * SPK_DIM * sizeof(float));
    if (r == ESP_OK) r = nvs_commit(h);
    nvs_close(h);
    ESP_RETURN_ON_ERROR(r, TAG, "nvs write");
    ESP_LOGI(TAG, "cadastrado %s no slot %d (%d vetores)", clean, slot, nvec);
    return ESP_OK;
}

esp_err_t speaker_db_reset(void)
{
    ESP_RETURN_ON_FALSE(s_ok, ESP_ERR_INVALID_STATE, TAG, "init primeiro");
    nvs_handle_t h;
    ESP_RETURN_ON_ERROR(nvs_open(NS, NVS_READWRITE, &h), TAG, "nvs open");
    for (int i = 0; i < SPK_EXTRA_MAX; i++) {
        char kname[8], kvec[8];
        snprintf(kname, sizeof kname, "x%dn", i);
        snprintf(kvec, sizeof kvec, "x%dv", i);
        nvs_erase_key(h, kname);
        nvs_erase_key(h, kvec);
    }
    esp_err_t r = nvs_commit(h);
    nvs_close(h);
    memset(s_extra, 0, sizeof(s_extra));
    ESP_RETURN_ON_ERROR(r, TAG, "nvs commit");
    return ESP_OK;
}

float speaker_db_get_tau(void) { return s_tau; }

esp_err_t speaker_db_set_tau(float tau)
{
    ESP_RETURN_ON_FALSE(isfinite(tau) && tau > 0 && tau < 1, ESP_ERR_INVALID_ARG, TAG, "tau");
    nvs_handle_t h;
    ESP_RETURN_ON_ERROR(nvs_open(NS, NVS_READWRITE, &h), TAG, "nvs open");
    uint32_t bits;
    memcpy(&bits, &tau, 4);
    esp_err_t r = nvs_set_u32(h, "tau", bits);
    if (r == ESP_OK) r = nvs_commit(h);
    nvs_close(h);
    ESP_RETURN_ON_ERROR(r, TAG, "nvs write");
    s_tau = tau;
    return ESP_OK;
}

int speaker_db_classify(const float *mean39, float *out_dist)
{
    if (!s_ok || !mean39) return -1;
    // Junta fábrica + extras numa lista plana (máx 60 vetores).
    static struct { float d; int spk; } cand[ENROLL_DEFAULT_NVEC + SPK_EXTRA_MAX * SPK_EXTRA_VECS];
    int nc = 0;
    for (int s = 0; s < ENROLL_DEFAULT_NSPK; s++) {
        int n = factory_nvec(s);
        for (int j = 0; j < n; j++)
            cand[nc++] = (typeof(cand[0])){cos_dist(mean39, factory_vec(s, j)), s};
    }
    for (int e = 0; e < SPK_EXTRA_MAX; e++) {
        if (!s_extra[e].used) continue;
        for (int j = 0; j < s_extra[e].nvec; j++)
            cand[nc++] = (typeof(cand[0])){cos_dist(mean39, s_extra[e].vecs[j]),
                                           ENROLL_DEFAULT_NSPK + e};
    }
    if (nc == 0) return -1;
    // 3 menores (k=3 do notebook).
    int k = nc < 3 ? nc : 3, top[3] = {-1, -1, -1};
    for (int i = 0; i < nc; i++) {
        for (int t = 0; t < k; t++) {
            if (top[t] < 0 || cand[i].d < cand[top[t]].d) {
                for (int u = k - 1; u > t; u--) top[u] = top[u - 1];
                top[t] = i;
                break;
            }
        }
    }
    float dmin = cand[top[0]].d;
    if (out_dist) *out_dist = dmin;
    if (dmin > s_tau) return -1; // DESCONHECIDO
    if (dmin < 1e-9f) return cand[top[0]].spk; // idêntico: voto direto
    // Voto ponderado 1/d por locutor.
    float best_w = -1;
    int best = cand[top[0]].spk;
    for (int t = 0; t < k; t++) {
        int s = cand[top[t]].spk;
        float w = 0;
        for (int u = 0; u < k; u++)
            if (cand[top[u]].spk == s) w += 1.0f / (cand[top[u]].d + 1e-9f);
        if (w > best_w) { best_w = w; best = s; }
    }
    return best;
}
