// speaker_db — base de locutores: fábrica (flash) + extras (NVS).
//
// Guarda vetores (persistência); QUEM decide é o componente `knn`
// (k-NN k=3, cosseno, voto 1/d — ver components/knn/README.md).
// O speaker_db monta a base plana (fábrica ativa + extras, ordem mesclada)
// e passa rótulos = posições mescladas (as mesmas de count/name).
//
// Fábrica = enroll_default.h (ADRIAN + PEDRO, 36 vetores), sempre disponível.
// Extras (ex enroll "VISITANTE") vão p/ NVS "recvoz": no máx 3 locutores x
// 8 vetores (blob de 1249 B cada — cabe folgado no limite de ~2 KB/entrada).
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "esp_log.h"
#include "esp_check.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "speaker_db.h"
#include "knn.h"
#include "enroll_default.h"

#define SPK_K 3 // k do notebook (sklearn n_neighbors=3)

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
static uint8_t s_ovr; // bit s = fábrica s sobrescrita por enroll ao vivo
static int s_ok;

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
    // READWRITE (não READONLY): no 1º boot o namespace ainda não existe
    // e o READONLY falharia com NOT_FOUND, matando o app no ESP_ERROR_CHECK.
    ESP_RETURN_ON_ERROR(nvs_open(NS, NVS_READWRITE, &h), TAG, "nvs open");
    uint32_t bits = 0;
    size_t len = sizeof(bits);
    uint8_t tauv = 0;
    nvs_get_u8(h, "tauver", &tauv);
    if (tauv != 2) {
        // TAU_DEFAULT mudou (-> 0.017, base só-com-voz): ignora valor
        // antigo, grava o novo + carimba versão.
        s_tau = SPK_TAU_DEFAULT;
        memcpy(&bits, &s_tau, 4);
        nvs_set_u32(h, "tau", bits);
        nvs_set_u8(h, "tauver", 2);
        nvs_commit(h);
    } else if (nvs_get_u32(h, "tau", &bits) == ESP_OK) {
        float t;
        memcpy(&t, &bits, 4);
        if (isfinite(t) && t > 0 && t < 1) s_tau = t;
    }
    memset(s_extra, 0, sizeof(s_extra));
    s_ovr = 0;
    uint8_t ovr = 0;
    if (nvs_get_u8(h, "ovr", &ovr) == ESP_OK)
        s_ovr = ovr & ((1u << ENROLL_DEFAULT_NSPK) - 1);
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
    int n = 0;
    for (int s = 0; s < ENROLL_DEFAULT_NSPK; s++)
        if (!(s_ovr & (1u << s))) n++;
    for (int i = 0; i < SPK_EXTRA_MAX; i++)
        if (s_extra[i].used) n++;
    return n;
}

// Ordem mesclada: fábrica ativa + extras. Resolve pos -> fábrica s ou extra e.
static int merged_resolve(int pos, int *out_factory, int *out_extra)
{
    for (int s = 0; s < ENROLL_DEFAULT_NSPK; s++) {
        if (s_ovr & (1u << s)) continue;
        if (pos-- == 0) { *out_factory = s; *out_extra = -1; return 0; }
    }
    for (int e = 0; e < SPK_EXTRA_MAX; e++) {
        if (!s_extra[e].used) continue;
        if (pos-- == 0) { *out_factory = -1; *out_extra = e; return 0; }
    }
    return -1;
}

const char *speaker_db_name(int idx)
{
    int fs, fe;
    if (merged_resolve(idx, &fs, &fe) != 0) return "?";
    return fs >= 0 ? ENROLL_DEFAULT_NAMES[fs] : s_extra[fe].name;
}

// Menor distância do vetor aos vetores do locutor idx (p/ diag/calibração).
float speaker_db_dist_to(int idx, const float *mean39)
{
    if (!s_ok || !mean39) return 1.0f;
    int fs, fe;
    if (merged_resolve(idx, &fs, &fe) != 0) return 1.0f;
    float best = 1.0f;
    if (fs >= 0) {
        int n = factory_nvec(fs);
        for (int j = 0; j < n; j++) {
            float d = knn_cos_dist(mean39, factory_vec(fs, j), SPK_DIM, 1);
            if (d < best) best = d;
        }
    } else {
        for (int j = 0; j < s_extra[fe].nvec; j++) {
            float d = knn_cos_dist(mean39, s_extra[fe].vecs[j], SPK_DIM, 1);
            if (d < best) best = d;
        }
    }
    return best;
}

// Junta fábrica ativa + extras em arrays planos p/ o k-NN (ordem mesclada,
// rótulos = posições — as mesmas de count/name). Retorna nº de vetores
// (<=0 se vazia/erro); chamador libera *vecs e *labels. skip_first=1: sem
// c0 (energia log ~ ganho do mic, não identidade; causava overlap).
static int gather_flat(float **out_vecs, int **out_labels)
{
    int n = 0, fs, fe;
    for (int p = 0; ; p++) {
        if (merged_resolve(p, &fs, &fe) != 0) break;
        n += (fs >= 0) ? factory_nvec(fs) : s_extra[fe].nvec;
    }
    *out_vecs = NULL; *out_labels = NULL;
    if (n <= 0) return 0;
    float *v = (float *)malloc(n * SPK_DIM * sizeof(float));
    int *l = (int *)malloc(n * sizeof(int));
    if (!v || !l) { free(v); free(l); return -1; }
    int pos = 0, c = 0;
    for (int p = 0; ; p++) {
        if (merged_resolve(p, &fs, &fe) != 0) break;
        if (fs >= 0) {
            int nv = factory_nvec(fs);
            for (int j = 0; j < nv; j++, c++) {
                memcpy(v + c * SPK_DIM, factory_vec(fs, j), SPK_DIM * sizeof(float));
                l[c] = pos;
            }
        } else {
            for (int j = 0; j < s_extra[fe].nvec; j++, c++) {
                memcpy(v + c * SPK_DIM, s_extra[fe].vecs[j], SPK_DIM * sizeof(float));
                l[c] = pos;
            }
        }
        pos++;
    }
    *out_vecs = v; *out_labels = l;
    return n;
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

    // Nome de fábrica? Os vetores ao vivo (mesmo mic/sala) passam a valer;
    // os de fábrica desse locutor saem da votação (bit ovr persistido).
    for (int s = 0; s < ENROLL_DEFAULT_NSPK; s++)
        if (strcmp(clean, ENROLL_DEFAULT_NAMES[s]) == 0) {
            s_ovr |= (1u << s);
            ESP_LOGI(TAG, "%s: fábrica sobrescrita por enroll ao vivo", clean);
        }

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
    if (r == ESP_OK) r = nvs_set_u8(h, "ovr", s_ovr);
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
    nvs_erase_key(h, "ovr");
    esp_err_t r = nvs_commit(h);
    nvs_close(h);
    memset(s_extra, 0, sizeof(s_extra));
    s_ovr = 0;
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
    if (r == ESP_OK) r = nvs_set_u8(h, "tauver", 2);
    if (r == ESP_OK) r = nvs_commit(h);
    nvs_close(h);
    ESP_RETURN_ON_ERROR(r, TAG, "nvs write");
    s_tau = tau;
    return ESP_OK;
}

int speaker_db_classify_full(const float *mean39, spk_result_t *out)
{
    spk_result_t r = {-1, 1.0f, -1, 1.0f, -1};
    if (out) *out = r;
    if (!s_ok || !mean39) return -1;
    float *v = NULL;
    int *l = NULL;
    int n = gather_flat(&v, &l);
    if (n <= 0) { free(v); free(l); return -1; }
    knn_result_t kr;
    knn_classify_full(mean39, v, l, n, SPK_DIM, 1, SPK_K, s_tau, &kr);
    free(v); free(l);
    r.winner = kr.winner; r.dmin = kr.dmin; r.runner = kr.runner;
    r.drunner = kr.drunner; r.nearest = kr.nearest;
    if (out) *out = r;
    return r.winner;
}

int speaker_db_classify(const float *mean39, float *out_dist)
{
    if (!s_ok || !mean39) { if (out_dist) *out_dist = 1.0f; return -1; }
    float *v = NULL;
    int *l = NULL;
    int n = gather_flat(&v, &l);
    if (n <= 0) { free(v); free(l); if (out_dist) *out_dist = 1.0f; return -1; }
    int w = knn_classify(mean39, v, l, n, SPK_DIM, 1, SPK_K, s_tau, out_dist);
    free(v); free(l);
    return w;
}
