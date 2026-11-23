#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "esp_log.h"
#include "esp_check.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "speaker_db.h"
#include "knn.h"
#include "gmm_ubm.h"
#include "enroll_default.h"
#include "gmm_default.h"

#define SPK_K 3 // k do notebook (sklearn n_neighbors=3)

static const char *TAG = "speaker_db";
#define NS "recvoz"

typedef struct {
    char name[SPK_NAME_LEN];
    uint8_t nvec;
    float vecs[SPK_EXTRA_VECS][SPK_DIM];
    float gmm_means[GMM_UBM_M * GMM_UBM_D];
    uint8_t has_gmm;
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
    uint8_t tauv = 0;
    nvs_get_u8(h, "tauver", &tauv);
    if (tauv != 3) {
        // TAU_DEFAULT mudou (-> 0.42, GMM-UBM + CMN): ignora valor
        // antigo, grava o novo + carimba versão.
        s_tau = SPK_TAU_DEFAULT;
        memcpy(&bits, &s_tau, 4);
        esp_err_t r = nvs_set_u32(h, "tau", bits);
        if (r == ESP_OK) r = nvs_set_u8(h, "tauver", 3);
        if (r == ESP_OK) r = nvs_commit(h);
        if (r != ESP_OK) { nvs_close(h); return r; }
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
        if (bl < 1) continue;
        uint8_t n = blob[0];
        if (n == 0 || n > SPK_EXTRA_VECS || bl < 1 + n * SPK_DIM * sizeof(float)) {
            s_extra[i].name[0] = 0;
            continue;
        }
        s_extra[i].nvec = n;
        memcpy(s_extra[i].vecs, blob + 1, n * SPK_DIM * sizeof(float));
        char kgmm[8];
        snprintf(kgmm, sizeof kgmm, "x%dg", i);
        size_t gl = sizeof(s_extra[i].gmm_means);
        if (nvs_get_blob(h, kgmm, s_extra[i].gmm_means, &gl) == ESP_OK && gl == sizeof(s_extra[i].gmm_means)) {
            s_extra[i].has_gmm = 1;
        } else {
            gmm_adapt_ctx_t ctx;
            gmm_adapt_init(&ctx);
            gmm_adapt_finish(&ctx, s_extra[i].gmm_means);
            s_extra[i].has_gmm = 0;
        }
        s_extra[i].used = 1;
        ESP_LOGI(TAG, "extra[%d] %s (%d vetores, gmm=%d)", i, s_extra[i].name, n, s_extra[i].has_gmm);
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

esp_err_t speaker_db_enroll_gmm(const char *name, const float *vecs, int nvec, const float *gmm_means_Mx24)
{
    ESP_RETURN_ON_FALSE(s_ok && name && name[0] && vecs, ESP_ERR_INVALID_ARG, TAG, "arg");
    ESP_RETURN_ON_FALSE(nvec >= 1 && nvec <= SPK_EXTRA_VECS, ESP_ERR_INVALID_ARG, TAG, "nvec=%d", nvec);
    for (int i = 0; i < nvec * SPK_DIM; i++)
        ESP_RETURN_ON_FALSE(isfinite(vecs[i]), ESP_ERR_INVALID_ARG, TAG, "non-finite vector");
    if (gmm_means_Mx24) {
        for (int i = 0; i < GMM_UBM_M * GMM_UBM_D; i++)
            ESP_RETURN_ON_FALSE(isfinite(gmm_means_Mx24[i]), ESP_ERR_INVALID_ARG, TAG, "non-finite gmm");
    }
    char clean[SPK_NAME_LEN] = {0};
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

    uint8_t next_ovr = s_ovr;
    for (int s = 0; s < ENROLL_DEFAULT_NSPK; s++)
        if (strcmp(clean, ENROLL_DEFAULT_NAMES[s]) == 0) {
            next_ovr |= (1u << s);
            ESP_LOGI(TAG, "%s: fábrica sobrescrita por enroll ao vivo", clean);
        }

    nvs_handle_t h;
    ESP_RETURN_ON_ERROR(nvs_open(NS, NVS_READWRITE, &h), TAG, "nvs open");
    char kname[8], kvec[8], kgmm[8];
    snprintf(kname, sizeof kname, "x%dn", slot);
    snprintf(kvec, sizeof kvec, "x%dv", slot);
    snprintf(kgmm, sizeof kgmm, "x%dg", slot);
    uint8_t blob[1 + SPK_EXTRA_VECS * SPK_DIM * sizeof(float)];
    blob[0] = nvec;
    memcpy(blob + 1, vecs, nvec * SPK_DIM * sizeof(float));
    esp_err_t r = nvs_set_str(h, kname, clean);
    if (r == ESP_OK) r = nvs_set_blob(h, kvec, blob, 1 + nvec * SPK_DIM * sizeof(float));
    if (r == ESP_OK && gmm_means_Mx24)
        r = nvs_set_blob(h, kgmm, gmm_means_Mx24, GMM_UBM_M * GMM_UBM_D * sizeof(float));
    if (r == ESP_OK) r = nvs_set_u8(h, "ovr", next_ovr);
    if (r == ESP_OK) r = nvs_commit(h);
    nvs_close(h);
    ESP_RETURN_ON_ERROR(r, TAG, "nvs write");
    memcpy(s_extra[slot].name, clean, SPK_NAME_LEN);
    s_extra[slot].nvec = nvec;
    memcpy(s_extra[slot].vecs, vecs, nvec * SPK_DIM * sizeof(float));
    if (gmm_means_Mx24) {
        memcpy(s_extra[slot].gmm_means, gmm_means_Mx24, sizeof(s_extra[slot].gmm_means));
        s_extra[slot].has_gmm = 1;
    } else {
        s_extra[slot].has_gmm = 0;
    }
    s_extra[slot].used = 1;
    s_ovr = next_ovr;
    ESP_LOGI(TAG, "cadastrado %s no slot %d (%d vetores, gmm=%d)", clean, slot, nvec, s_extra[slot].has_gmm);
    return ESP_OK;
}

esp_err_t speaker_db_enroll(const char *name, const float *vecs, int nvec)
{
    return speaker_db_enroll_gmm(name, vecs, nvec, NULL);
}

esp_err_t speaker_db_reset(void)
{
    ESP_RETURN_ON_FALSE(s_ok, ESP_ERR_INVALID_STATE, TAG, "init primeiro");
    nvs_handle_t h;
    ESP_RETURN_ON_ERROR(nvs_open(NS, NVS_READWRITE, &h), TAG, "nvs open");
    esp_err_t r = ESP_OK;
    for (int i = 0; i < SPK_EXTRA_MAX && r == ESP_OK; i++) {
        char key[8];
        const char suf[3] = {'n', 'v', 'g'};
        for (int field = 0; field < 3 && r == ESP_OK; field++) {
            snprintf(key, sizeof key, "x%d%c", i, suf[field]);
            r = nvs_erase_key(h, key);
            if (r == ESP_ERR_NVS_NOT_FOUND) r = ESP_OK;
        }
    }
    if (r == ESP_OK) {
        r = nvs_erase_key(h, "ovr");
        if (r == ESP_ERR_NVS_NOT_FOUND) r = ESP_OK;
    }
    if (r == ESP_OK) r = nvs_commit(h);
    nvs_close(h);
    ESP_RETURN_ON_ERROR(r, TAG, "nvs commit");
    memset(s_extra, 0, sizeof(s_extra));
    s_ovr = 0;
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
    if (r == ESP_OK) r = nvs_set_u8(h, "tauver", 3);
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

static float llr_to_dist(float llr)
{
    float d = 0.5f * (1.0f - llr);
    if (d < 0.0f) d = 0.0f;
    if (d > 1.0f) d = 1.0f;
    return d;
}

float speaker_db_llr_to(int idx, const float *m39xT, int n_frames)
{
    if (!s_ok || !m39xT || n_frames <= 0) return -10.0f;
    int fs, fe;
    if (merged_resolve(idx, &fs, &fe) != 0) return -10.0f;
    if (fs >= 0) return gmm_score_llr(m39xT, n_frames, &GMM_SPK_MEANS[fs][0][0]);
    if (s_extra[fe].has_gmm) return gmm_score_llr(m39xT, n_frames, s_extra[fe].gmm_means);
    return -10.0f;
}

int speaker_db_classify_gmm(const float *m39xT, int n_frames, spk_result_t *out,
                            float *out_llr1, float *out_llr2)
{
    spk_result_t r = {-1, 1.0f, -1, 1.0f, -1};
    if (out) *out = r;
    if (out_llr1) *out_llr1 = -10.0f;
    if (out_llr2) *out_llr2 = -10.0f;
    if (!s_ok || !m39xT || n_frames <= 0) return -1;

    float *cmn = (float *)malloc((size_t)n_frames * GMM_UBM_D * sizeof(float));
    if (!cmn) return -1;
    gmm_cmn24_window(m39xT, n_frames, cmn);
    float ubm_ll = gmm_logpdf_cmn(cmn, n_frames, NULL);

    int nspk = speaker_db_count();
    float best1 = -1e30f, best2 = -1e30f;
    int i1 = -1, i2 = -1;
    for (int idx = 0; idx < nspk; idx++) {
        int fs, fe;
        if (merged_resolve(idx, &fs, &fe) != 0) continue;
        const float *means = NULL;
        if (fs >= 0) means = &GMM_SPK_MEANS[fs][0][0];
        else if (s_extra[fe].has_gmm) means = s_extra[fe].gmm_means;
        if (!means) continue;
        float llr = gmm_score_llr_cmn(cmn, n_frames, means, ubm_ll);
        if (llr > best1) {
            best2 = best1; i2 = i1;
            best1 = llr; i1 = idx;
        } else if (llr > best2) {
            best2 = llr; i2 = idx;
        }
    }
    free(cmn);
    if (i1 < 0) return -1;
    r.dmin = llr_to_dist(best1);
    r.drunner = (i2 >= 0) ? llr_to_dist(best2) : 1.0f;
    r.nearest = i1;
    r.runner = i2;
    r.winner = (r.dmin > s_tau) ? -1 : i1;
    if (out) *out = r;
    if (out_llr1) *out_llr1 = best1;
    if (out_llr2) *out_llr2 = best2;
    return r.winner;
}
