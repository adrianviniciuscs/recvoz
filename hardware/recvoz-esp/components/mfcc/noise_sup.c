#include <math.h>
#include <string.h>
#include "noise_sup.h"

#define NS_ANCHOR_DIM 39

static float s_N[NS_BINS];            // perfil médio do ruído (~2 KB)
static float s_anchor[NS_ANCHOR_DIM]; // âncora MFCC do ruído ambiente
static int s_has_anchor;
static int s_noise_rms;               // RMS int16 na banda de voz do silêncio calibrado
static int s_ready;                   // 1 após a 1ª calibração
static int s_enabled;                 // padrão: desligado (paridade bit a bit)
static float s_alpha = NS_ALPHA_DEFAULT;
static float s_floor = NS_FLOOR_DEFAULT;
static float s_beta = NS_BETA_DEFAULT;

void ns_reset(void)
{
    memset(s_N, 0, sizeof s_N);
    memset(s_anchor, 0, sizeof s_anchor);
    s_has_anchor = 0;
    s_noise_rms = 0;
    s_ready = 0;
}

int ns_is_ready(void) { return s_ready; }

void ns_set_enabled(int en) { s_enabled = en ? 1 : 0; }
int ns_is_enabled(void) { return s_enabled; }

int ns_set_params(float alpha, float floor_ratio)
{
    if (!(alpha > 0.0f && alpha <= 4.0f)) return -1;
    if (!(floor_ratio >= 0.0f && floor_ratio <= 0.5f)) return -1;
    s_alpha = alpha;
    s_floor = floor_ratio;
    return 0;
}

void ns_get_params(float *alpha, float *floor_ratio)
{
    if (alpha) *alpha = s_alpha;
    if (floor_ratio) *floor_ratio = s_floor;
}

void ns_set_beta(float beta)
{
    if (beta > 0.0f && beta <= 1.0f) s_beta = beta;
}
float ns_get_beta(void) { return s_beta; }

void ns_update_avg(const float *avg_pow513)
{
    if (!avg_pow513) return;
    if (!s_ready) {
        memcpy(s_N, avg_pow513, sizeof s_N);
        s_ready = 1;
        return;
    }
    float b = s_beta;
    for (int k = 0; k < NS_BINS; k++)
        s_N[k] += b * (avg_pow513[k] - s_N[k]);
    // Suavização 3-pt em frequência (in-place, p/ frente: k+1 ainda intacto).
    // Mata picos isolados de 1 bin — principal fonte de "ruído musical".
    float prev = s_N[0];
    for (int k = 1; k < NS_BINS - 1; k++) {
        float cur = s_N[k];
        s_N[k] = 0.25f * prev + 0.5f * cur + 0.25f * s_N[k + 1];
        prev = cur;
    }
}

void ns_apply(float *pow513)
{
    if (!pow513 || !s_enabled || !s_ready) return;
    float a = s_alpha, fl = s_floor;
    for (int k = 0; k < NS_BINS; k++) {
        float p = pow513[k];
        float sub = p - a * s_N[k];
        float mn = fl * p;
        pow513[k] = sub > mn ? sub : mn;
    }
}

float ns_snr_db(const float *pow513)
{
    if (!pow513 || !s_ready) return -99.0f;
    double sp = 0.0, sn = 0.0;
    for (int k = 1; k < NS_BINS; k++) { // pula o bin DC
        sp += pow513[k];
        sn += s_N[k];
    }
    if (sn <= 0.0) return 99.0f;
    if (sp <= 0.0) return -99.0f;
    return (float)(10.0 * log10(sp / sn));
}

void ns_set_noise_anchor(const float *vec39, int rms16)
{
    if (!vec39 || rms16 < 0) return;
    if (!s_has_anchor) {
        memcpy(s_anchor, vec39, sizeof s_anchor);
        s_noise_rms = rms16;
        s_has_anchor = 1;
        return;
    }
    float b = s_beta;
    for (int d = 0; d < NS_ANCHOR_DIM; d++)
        s_anchor[d] += b * (vec39[d] - s_anchor[d]);
    s_noise_rms = (int)((1.0f - b) * (float)s_noise_rms + b * (float)rms16 + 0.5f);
}

const float *ns_get_profile(void)
{
    return s_ready ? s_N : NULL;
}

const float *ns_get_noise_anchor(void)
{
    return (s_ready && s_has_anchor) ? s_anchor : NULL;
}

int ns_get_noise_rms(void)
{
    return s_ready ? s_noise_rms : 0;
}

float ns_noise_dist(const float *mean39)
{
    if (!mean39 || !s_enabled || !s_ready || !s_has_anchor) return 1.0f;
    double d = 0.0, na = 0.0, nb = 0.0;
    for (int i = 1; i < NS_ANCHOR_DIM; i++) {
        d += (double)mean39[i] * s_anchor[i];
        na += (double)mean39[i] * mean39[i];
        nb += (double)s_anchor[i] * s_anchor[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 1.0f;
    float c = (float)(d / (sqrt(na) * sqrt(nb)));
    if (c > 1.0f) c = 1.0f;
    if (c < -1.0f) c = -1.0f;
    return 1.0f - c;
}
