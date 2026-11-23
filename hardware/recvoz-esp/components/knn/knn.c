#include <math.h>
#include <stdlib.h>
#include "knn.h"

#define KNN_IDENTICAL_EPS 1e-9f // abaixo disso é o próprio vetor: voto direto

float knn_cos_dist(const float *a, const float *b, int dim, int skip_first)
{
    if (!a || !b || dim <= 0) return 1.0f;
    int start = (skip_first && dim > 1) ? 1 : 0;
    double d = 0, na = 0, nb = 0;
    for (int i = start; i < dim; i++) {
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

// Para cada rótulo distinto em `labels`, calcula a média das `min(k, count)`
// menores distâncias (Local Mean k-NN): elimina o viés de classes com mais
// vetores (ex. 14 PEDRO vs 9 ADRIAN vs 7 VISITANTE) e evita que 1 outlier vença.
static int score_speakers(const float *d, const int *labels, int nvec, int k,
                          int *dlab, float *ddist)
{
    int ndl = 0;
    for (int i = 0; i < nvec; i++) {
        int seen = 0;
        for (int j = 0; j < ndl; j++)
            if (dlab[j] == labels[i]) { seen = 1; break; }
        if (seen) continue;
        int s = labels[i];
        float top[KNN_K_MAX];
        int cnt = 0;
        for (int m = 0; m < nvec; m++) {
            if (labels[m] != s) continue;
            float dm = d[m];
            if (cnt < k) {
                top[cnt++] = dm;
                for (int u = cnt - 1; u > 0 && top[u] < top[u - 1]; u--) {
                    float t = top[u]; top[u] = top[u - 1]; top[u - 1] = t;
                }
            } else if (dm < top[k - 1]) {
                top[k - 1] = dm;
                for (int u = k - 1; u > 0 && top[u] < top[u - 1]; u--) {
                    float t = top[u]; top[u] = top[u - 1]; top[u - 1] = t;
                }
            }
        }
        double sum = 0.0;
        for (int u = 0; u < cnt; u++) sum += top[u];
        dlab[ndl] = s;
        ddist[ndl] = (float)(sum / cnt);
        ndl++;
    }
    return ndl;
}

// Checagem comum (classify e full falham igual nos mesmos args).
static int bad_args(const float *q, const float *v, const int *l,
                    int nvec, int dim, int skip_first, int k, float tau)
{
    if (!q || !v || !l || nvec < 1 || dim < 1 || k < 1) return 1;
    if (skip_first != 0 && skip_first != 1) return 1;
    if (!(tau > 0.0f) || !isfinite(tau)) return 1;
    for (int i = 0; i < nvec; i++)
        if (l[i] < 0) return 1; // rótulos colidiriam com o −1 de desconhecido
    return 0;
}

int knn_classify(const float *query,
                 const float *vecs, const int *labels, int nvec,
                 int dim, int skip_first, int k, float tau,
                 float *out_dmin)
{
    knn_result_t r;
    int w = knn_classify_full(query, vecs, labels, nvec, dim, skip_first, k, tau, &r);
    if (out_dmin) *out_dmin = r.dmin;
    return w;
}

int knn_classify_full(const float *query,
                      const float *vecs, const int *labels, int nvec,
                      int dim, int skip_first, int k, float tau,
                      knn_result_t *out)
{
    knn_result_t r = {-1, 1.0f, -1, 1.0f, -1};
    if (out) *out = r;
    if (bad_args(query, vecs, labels, nvec, dim, skip_first, k, tau)) return -1;
    if (k > nvec) k = nvec;
    if (k > KNN_K_MAX) k = KNN_K_MAX;
    float *d = (float *)malloc(nvec * sizeof(float));
    int *dlab = (int *)malloc(nvec * sizeof(int));
    float *ddist = (float *)malloc(nvec * sizeof(float));
    if (!d || !dlab || !ddist) { free(d); free(dlab); free(ddist); return -1; }
    for (int i = 0; i < nvec; i++)
        d[i] = knn_cos_dist(query, vecs + i * dim, dim, skip_first);
    int ndl = score_speakers(d, labels, nvec, k, dlab, ddist);
    float best1 = 1.0f, best2 = 1.0f;
    int i1 = -1, i2 = -1;
    for (int j = 0; j < ndl; j++) {
        if (ddist[j] < best1) { best2 = best1; i2 = i1; best1 = ddist[j]; i1 = dlab[j]; }
        else if (ddist[j] < best2) { best2 = ddist[j]; i2 = dlab[j]; }
    }
    if (i1 < 0) { free(d); free(dlab); free(ddist); return -1; }
    r.dmin = best1;
    r.winner = (best1 > tau) ? -1 : i1;
    r.nearest = i1;
    r.runner = i2;
    r.drunner = (i2 >= 0) ? best2 : 1.0f;
    free(d); free(dlab); free(ddist);
    if (out) *out = r;
    return r.winner;
}
