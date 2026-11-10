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

// k menores distâncias por inserção (idêntico ao n_neighbors sobre lista).
static void topk(const float *d, int n, int k, int *top)
{
    for (int t = 0; t < k; t++) top[t] = -1;
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < k; t++) {
            if (top[t] < 0 || d[i] < d[top[t]]) {
                for (int u = k - 1; u > t; u--) top[u] = top[u - 1];
                top[t] = i;
                break;
            }
        }
    }
}

// Voto ponderado por 1/d entre os top-k (soma por rótulo; vence o maior
// peso). Retorna o rótulo vencedor; *out_dmin = d[top0].
static int vote(const float *d, const int *labels, int k, const int *top, float *out_dmin)
{
    float dmin = d[top[0]];
    if (out_dmin) *out_dmin = dmin;
    if (dmin < KNN_IDENTICAL_EPS) return labels[top[0]]; // idêntico: direto
    float best_w = -1;
    int best = labels[top[0]];
    for (int t = 0; t < k; t++) {
        int s = labels[top[t]];
        float w = 0;
        for (int u = 0; u < k; u++)
            if (labels[top[u]] == s) w += 1.0f / (d[top[u]] + 1e-9f);
        if (w > best_w) { best_w = w; best = s; }
    }
    return best;
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
    if (bad_args(query, vecs, labels, nvec, dim, skip_first, k, tau)) {
        if (out_dmin) *out_dmin = 1.0f;
        return -1;
    }
    if (k > nvec) k = nvec;
    if (k > KNN_K_MAX) k = KNN_K_MAX;
    float *d = (float *)malloc(nvec * sizeof(float));
    if (!d) {
        if (out_dmin) *out_dmin = 1.0f;
        return -1;
    }
    for (int i = 0; i < nvec; i++)
        d[i] = knn_cos_dist(query, vecs + i * dim, dim, skip_first);
    int top[KNN_K_MAX]; // k <= KNN_K_MAX garantido acima (64 B de stack)
    topk(d, nvec, k, top);
    float dmin;
    int best = vote(d, labels, k, top, &dmin);
    free(d);
    if (out_dmin) *out_dmin = dmin;
    if (dmin > tau) return -1; // DESCONHECIDO (open-set, não chute)
    return best;
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
    int *dlab = (int *)malloc(nvec * sizeof(int));     // rótulos distintos
    float *ddist = (float *)malloc(nvec * sizeof(float)); // min por rótulo
    if (!d || !dlab || !ddist) { free(d); free(dlab); free(ddist); return -1; }
    for (int i = 0; i < nvec; i++)
        d[i] = knn_cos_dist(query, vecs + i * dim, dim, skip_first);
    int ndl = 0;
    for (int i = 0; i < nvec; i++) {
        int f = -1;
        for (int j = 0; j < ndl; j++)
            if (dlab[j] == labels[i]) { f = j; break; }
        if (f < 0) { dlab[ndl] = labels[i]; ddist[ndl] = d[i]; ndl++; }
        else if (d[i] < ddist[f]) ddist[f] = d[i];
    }
    // Top-2 rótulos por distância.
    float best1 = 1.0f, best2 = 1.0f;
    int i1 = -1, i2 = -1;
    for (int j = 0; j < ndl; j++) {
        if (ddist[j] < best1) { best2 = best1; i2 = i1; best1 = ddist[j]; i1 = dlab[j]; }
        else if (ddist[j] < best2) { best2 = ddist[j]; i2 = dlab[j]; }
    }
    if (i1 < 0) { free(d); free(dlab); free(ddist); return -1; } // inalcançável (nvec>=1)
    int top[KNN_K_MAX];
    topk(d, nvec, k, top);
    float dummy;
    int w = vote(d, labels, k, top, &dummy);
    int gated = (best1 > tau) ? -1 : w;
    float dw = best1;
    if (gated >= 0)
        for (int j = 0; j < ndl; j++)
            if (dlab[j] == w) { dw = ddist[j]; break; }
    r.dmin = dw;
    r.winner = gated;
    r.nearest = i1;
    r.runner = (gated < 0 || i1 == gated) ? i2 : i1;
    if (r.runner >= 0)
        for (int j = 0; j < ndl; j++)
            if (dlab[j] == r.runner) { r.drunner = ddist[j]; break; }
    free(d); free(dlab); free(ddist);
    if (out) *out = r;
    return r.winner;
}
