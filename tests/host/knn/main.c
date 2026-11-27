// Teste unitário do k-NN: vetores sintéticos 2-4 dim, determinístico.
// Cobre distância, voto ponderado vs majoritário, tau, top-2 e args.
// Uso: make run
#include <math.h>
#include <stdio.h>
#include <stdint.h> // enroll_default.h (gerado) usa uint8_t sem incluir
#include <string.h>
#include "knn.h"
#include "enroll_default.h"

static int fails;

static void check(int ok, const char *name, const char *detail)
{
    printf("%s %s %s\n", ok ? "ok  " : "FAIL", name, detail ? detail : "");
    if (!ok) fails++;
}

// Base 2 locutores em 3D: A perto do eixo x, B perto do eixo y.
static const float BASE[] = {
    1.0f, 0.0f, 0.0f,   // A
    0.9f, 0.1f, 0.0f,   // A
    0.95f, -0.05f, 0.1f, // A
    0.0f, 1.0f, 0.0f,   // B
    0.1f, 0.9f, 0.0f,   // B
    0.0f, 0.95f, 0.05f, // B
};
static const int LBL[] = {0, 0, 0, 1, 1, 1};

int main(void)
{
    // ---- K1: distância cosseno ----
    {
        float a[4] = {1, 0, 0, 0}, b[4] = {0, 1, 0, 0}, m[4] = {-1, 0, 0, 0};
        float z[4] = {0, 0, 0, 0};
        int ok = fabsf(knn_cos_dist(a, a, 4, 0) - 0.0f) < 1e-6f
              && fabsf(knn_cos_dist(a, b, 4, 0) - 1.0f) < 1e-6f
              && fabsf(knn_cos_dist(a, m, 4, 0) - 2.0f) < 1e-6f
              && fabsf(knn_cos_dist(z, a, 4, 0) - 1.0f) < 1e-6f;
        check(ok, "K1", "0/1/2/zero->1");
    }

    // ---- K1b: skip_first (seleção de features, como o c0) ----
    {
        float a[4] = {100, 1, 0, 0}, b[4] = {200, 1, 0, 0};
        float full = knn_cos_dist(a, b, 4, 0);
        float skip = knn_cos_dist(a, b, 4, 1);
        char d[64];
        snprintf(d, sizeof d, "full=%.3g skip=%.3g", full, skip);
        check(full > 1e-6f && skip == 0.0f, "K1b", d);
    }

    // ---- K2: classifica o óbvio ----
    {
        float q[3] = {0.98f, 0.02f, 0.0f}, dmin = -1;
        int w = knn_classify(q, BASE, LBL, 6, 3, 0, 3, 1.0f, &dmin);
        char d[64];
        snprintf(d, sizeof d, "w=%d dmin=%.5f", w, dmin);
        check(w == 0 && dmin < 0.01f, "K2", d);
    }

    // ---- K3: ponderado 1/d vence majoritário ----
    // 1 vizinho A colado (peso ~2e6) vs 2 vizinhos B a d=0.05 (peso 20+20):
    // maioria diria B; o ponderado diz A.
    {
        float v[3 * 2] = {1.0f, 0.001f, 0.975f, 0.222f, 0.980f, 0.199f};
        int l[3] = {0, 1, 1};
        float q[2] = {1.0f, 0.0f}, dmin = -1;
        int w = knn_classify(q, v, l, 3, 2, 0, 3, 1.0f, &dmin);
        char d[64];
        snprintf(d, sizeof d, "w=%d dmin=%.3g (1 voto A vs 2 votos B)", w, dmin);
        check(w == 0, "K3", d);
    }

    // ---- K4: rejeição por tau (open-set) ----
    {
        float q[3] = {0.0f, 0.0f, 1.0f}, dmin = -1; // ortogonal a tudo
        int w = knn_classify(q, BASE, LBL, 6, 3, 0, 3, 0.017f, &dmin);
        char d[64];
        snprintf(d, sizeof d, "w=%d dmin=%.3f (tau=0.017)", w, dmin);
        check(w == -1 && dmin > 0.017f, "K4", d);
    }

    // ---- K5: full expõe top-2 + margem ----
    {
        float q[3] = {0.98f, 0.02f, 0.0f};
        knn_result_t r;
        int w = knn_classify_full(q, BASE, LBL, 6, 3, 0, 3, 1.0f, &r);
        char d[96];
        snprintf(d, sizeof d, "w=%d near=%d run=%d margem=%.4f",
                 r.winner, r.nearest, r.runner, r.drunner - r.dmin);
        check(w == 0 && r.nearest == 0 && r.runner == 1
              && (r.drunner - r.dmin) > 0, "K5", d);
    }

    // ---- K5b: full desconhecido mostra o "quase foi" ----
    {
        float q[3] = {0.0f, 0.0f, 1.0f};
        knn_result_t r;
        int w = knn_classify_full(q, BASE, LBL, 6, 3, 0, 3, 0.017f, &r);
        check(w == -1 && r.winner == -1 && r.nearest >= 0, "K5b", "winner=-1, nearest>=0");
    }

    // ---- K6: args inválidos falham fechado ----
    {
        float q[3] = {1, 0, 0};
        int ok = knn_classify(NULL, BASE, LBL, 6, 3, 0, 3, 1.0f, NULL) == -1
              && knn_classify(q, BASE, LBL, 0, 3, 0, 3, 1.0f, NULL) == -1
              && knn_classify(q, BASE, LBL, 6, 3, 0, 0, 1.0f, NULL) == -1
              && knn_classify(q, BASE, LBL, 6, 3, 0, 3, -1.0f, NULL) == -1;
        check(ok, "K6", "NULL/nvec=0/k=0/tau<0 -> -1");
    }

    // ---- K7: leave-one-out nos 24 vetores de fábrica (dados reais) ----
    // Espelha CLASSIFICACAO.md §3 (sklearn: 24/24, gap 0.0119–0.0219).
    // Prova que a extração p/ o componente knn preservou o comportamento.
    {
        (void)ENROLL_DEFAULT_NAMES; // só p/ -Wall não reclamar do gerado
        int correct = 0;
        float max_same = 0, min_other = 1;
        float base[(ENROLL_DEFAULT_NVEC - 1) * 51];
        int lb[ENROLL_DEFAULT_NVEC - 1];
        for (int i = 0; i < ENROLL_DEFAULT_NVEC; i++) {
            int c = 0;
            for (int j = 0; j < ENROLL_DEFAULT_NVEC; j++) {
                if (j == i) continue;
                memcpy(base + c * 51, ENROLL_DEFAULT_VECS[j], 51 * sizeof(float));
                lb[c] = ENROLL_DEFAULT_LABELS[j];
                c++;
            }
            float dmin = -1;
            int w = knn_classify(ENROLL_DEFAULT_VECS[i], base, lb,
                                 ENROLL_DEFAULT_NVEC - 1, 51, 1, 3, 0.065f, &dmin);
            if (w == ENROLL_DEFAULT_LABELS[i]) correct++;
            float ds = 1, doo = 1;
            for (int j = 0; j < ENROLL_DEFAULT_NVEC; j++) {
                if (j == i) continue;
                float d = knn_cos_dist(ENROLL_DEFAULT_VECS[i], ENROLL_DEFAULT_VECS[j], 51, 1);
                if (ENROLL_DEFAULT_LABELS[j] == ENROLL_DEFAULT_LABELS[i]) { if (d < ds) ds = d; }
                else { if (d < doo) doo = d; }
            }
            if (ds > max_same) max_same = ds;
            if (doo < min_other) min_other = doo;
        }
        char d[96];
        snprintf(d, sizeof d, "%d/%d mesma-voz<=%.4f outra-voz>=%.4f",
                 correct, ENROLL_DEFAULT_NVEC, max_same, min_other);
        check(correct == ENROLL_DEFAULT_NVEC && max_same < 0.02f, "K7", d);
    }

    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
