#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "gmm_ubm.h"
#include "gmm_default.h"

#define NF 126
#define DIM 51

int main(void)
{
    FILE *f = fopen("../../fixtures/expected_mfcc.bin", "rb");
    if (!f) { printf("FALTA ../../fixtures/expected_mfcc.bin\n"); return 2; }
    float *m = malloc((size_t)DIM * NF * sizeof(float));
    assert(m && fread(m, sizeof(float), DIM * NF, f) == DIM * NF);
    fclose(f);

    float llr_adrian = gmm_score_llr(m, NF, &GMM_SPK_MEANS[0][0][0]);
    float llr_pedro  = gmm_score_llr(m, NF, &GMM_SPK_MEANS[1][0][0]);
    printf("ok   G1 fixture ADRIAN llr=%.4f PEDRO llr=%.4f (margem=%.4f)\n",
           llr_adrian, llr_pedro, llr_adrian - llr_pedro);
    assert(llr_adrian > 1.0f && llr_adrian > llr_pedro + 0.5f);

    // G2: MAP adaptation on fixture frames produces higher LLR for adapted model than UBM
    gmm_adapt_ctx_t ctx;
    gmm_adapt_init(&ctx);
    gmm_adapt_add_window(&ctx, m, NF);
    float adapted[GMM_UBM_M * GMM_UBM_D];
    gmm_adapt_finish(&ctx, adapted);
    float llr_ad = gmm_score_llr(m, NF, adapted);
    printf("ok   G2 adapted llr=%.4f (>0.15)\n", llr_ad);
    assert(llr_ad > 0.15f);

    free(m);
    puts("PASS");
    return 0;
}
