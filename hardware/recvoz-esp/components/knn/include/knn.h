#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** k-NN didático e portátil (só libc + libm, sem ESP-IDF).
 *
 *  O algoritmo que decide "quem fala": dados vetores cadastrados com rótulo
 *  de locutor, classifica uma consulta por voto dos k vizinhos mais próximos.
 *  Teoria completa em `components/knn/README.md`; resumo operacional:
 *
 *  1. **Distância cosseno** entre consulta e cada vetor: `d = 1 − cos(a,b)`.
 *     Cosseno mede ÂNGULO, não tamanho — falar alto/baixo (ganho) não muda
 *     a direção do vetor, só o módulo. Por isso voz usa cosseno, não euclidiana.
 *  2. **k vizinhos** mais próximos (este projeto: k=3, como o sklearn do
 *     notebook). k ímpar evita empate; k pequeno = fronteira detalhada.
 *  3. **Voto ponderado por 1/d**: vizinho perto vale mais que vizinho longe.
 *     (Voto majoritário puro perderia essa nuance — ver teste K3.)
 *  4. **Rejeição open-set**: se nem o vizinho mais próximo chega perto
 *     (`dmin > tau`), responde DESCONHECIDO (−1) em vez de chutar alguém.
 *     É o que separa "classificação fechada" de "reconhecimento de verdade".
 *  5. **Lazy**: não há treino — matricular = armazenar vetores. O `enroll`
 *     do firmware só guarda médias 39-dim; na próxima decisão elas já votam.
 *
 */
#define KNN_K_DEFAULT 3   ///
#define KNN_K_MAX 16      ///< teto do k (protege a stack; N aqui é pequeno)

/** Resultado completo p/ didática (top-2 + margem no display). */
typedef struct {
    int winner;      ///< rótulo vencedor (ou −1 = desconhecido)
    float dmin;      ///< distância ao vencedor (ou ao nearest, se −1)
    int runner;      ///< vice por distância (−1 se só há 1 rótulo)
    float drunner;   ///< distância ao vice
    int nearest;     ///< mais próximo por distância (mesmo se desconhecido)
} knn_result_t;

/**
 * @brief Distância cosseno 1 − cos(a,b), em [0, 2] (0 = idênticos).
 * @param dim        dimensão dos vetores (>= 1).
 * @param skip_first 1 = ignora o índice 0 (seleção de features: o speaker_db
 *        usa 1 p/ descartar o c0 — energia log, que depende do ganho/mic e
 *        não da identidade; ver CLASSIFICACAO.md §3).
 * @return 1.0 se algum vetor é nulo/inválido (ortogonal por convenção).
 */
float knn_cos_dist(const float *a, const float *b, int dim, int skip_first);

/**
 * @brief Classifica por k-NN com rejeição por limiar.
 * @param query  vetor consulta (ex vetor médio 39-dim da janela de 2 s).
 * @param vecs   base plana [nvec][dim], contígua.
 * @param labels rótulo de cada vetor (ints >= 0, ecoados de volta; o
 *        speaker_db passa a posição mesclada, a mesma de speaker_db_name).
 * @param nvec   nº de vetores (>= 1). @param k vizinhos (>= 1; corta em nvec).
 * @param tau    limiar de desconhecido (> 0): dmin > tau → −1.
 * @param out_dmin recebe a menor distância (pode ser NULL).
 * @return rótulo vencedor, ou −1 = DESCONHECIDO (ou args inválidos).
 */
int knn_classify(const float *query,
                 const float *vecs, const int *labels, int nvec,
                 int dim, int skip_first, int k, float tau,
                 float *out_dmin);

/**
 * @brief Classifica e preenche top-1/top-2 (margem = drunner − dmin).
 * Mesmos parâmetros do classify + @param out (pode ser NULL).
 * @return idem (winner, −1 se desconhecido).
 */
int knn_classify_full(const float *query,
                      const float *vecs, const int *labels, int nvec,
                      int dim, int skip_first, int k, float tau,
                      knn_result_t *out);

#ifdef __cplusplus
}
#endif
