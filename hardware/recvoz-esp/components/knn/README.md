# knn — quem fala? (k-NN com cosseno, voto ponderado e limiar)

O "cérebro" de Sistemas Inteligentes do projeto, isolado em componente
próprio: **algoritmo puro** (sem flash, sem NVS, sem microfone). Quem guarda
os vetores é o `speaker_db`; aqui só mora a decisão. Tanto que o mesmo
`knn.c` compila no firmware **e** no host (`make -C tests/host/knn run`).

## O algoritmo em 5 passos

Dada uma consulta `q` (vetor médio 39-dim da janela de 2 s) e uma base de
vetores com rótulo de locutor:

1. **Distância a todo mundo** — cosseno: `d = 1 − a·b/(|a||b|)`, em [0, 2].
   Por que cosseno e não euclidiana? Cosseno mede **ângulo**, não tamanho:
   falar alto ou baixo (ganho do mic, distância da boca) muda o módulo do
   vetor, não a direção. A identidade (timbre) está na direção.
2. **k vizinhos** mais próximos — aqui `k=3`, igual ao `sklearn` do notebook
   (`n_neighbors=3, metric="cosine", weights="distance"`). `k` ímpar evita
   empate; `k` pequeno = fronteira de decisão detalhada (viés baixo).
3. **Voto ponderado por `1/d`** — soma os pesos por locutor, vence o maior.
   Um vizinho colado (d=0.001, peso 1000) vale mais que dois meia-boca
   (d=0.1, peso 10+10). O teste K3 do hosttest prova: 1 voto contra 2, e o
   1 ganha — voto majoritário puro erraria.
4. **Rejeição open-set (`tau`)** — se nem o melhor vizinho chega perto
   (`dmin > tau`), devolve −1 = DESCONHECIDO em vez de chutar o "menos pior".
   Classificador fechado sempre responde alguém; reconhecedor de verdade
   precisa saber dizer "não sei". Default `tau=0.017` (meio do gap
   0.0119–0.0219, leave-one-out 24/24 — ver `docs/CLASSIFICACAO.md` §3).
5. **Lazy (sem treino)** — matricular um locutor = armazenar seus vetores.
   O `enroll` do firmware não ajusta peso nenhum; na decisão seguinte os
   vetores novos já votam. É a beleza e o limite do k-NN: simples e
   incremental, mas cada decisão custa O(N·dim) — aqui N ≤ 48, dim = 39,
   trivial para o S3.

## Seleção de features: por que sem o c0?

O `speaker_db` chama com `skip_first=1`: o índice 0 (c0 = energia log média)
depende do ganho e da distância do mic, não de quem fala — ele causava
overlap mesma-voz/outra-voz e foi removido. Isso é **seleção de features**
clássica: jogar fora a dimensão que carrega ruído de aquisição, não sinal
de identidade. O k-NN não precisa saber disso — recebe `dim` e `skip_first`
como parâmetros (testável com e sem, ver K1b).

## Didática: top-2 + margem

`knn_classify_full()` devolve `winner/dmin/runner/drunner/nearest`:
**margem = drunner − dmin**. Margem grande = decisão folgada; barra
encostando no traço do τ = caso duvidoso (gancho perfeito p/ falar de limiar
open-set na banca). Se o k-NN discordar do vizinho mais próximo, vale o
k-NN — e o vice continua sendo o 2º por distância (honestidade didática).

## Uso

```c
#include "knn.h"

float dmin;
int label = knn_classify(q, vecs, labels, nvec,
                         39, 1, KNN_K_DEFAULT, 0.017f, &dmin);
// >=0 rótulo, -1 DESCONHECIDO
```

Rótulos são ints ≥ 0 ecoados de volta (o `speaker_db` passa a posição
mesclada — a mesma de `speaker_db_name()`).
