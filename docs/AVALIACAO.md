# Avaliação e resultados

## Protocolo atual

1. Cada WAV é uma sessão. VAD e segmentação são aplicados separadamente nela.
2. MFCC é calculado em cada janela de 32.000 amostras, produzindo 126 frames.
3. Na identificação fechada, uma sessão de cada pessoa fica no teste; as demais
   ficam no treino. A seed determina a sessão reservada.
4. No open-set, outra sessão de cada conhecido fica na calibração. São
   necessárias pelo menos três sessões por conhecido.
5. O limiar minimiza a média de falsa rejeição e falsa aceitação na calibração.
   O limiar fica no intervalo aceito pelo firmware, estritamente entre 0 e 1.
   Em empate, escolhe o menor limiar. O conjunto de teste não participa.
6. Desconhecidos de calibração e de teste são pessoas distintas entre si e dos
   conhecidos. O relatório traz FAR, FRR e acurácia de identificação conhecida.

As taxas são calculadas por janela. Com poucas sessões elas têm alta incerteza;
mais janelas da mesma gravação não substituem mais sessões ou mais pessoas.
O notebook compara cinco métodos em splits por sessão; repetições podem usar
os mesmos splits quando a base tem poucas sessões.

## Resultado local — seed 42 (base anonimizada, 6 locutores / 113 janelas)

Veja [session-holdout.json](results/session-holdout.json), [open-set.json](results/open-set.json) e [benchmark-openset.json](results/benchmark-openset.json):

- **Conhecidos (`audio_exemplo/`):** `S1` (3 sessões, 27 janelas) e `S2` (4 sessões, 14 janelas).
- **Impostores de calibração (`dados/desconhecidos_cal/`):** `U1` e `U2` (6 sessões, 38 janelas).
- **Impostores de teste (`dados/desconhecidos_teste/`):** `U3` e `U4` (6 sessões, 34 janelas).
- **Identificação fechada (seed 42):** **14/14 (100%)** em `S1/s3` e `S2/s3`.
- **Open-set (seed 42, 5 modelos via `tools/benchmark_openset.py`):**
  - Centroide Cosseno: Acc 100%, FAR 2,9%, FRR 21,4%, **EER 5,0%** (10 splits: `7,3 ± 7,5%`).
  - $k$-NN Cosseno ($k=3$): Acc 100%, FAR 8,8%, FRR 7,1%, **EER 5,0%** (10 splits: `11,7 ± 9,1%`).
  - SVM (RBF): Acc 100%, FAR 5,9%, FRR 35,7%, **EER 29,0%** (10 splits: `22,3 ± 9,7%`).
  - GMM Indep. + CMN: Acc 100%, FAR 0,0%, FRR 35,7%, **EER 48,7%** (10 splits: `31,3 ± 10,4%`).
  - **GMM-UBM + CMN ($M=8$, NS $\alpha=0,8$):** Acc 100%, FAR 29,4%, FRR 7,1%, **EER 21,0%** (10 splits: `30,2 ± 8,4%`).
- **Ablação GMM-UBM:** sem CMN o EER sobe para **64,5%** (FAR 97,1%); com CMVN vai a **38,4%**; sem NS vai a **34,0%** (FRR 21,4%).

O EER mais alto do GMM-UBM + CMN frente ao $k$-NN no mesmo microfone reflete o
UBM estimado com apenas 2 locutores de treino (`S1`, `S2`). Treinar o UBM com
um conjunto mais amplo de falantes externos continua sendo o próximo passo de
maior impacto.

## Resultados históricos

O antigo 93,3% ± 4,8% usava splits aleatórios de segmentos, podendo compartilhar
uma gravação entre treino e teste. Não é comparável diretamente ao protocolo
atual. O LOO 24/24 verifica a separabilidade dos vetores da base de fábrica
histórica; também não mede generalização para novas sessões ou desconhecidos.

Os notebooks foram salvos sem saídas antigas para evitar apresentar métricas
de um protocolo anterior como se fossem resultados do código atual.
