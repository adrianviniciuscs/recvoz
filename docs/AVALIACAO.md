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

## Resultado local — seed 42 (base anonimizada limpa, 6 locutores / 102 janelas)

Veja [session-holdout.json](results/session-holdout.json), [open-set.json](results/open-set.json) e [benchmark-openset.json](results/benchmark-openset.json):

- **Conhecidos (`audio_exemplo/`):** `S1` (3 sessões limpas, 16 janelas) e `S2` (4 sessões, 14 janelas).
- **Impostores de calibração (`dados/desconhecidos_cal/`):** `U1` e `U2` (6 sessões, 38 janelas).
- **Impostores de teste (`dados/desconhecidos_teste/`):** `U3` e `U4` (6 sessões, 34 janelas).
- **Identificação fechada (seed 42):** **10/10 (100%)** em `S1/s3` e `S2/s3`.
- **Open-set (10 splits únicos de sessão via `tools/benchmark_openset.py`, com filtro de Wiener casado à voz, banco Mel HTK 40, 17 MFCCs / $D=32$ e Frame-VAD 70%):**
  - **GMM-UBM (UBM externo FLEURS PT-BR, CMN-$c_1$):** Acc **100,0%**, **AUC: `95,5 ± 2,7%`** *(maior AUC)*, **FAR: `6,8 ± 5,3%`**, **EER: `9,4 ± 6,0%`** *(menor EER)*
  - **GMM-UBM (UBM externo FLEURS PT-BR, sem CMN):** Acc **100,0%**, AUC `93,1 ± 3,3%`, FAR `11,8%`, EER `12,4 ± 7,5%`
  - Centroide Cosseno: Acc 98,9%, AUC `81,7 ± 7,4%`, EER `24,6 ± 9,4%`
  - $k$-NN Cosseno ($k=3$): Acc 97,8%, AUC `79,2 ± 11,3%`, EER `31,8 ± 10,4%`
  - SVM (RBF): Acc 76,4%, AUC `69,1 ± 14,6%`, EER `35,6 ± 14,7%`
  - GMM Indep. + CMN: Acc 96,8%, AUC `66,2 ± 8,8%`, EER `40,7 ± 12,8%`
  - GMM-UBM + CMN (UBM interno, 2 loc.): Acc 96,7%, AUC `47,1 ± 14,2%`, EER `52,7 ± 11,3%`

Com o UBM externo treinado em 350 janelas (44.100 frames) do corpus público
*Google FLEURS PT-BR* e CMN seletiva em $c_1$, o GMM-UBM atinge o menor EER
médio entre todos os classificadores avaliados mantendo exatamente os mesmos
`1,66 KiB` em Flash e `768 B`/locutor na NVS.

## Resultados históricos

O antigo 93,3% ± 4,8% usava splits aleatórios de segmentos, podendo compartilhar
uma gravação entre treino e teste. Não é comparável diretamente ao protocolo
atual. O LOO 24/24 verifica a separabilidade dos vetores da base de fábrica
histórica; também não mede generalização para novas sessões ou desconhecidos.

Os notebooks foram salvos sem saídas antigas para evitar apresentar métricas
de um protocolo anterior como se fossem resultados do código atual.
