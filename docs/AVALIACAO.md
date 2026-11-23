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

## Resultado local — seed 42

Veja [relatório JSON](results/session-holdout.json):

- Treino: ADRIAN/s1, PEDRO/s2, PEDRO/s3 e PEDRO/s4.
- Teste: ADRIAN/s2 e PEDRO/s1, total de nove janelas.
- Identificação fechada: **9/9**.
- Configuração: VAD 300, sem c0, supressão desligada.
- Rejeição open-set: **não avaliada**. Faltam desconhecidos de calibração/teste
  e uma terceira sessão de ADRIAN.

Este resultado não valida o limiar 0,017 nem o firmware com supressão automática.
A execução na placa e a coleta de novos locutores continuam necessárias.

## Resultados históricos

O antigo 93,3% ± 4,8% usava splits aleatórios de segmentos, podendo compartilhar
uma gravação entre treino e teste. Não é comparável diretamente ao protocolo
atual. O LOO 24/24 verifica a separabilidade dos vetores da base de fábrica
histórica; também não mede generalização para novas sessões ou desconhecidos.

Os notebooks foram salvos sem saídas antigas para evitar apresentar métricas
de um protocolo anterior como se fossem resultados do código atual.
