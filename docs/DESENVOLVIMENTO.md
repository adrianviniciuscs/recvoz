# Desenvolvimento

## Comandos

Na raiz: `make test`, `make test-host`, `make test-python`, `make evaluate` e
`make notebooks`. Use `make test PYTHON=python` em um ambiente já ativado.

Os testes C compilam com `-Wall -Wextra -Werror`. Os Makefiles dependem dos
headers; alterar tabelas ou contratos públicos provoca recompilação. A CI
executa testes sem os áudios privados e compila o firmware com ESP-IDF 6.1.

## Cobertura

- k-NN: distâncias, votos, rejeição, argumentos e regressão da base histórica.
- MFCC: referência Python/C, contagem de frames e distância do vetor médio.
- Supressão: bypass, piso, atualização do perfil e sinal sintético com ruído.
- Áudio: consumo exato de 1.600 amostras, leituras parciais e leitura vazia.
- Console: argumentos inválidos, limites e valores não finitos.
- Persistência: falhas de escrita/commit não publicam alterações em RAM.
- Python: separação por sessão, ausência de mistura de gravações, timeout de
  captura, perfil de ruído zero, métricas open-set, limiar independente do teste
  e exportação com número variável de locutores.

Os testes de áudio e NVS usam doubles mínimos da API ESP-IDF; não comprovam
comportamento elétrico, concorrência do RTOS ou resistência a falta de energia.
O build ESP-IDF complementa a checagem de interfaces reais.

## Alterações no DSP

Mude `src/recvoz/features.py`, confira a implementação C e exporte para uma
pasta separada. Compare as novas fixtures com o teste de host e revise o
manifesto. Alterações de features exigem recalibrar a base e tau.
Não edite manualmente `mfcc_tables.h` ou `enroll_default.h`.

## Notebooks

Mantenha funções reutilizáveis em `src/recvoz`. Notebooks organizam os gráficos
e a interpretação; salve sem saídas pesadas antes de versionar. Execute-os
inteiros após mudanças no pipeline, usando os áudios locais.
