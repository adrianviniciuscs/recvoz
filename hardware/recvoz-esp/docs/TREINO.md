# Treino e base embarcada

O roteiro atualizado está em [REPRODUCAO](../../../docs/REPRODUCAO.md) e o
protocolo de métricas em [AVALIACAO](../../../docs/AVALIACAO.md).

## Coleta e processamento

Use `locutor_NOME_sN.wav`: cada arquivo é uma sessão, nomes ASCII de até 15
caracteres. Grave pelo menos três sessões por conhecido e pessoas distintas
para os conjuntos de desconhecidos de calibração e teste.
O utilitário `tools/grab_raw.py` exige o firmware externo de aquisição
`inmp441_mic`; o console recvoz não transmite áudio binário.

O pipeline compartilhado em `src/recvoz/features.py` aplica VAD por blocos de
100 ms e calcula MFCC em janelas independentes de 32.000 amostras. Cada janela
produz 126 frames e um vetor médio de 39 dimensões. Sessões nunca são
concatenadas. A classificação ignora c0.

## Exportação

Na raiz, forneça explicitamente os dados e o áudio de referência:

```bash
.venv/bin/python tools/export_mfcc.py \
  --audio-dir audio_exemplo \
  --fixture-wav audio_exemplo/locutor_NOME_s1.wav \
  --output-root export
```

Gera `mfcc_tables.h`, `enroll_default.h`, fixtures, vetores com sessão de origem
e manifesto. Todos os dados são validados antes de publicar os arquivos.
Use `--output-root .` apenas ao publicar um modelo já avaliado; execute os
testes e calibre tau para essa base. Não ajuste o limiar no conjunto de teste.

## Base histórica preservada

A base versionada conserva 24 vetores obtidos pela concatenação antiga:
10 do locutor A e 14 do locutor B. O teste LOO histórico retorna 24/24, com
maior distância da mesma voz 0,0119 e menor da outra voz 0,0219. O limiar
histórico é 0,017. Esses números não validam rejeição de pessoas desconhecidas.

O novo exportador gera 23 vetores a partir dos áudios locais, preservando
sessões. Essa base foi gerada separadamente para validação do exportador;
a publicação depende de dados de calibração open-set ainda ausentes.

## Persistência

Fábrica fica em flash; extras e override ficam em NVS. Matrícula com nome de
fábrica substitui seus vetores ativos; `reset` remove extras e override.
Os cadastros em RAM só mudam após sucesso das operações NVS.
