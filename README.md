# recvoz — reconhecimento de locutor

Protótipo de identificação de locutor em conjunto aberto (*open-set*) com
experimentos em Python e execução local no ESP32-S3, microfone INMP441 e
display SSD1306. O firmware captura voz com VAD passa-banda adaptativo, aplica
filtro de Wiener espectral casado à voz com veto de ruído por modulação
articulatória, extrai MFCC + Δ com CMN seletiva em $c_1$ e classifica via
**GMM-UBM** com UBM externo (Google FLEURS PT-BR), adaptação MAP incremental e
persistência em NVS.

## Começar

Execute na **raiz do repositório**, com Python 3.12 e GCC/Make:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
make test
make notebooks
```

Os testes usam fixtures versionadas e sinais sintéticos, sem precisar de placa
ou das gravações locais. Os notebooks e a avaliação real precisam de WAVs em
`audio_exemplo/locutor_NOME_sN.wav`. Cada arquivo representa uma sessão distinta.

```bash
make evaluate
# Resultado: docs/results/session-holdout.json
```

## Estrutura

```text
src/recvoz/             processamento, modelos exploratórios e avaliação
notebooks/             exploração do áudio e comparação de classificadores
tools/                 exportação para C, coleta serial (`grab_esp.py`, `grab_raw.py`) e `tuner.py`
tests/host/            testes C de DSP, GMM-UBM, k-NN, áudio, console e persistência
tests/python/          testes de sessões, janelas, coleta, exportação e open-set
tests/fixtures/        referências numéricas e base histórica
docs/                  arquitetura, reprodução, resultados e artigo técnico (`docs/artigo/`)
hardware/recvoz-esp/    firmware ESP-IDF 6.1 para ESP32-S3
audio_exemplo/         gravações locais, não versionadas
```

## Firmware

Depois de ativar seu ambiente ESP-IDF 6.1:

```bash
cd hardware/recvoz-esp
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

Confira a [fiação](hardware/recvoz-esp/docs/PINOUT.md) antes de ligar a placa.
Comandos: `enroll`, `list`, `reset`, `tau`, `vad`, `hold`, `diag`, `ns`, `nscal`, `grab`.
O console aguarda o fim da operação em andamento; toda captura e alteração de
estado acontece na tarefa principal.

## Documentação

- [Artigo técnico (LaTeX / PDF)](docs/artigo/recvoz_gmm_ubm.pdf) — modelagem GMM-UBM + CMN, supressão espectral e resultados
- [Guia de coleta do dataset (Open-Set e `make collect`)](docs/COLETA.md)
- [Reprodução: instalar, testar, coletar, avaliar, exportar e gravar](docs/REPRODUCAO.md)
- [Arquitetura e responsabilidades](docs/ARQUITETURA.md)
- [Protocolo de avaliação e resultados](docs/AVALIACAO.md)
- [Desenvolvimento e testes](docs/DESENVOLVIMENTO.md)
- [Firmware e componentes](hardware/recvoz-esp/README.md)
- [Calibração e demonstração](hardware/recvoz-esp/docs/CALIBRACAO.md)

## Estado da avaliação

O teste local por sessão (seed 42) acertou 9/9 janelas de duas sessões reservadas.
Isso é um resultado exploratório de **dois locutores**, não uma estimativa de
uso geral. A rejeição de desconhecidos ainda exige dados separados de calibração
e teste. Os 93,3% do experimento antigo e o LOO 24/24 da base histórica pertencem
a protocolos diferentes; veja [avaliação](docs/AVALIACAO.md).
