# Reprodução de ponta a ponta

Todos os comandos abaixo partem da **raiz**, exceto o bloco de ESP-IDF.

## 1. Instalar e testar sem placa

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
make test
```

Python 3.11+, GCC e Make são necessários. O lock completo foi validado com
Python 3.12 em Linux; para outras versões use `pip install -e '.[notebooks,collect]'`. `librosa==1.0.0` é a versão usada na
referência numérica. O `pyproject.toml` centraliza dependências; `requirements.txt`
continua disponível como entrada compatível para `pip install -r`.

## 2. Coletar

Salve cada sessão como `audio_exemplo/locutor_NOME_sN.wav`, preferencialmente
PCM16 mono a 16 kHz, com o mesmo INMP441, distância e ganho da demonstração.
Use nomes ASCII de até 15 caracteres. Não junte sessões no mesmo arquivo.
Grave pelo menos 15–20 s por sessão; colete três sessões por conhecido para
separar treino, calibração e teste. Varie dia, posição e fala entre sessões.

### Opção A (recomendada): direto do firmware `recvoz-esp`

Para uma coleta guiada com interface gráfica (seleção de grupo, teleprompter,
numeração automática `sN`, checagem de janelas VAD/auto-ruído e reprodução):

```bash
make collect
# ou: .venv/bin/python tools/coletor.py
```

Via linha de comando, o firmware `recvoz-esp` expõe o comando serial
`grab [1..3]` (PCM cru em Base64) e o utilitário `tools/grab_esp.py` encadeia
blocos consecutivos na mesma conexão serial sem trocar de firmware:

```bash
.venv/bin/python tools/grab_esp.py -d 18 \
  -o audio_exemplo/locutor_NOME_s1.wav
# Ou especificando a porta manualmente:
.venv/bin/python tools/grab_esp.py -p /dev/ttyACM0 -d 18 \
  -o dados/desconhecidos_teste/locutor_IMPOSTOR1_s1.wav
```

Para ajustar parâmetros de VAD, supressão espectral e classificação de forma
interativa com áudio gravado da placa, use `tools/tuner.py`.

### Opção B (legado): firmware de aquisição `inmp441_mic`

Para um firmware de aquisição que transmita o protocolo serial binário
`AA 55 | tamanho uint16 LE | PCM int16 LE`, use:

```bash
.venv/bin/python tools/grab_raw.py -p /dev/ttyACM0 -d 20 \
  -o audio_exemplo/locutor_NOME_s1.wav
```

**O firmware `recvoz-esp` não transmite o protocolo binário `AA 55` do `grab_raw.py`.**
Depois da coleta com `inmp441_mic`, grave novamente o firmware `recvoz-esp`.
Também é possível fornecer WAVs de outro gravador, registrando microfone e
processamento utilizados. O pipeline assume PCM após DC blocker; não reaplica
esse filtro aos WAVs.

## 3. Avaliar

```bash
make evaluate
```

Para medir rejeição de desconhecidos, mantenha três diretórios: conhecidos,
desconhecidos de calibração e desconhecidos de teste. Os dois últimos devem
conter **pessoas diferentes**, também distintas dos conhecidos.

```bash
PYTHONPATH=src .venv/bin/python -m recvoz.evaluation \
  --audio-dir audio_exemplo \
  --unknown-cal dados/desconhecidos_cal \
  --unknown-test dados/desconhecidos_teste \
  --output docs/results/open-set.json
```

`--noise-wav silencio.wav` aplica um perfil fixo calibrado nas primeiras 8192
amostras. Compare resultados com e sem esse argumento. Para reproduzir essa
condição na placa: `ns auto off`, `ns on`, `nscal` em silêncio; para o cenário
sem supressão: `ns off`. A adaptação automática do ambiente não é simulada pela
avaliação offline. O gatilho inicial de 300 ms do firmware também é excluído da
segmentação offline; as janelas e o VAD de captura são compartilhados.

## 4. Exportar e revisar

```bash
.venv/bin/python tools/export_mfcc.py \
  --audio-dir audio_exemplo \
  --fixture-wav audio_exemplo/locutor_NOME_s1.wav \
  --output-root export
```

A saída inclui tabelas C, base de locutores, fixtures e manifesto com hashes das
entradas. A quantidade de locutores é calculada a partir dos dados, limitada a
8 pela máscara de override do firmware. Os WAVs precisam ter voz suficiente;
entradas inválidas interrompem o processo antes de sobrescrever as saídas.
O export usa supressão desligada. Para uso com supressão, calibre e valide o
sistema nessa condição, preferencialmente com matrícula ao vivo no ambiente.

A base de fábrica atualmente versionada é a **base histórica de 24 vetores**.
O export novo gerou 23 vetores com os dados locais, pois preserva sessões e
calcula MFCC por janela. Não se substituiu automaticamente o modelo e seu limiar:
faltam dados independentes para validar a rejeição de desconhecidos. Quando
esses dados estiverem disponíveis, revise o resultado e publique a exportação:

```bash
# Mesmo comando de exportação, agora com --output-root .
# Após revisar a base e recalibrar tau:
make test
```

Se mudar a base, atualize as expectativas históricas do teste k-NN e registre
os resultados correspondentes. Não escolha tau usando o conjunto de teste.

## 5. Compilar e gravar

Ative o ESP-IDF 6.1 instalado na sua máquina (`source CAMINHO/esp-idf/export.sh`).
Depois:

```bash
cd hardware/recvoz-esp
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

## 6. Calibrar na sala

Confira [PINOUT](../hardware/recvoz-esp/docs/PINOUT.md) e
[CALIBRACAO](../hardware/recvoz-esp/docs/CALIBRACAO.md). Teste voz conhecida,
desconhecida, silêncio, matrícula, cancelamento pelo botão e reinício com NVS.
`tau` deve refletir a base e a configuração de supressão em uso.
