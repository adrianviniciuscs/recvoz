# recvoz-esp — ESP32-S3 / ESP-IDF 6.1

Reconhecimento de locutor local: captura 2 s de voz com VAD passa-banda, aplica
supressão espectral (Berouti) e veto de ruído por modulação, extrai MFCC 39,
classifica via **GMM-UBM + CMN** (com adaptação MAP e $k$-NN legado), mostra o
resultado no OLED e permite matrícula em NVS.

| Componente | Responsabilidade |
|---|---|
| `audio_in` | I2S INMP441, PCM mono 16 kHz e DC blocker contínuo |
| `mfcc` | DSP portátil, supressão espectral de Berouti e métrica `mod` |
| `knn` | Distância cosseno, voto ponderado e rejeição (legado/comparação) |
| `speaker_db` | UBM + GMM-UBM + adaptação MAP incremental, base de fábrica e NVS |
| `recvoz_display` | SSD1306: estado, nome e placar |
| `main/capture.c` | VAD passa-banda adaptativo e janela de voz |
| `main/console.c` | Fila de comandos (inclui `grab` para coleta via `tools/grab_esp.py`) |
| `main/recognition.c` | Veto de ruído, classificação e apresentação |
| `main/recvoz-esp.c` | Coordenação na tarefa principal |

## Compilar

Com ESP-IDF 6.1 ativado, execute **nesta pasta**:

```bash
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

Execute testes e exportação **na raiz do repositório**:

```bash
make test
.venv/bin/python tools/export_mfcc.py --help
```

## Documentos

- [PINOUT](docs/PINOUT.md): fiação e pinos.
- [TREINO](docs/TREINO.md): coleta, exportação e base histórica.
- [MFCC](docs/MFCC.md): equações e tolerâncias numéricas.
- [CLASSIFICACAO](docs/CLASSIFICACAO.md): decisão e rejeição.
- [CALIBRACAO](docs/CALIBRACAO.md): comandos e demonstração.
- [Reprodução completa](../../docs/REPRODUCAO.md).

A base versionada tem dois locutores e 24 vetores históricos. Matrícula salva
até três extras. O exportador novo suporta 1–8 locutores de fábrica e precisa
de nova validação do limiar antes de substituir a base histórica.
