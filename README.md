# recvoz — Reconhecimento de Locutor

Código exploratório para **reconhecimento de quem está falando**
(disciplina de Sistemas Inteligentes): EDA do áudio + experimentos de
comparação de métodos, incluindo matrícula de novos locutores.
A parte embarcada (ESP32-S3 + microfone MEMS I2S)
será adicionada neste mesmo repositório.

## Estrutura

```
.
├── eda_audio.ipynb                 # exploração do sinal e das features
├── eda_speaker_recognition.ipynb   # experimentos de comparação entre métodos
└── audio_exemplo/                  # amostras locais (não versionadas)
```

## Como rodar

```bash
pip install -r requirements.txt
jupyter lab
# 1. eda_audio.ipynb — exploração do sinal
# 2. eda_speaker_recognition.ipynb — experimentos com os locutores
```

Amostras novas seguem a convenção `audio_exemplo/locutor_<nome>_s<numero>.wav`
(sessões do mesmo locutor são agrupadas automaticamente).
Os áudios são locais e **não são versionados** (ver `.gitignore`).

## Demo (2 conhecidos + 1 desconhecido)

1. Identifica os 2 locutores cadastrados.
2. Visitante fala → **rejeitado** (não cadastrado).
3. Matricula ao vivo (~15 s de voz).
4. Visitante fala de novo → **reconhecido**.
