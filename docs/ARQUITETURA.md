# Arquitetura

## Fluxo embarcado

```text
INMP441 → audio_in → capture/VAD → MFCC/Berouti → veto mod → CMN → GMM-UBM (speaker_db) → recognition → OLED
                        ↑                                               ↑
             tarefa principal ←─────────────── fila ←──────────── console serial
```

`main/recvoz-esp.c` coordena espera de voz, matrícula, calibração e comandos.
`capture.c` monta janelas de 32.000 amostras com blocos de voz de 100 ms.
`recognition.c` classifica e apresenta o resultado. `console.c` copia argumentos
para uma fila FreeRTOS e aguarda a resposta. `command_args.c` valida sintaxe e
intervalos antes de executar comandos.

A tarefa principal é a única dona do áudio, do DSP, do display e da base de
locutores. O console não lê I2S nem modifica configurações diretamente. Um
comando enviado durante uma captura/matrícula/exibição aguarda a operação
terminar; durante espera de voz é atendido após a leitura do bloco atual.
Uma matrícula pode durar mais de 15 s se precisar esperar voz.

`audio_in` mantém o estado do DC blocker entre leituras e solicita exatamente
as amostras restantes, inclusive quando o tamanho não é múltiplo de 256.
Chamadores não devem invocá-lo de outras tarefas.

`speaker_db` só atualiza o cadastro em RAM após o sucesso das gravações NVS e
do commit. Erros de escrita, remoção e commit são propagados. A persistência
continua usando várias chaves: isso não constitui uma transação atômica entre
chaves em caso de interrupção de energia. Uma futura versão de formato pode
usar um único blob versionado por snapshot.

## Fluxo Python

`features.py` reúne leitura, conversão PCM, VAD, filtragem e MFCC. Cada arquivo
mantém sua identidade de sessão. A segmentação é feita no áudio antes de MFCC;
uma janela de 2 s resulta em 126 frames. `evaluation.py` separa sessões e
implementa calibração e teste open-set; `models.py` contém os estimadores
exploratórios usados pelo notebook.

O exportador usa esse mesmo pipeline, valida todas as entradas e gera arquivos
em staging antes de publicá-los. A substituição é atômica por arquivo, não para
o conjunto inteiro; não rode build e exportação simultaneamente.

## Configuração DSP

16 kHz, PCM int16 mono, VAD passa-banda adaptativo ($\mathrm{RMS} > \max(300, 2\cdot\mathrm{RMS}_{\text{ruído}})$),
Butterworth ordem 5 de 80 a 7920 Hz (`filtfilt`), FFT 1024, hop 256, supressão
espectral de Berouti ($\alpha = 0{,}8$, piso $-13\text{ dB}$) com holdoff e
guarda de estacionariedade, 128 bandas mel, 13 MFCC + Δ + ΔΔ (126 frames) e
veto de ruído por modulação articulatória (`mod`).
A classificação principal usa **GMM-UBM diagonal** ($M = 8$, $D = 24$ em
$c_1\dots c_{12} + \Delta c_1\dots \Delta c_{12}$) com **CMN por janela** e
adaptação MAP incremental ($r = 8{,}0$, $768\text{ B}$ por locutor na NVS). O
$k$-NN legado (sem $c_0$, $k = 3$, distância cosseno) permanece disponível para
comparação e testes históricos. O piso de 80 dB é aplicado por frame, como no
firmware.

Python usa float64 e o firmware float32: a equivalência é numérica com
tolerâncias, não igualdade bit a bit. Supressão de ruído altera as features;
seu estado deve ser registrado em cada experimento.
