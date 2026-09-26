# speaker_db — quem fala? (k-NN + NVS)

k-NN **idêntico ao notebook** (`k=3`, distância cosseno, voto ponderado
por `1/d`) sobre vetores médios 39-dim. Desconhecido = menor distância
**> τ** (padrão `0.017`, calibrável).

## Base

- **Fábrica (flash, `enroll_default.h` — gerado):** LOCUTOR_A + LOCUTOR_B, 24 vetores.
- **Extras (NVS `recvoz`):** até 3 locutores × 8 vetores
  (15 s de enroll = 7 janelas de 2 s). `reset` apaga os extras.

## Uso

```c
speaker_db_init();
int idx = speaker_db_classify(mean39, &dist); // >=0 nome, -1 DESCONHECIDO
speaker_db_name(idx);
speaker_db_enroll("VISITANTE", vecs7, 7);
speaker_db_set_tau(0.020); // calibração em sala
```

## Calibrando τ

Na base atual (k-NN sem c0, só blocos com voz: 10 locutor A + 14 locutor B),
mesma-voz até 0.0119 e outra-voz desde 0.0219 (gap limpo) — o default
0.0170 fica no meio. Em sala, ajuste via serial (`tau 0.020`) até: você
e o locutor B sempre reconhecidos de várias distâncias, estranho sempre
rejeitado. Enroll ao vivo no mesmo mic ajuda mais que mexer no τ.
