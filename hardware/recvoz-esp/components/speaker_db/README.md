# speaker_db — quem fala? (k-NN + NVS)

k-NN **idêntico ao notebook** (`k=3`, distância cosseno, voto ponderado
por `1/d`) sobre vetores médios 39-dim. Desconhecido = menor distância
**> τ** (padrão `0.003`, calibrável).

## Base

- **Fábrica (flash, `enroll_default.h` — gerado):** ADRIAN + PEDRO, 36 vetores.
- **Extras (NVS `recvoz`):** até 3 locutores × 8 vetores
  (15 s de enroll = 7 janelas de 2 s). `reset` apaga os extras.

## Uso

```c
speaker_db_init();
int idx = speaker_db_classify(mean39, &dist); // >=0 nome, -1 DESCONHECIDO
speaker_db_name(idx);
speaker_db_enroll("VISITANTE", vecs7, 7);
speaker_db_set_tau(0.004); // calibração em sala
```

## Calibrando τ (F4)

Na base atual, mesma-voz chega a 0.0026 e outra-voz começa em 0.0016
(há overlap fino) — o default 0.0030 aceita a base toda com margem.
Em sala, ajuste via serial (`tau 0.004`) até: você e o Pedro sempre
reconhecidos de várias distâncias, estranho sempre rejeitado.
