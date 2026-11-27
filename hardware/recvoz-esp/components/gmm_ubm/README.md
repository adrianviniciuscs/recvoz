# gmm_ubm

Classificador probabilístico **GMM-UBM diagonal** ($M = 8$, $D = 24$ em
$c_1\dots c_{12} + \Delta c_1\dots \Delta c_{12}$) com **CMN seletiva em $c_1$**
e adaptação MAP incremental ($r = 8{,}0$) para cadastro ao vivo.

Portátil (somente `libc + libm`, sem dependências do ESP-IDF), testado no host
via `make -C tests/host/gmm run`.

- `include/gmm_ubm.h`: API de extração CMN-$c_1$, pontuação LLR e adaptação MAP.
- `include/gmm_default.h`: UBM externo (*Google FLEURS PT-BR*, $1{,}66\text{ KiB}$)
  e médias adaptadas dos locutores de fábrica (gerado por `tools/export_mfcc.py`).
- `gmm_ubm.c`: implementação em precisão simples com estabilidade *log-sum-exp*.
