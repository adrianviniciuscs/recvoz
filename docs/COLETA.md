# Guia de Coleta do Dataset (Open-Set)

Este documento explica o papel de cada grupo de gravações no coletor (`make collect`), por que essa separação é necessária e o passo a passo completo para montar o dataset sem vazamento de dados (*data leakage*).

---

## 1. O que é cada grupo?

Em reconhecimento de locutor em **conjunto aberto** (*open-set*), o sistema precisa fazer duas coisas ao mesmo tempo:
1. **Identificar** corretamente quem está cadastrado (**Conhecidos**);
2. **Rejeitar** como `DESCONHECIDO` qualquer pessoa fora do cadastro (**Impostores**).

Para medir isso de forma honesta (sem "viciar" o limiar de decisão $\tau$), as gravações são divididas em **quatro pastas**:

| Grupo no coletor | Pasta | Quem gravar | Quanto gravar | Para que serve |
| :--- | :--- | :--- | :--- | :--- |
| **Conhecido** | `audio_exemplo/` | Pessoas cadastradas no sistema (ex.: Adrian, Pedro, +1 ou +2 se puder) | **Mínimo 3 sessões** de 18 s por pessoa (`s1`, `s2`, `s3`) | Separar **treino**, **calibração de $\tau$** e **teste final** em sessões totalmente independentes. |
| **Impostor (Calibração)** | `dados/desconhecidos_cal/` | Pessoas **não cadastradas** | **2 a 3 pessoas** (1 sessão de 18 s cada) | Ajudar a escolher o melhor limiar $\tau^*$ que separa conhecidos de desconhecidos. |
| **Impostor (Teste)** | `dados/desconhecidos_teste/` | **Outras** pessoas **não cadastradas** | **3 a 5 pessoas** (1 sessão de 18 s cada) | Medir **FAR** (falsa aceitação), **FRR** (falsa rejeição) e **EER** em vozes que o sistema nunca viu. |
| **Ruído / Silêncio** *(opcional)* | `dados/ruido/` | Ambiente sem fala (sala vazia, ventilador, ar) | 1 a 2 arquivos de 10 s | Testar o veto de ruído estacionário (`mod`) e comparar `--noise-wav` fixo vs. automático. |

### Por que separar *Impostor (Calibração)* de *Impostor (Teste)*?

```text
FASE 1 — TREINO            FASE 2 — CALIBRAÇÃO (escolhe τ*)      FASE 3 — TESTE FINAL (relatório/artigo)
─────────────────────────  ────────────────────────────────────  ─────────────────────────────────────────
Conhecidos (sessões        Conhecidos (1 sessão reservada)       Conhecidos (1 sessão inédita)
  restantes, ex.: s1)        + Impostores de Calibração            + Impostores de Teste
       │                              │                                     │
       ▼                              ▼                                     ▼
  Adapta os modelos         Encontra o limiar τ* que               Mede Acurácia, FAR, FRR e EER
  GMM-UBM / k-NN            equilibra FAR e FRR                    sem nenhum dado vazado
```

- Se você escolhesse $\tau$ olhando os mesmos impostores do teste, a taxa de erro ficaria artificialmente otimista.
- Por isso, a **regra de ouro** é: **nunca repetir a mesma pessoa** entre *Conhecidos*, *Impostores (Calibração)* e *Impostores (Teste)*.
- Nos **Conhecidos**, por que **3 sessões** (`s1`, `s2`, `s3`)? Porque uma vai para o treino, uma para calibrar $\tau$ e uma para o teste — nunca misturando janelas da mesma gravação entre treino e teste.

---

## 2. Preciso gravar tudo na mesma sala ou gravar ruído separado?

**Não!**
- **Salas diferentes são até melhores** para o experimento, pois testam a robustez real do CMN + supressão espectral entre ambientes (*cross-room*).
- O pipeline em [features.py](../src/recvoz/features.py) estima o perfil de ruído **automaticamente dentro de cada arquivo WAV** (`auto_noise_profile`) usando os blocos mais silenciosos da própria gravação (`0,512 s`).
- **Única regra prática:** peça para a pessoa **esperar ~1 segundo em silêncio no início** (antes de começar a ler o texto). Isso já garante o perfil de ruído daquela sala dentro do próprio arquivo.

---

## 3. Passo a passo da coleta com `make collect`

### Antes de começar
1. Conecte o ESP32-S3 (já com o firmware `recvoz-esp`) preferencialmente pela **USB nativa (`/dev/ttyACM0`)** — a transferência via USB-CDC é quase instantânea.
2. Apoie a placa firme na mesa, a **~20–25 cm** da boca do locutor (evite segurar na mão para não gerar ruído mecânico no INMP441).
3. Abra o coletor gráfico na raiz do repositório:
   ```bash
   make collect
   ```

### Durante cada gravação (~1 minuto por pessoa)
1. Confira a **Porta** detectada no topo (clique em *Atualizar* se conectou depois) e deixe a duração em **18s** (padrão).
2. Selecione o **Destino** (*Conhecido*, *Impostor Calibração*, *Impostor Teste* ou *Ruído*).
3. Digite o **Nome** do locutor (sem acento; o app já limpa e sugere o próximo `sN` automaticamente, ex.: `locutor_adrian_s3.wav`).
4. Peça para a pessoa:
   - **Esperar ~1 segundo em silêncio** ao clicar em `● GRAVAR`;
   - **Ler em voz normal** o texto exibido no teleprompter (enquanto o OLED mostrar `GRAVANDO`).
5. Ao terminar, confira a linha de **Checagem Automática**:
   - **✓ OK (verde):** ≥ 5 janelas válidas no VAD, sem saturação (`pico < 0.98`) e `auto-ruído: OK`. Pode passar para a próxima pessoa/sessão!
   - **⚠ ATENÇÃO (amarelo):** se deu poucas janelas, clipping ou faltou pausa silenciosa, clique em **Ouvir último** para conferir e **Regravar último** se precisar.
6. Para a próxima sessão do mesmo conhecido (`s1` → `s2` → `s3`), o app já troca o texto do teleprompter e incrementa o `sN` — basta mudar um pouco a posição/distância ou dar um intervalo entre sessões.

> **Nota sobre privacidade/Git:** as pastas `dados/` e `audio_exemplo/*.wav` estão no `.gitignore`, então os áudios brutos ficam apenas na sua máquina e nunca sobem acidentalmente para o GitHub.

---

## 4. O que fazer depois da coleta

### A. Rodar a avaliação *open-set*
```bash
PYTHONPATH=src .venv/bin/python -m recvoz.evaluation \
  --audio-dir audio_exemplo \
  --unknown-cal dados/desconhecidos_cal \
  --unknown-test dados/desconhecidos_teste \
  --output docs/results/open-set.json
```

### B. Teste rápido ao vivo no ESP32-S3 (para fechar a Tabela IV do artigo)
Antes de liberar um voluntário não cadastrado, abra o monitor serial (`idf.py -p /dev/ttyACM0 monitor`):
1. **Impostor ($S_u$):** 5 falas e anotar os LLRs (esperado: `DESCONHECIDO`).
2. **Cadastro ao vivo ($S_v$):** digitar `enroll VISITANTE`, gravar as 7 janelas e testar 5 falas (e após `reboot` para confirmar a NVS).
3. **Conhecido ($S_2$):** 5 falas do segundo conhecido cadastrado.
4. **Ruído puro:** ventilador/ar ligado sem fala (esperado: veto `RUIDO AMBIENTE` com o valor de `mod`).
