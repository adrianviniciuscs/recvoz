# Fixtures

Os `.bin` são referências numéricas em little-endian, atualizadas pelo pipeline
`pcm-window-32000-v1`: SOS, log com piso por frame e janelas independentes.
`manifest.json` registra a versão e os hashes dos áudios usados. O PCM versionado
permite executar os testes sem acesso às gravações locais completas.

`enroll_vectors.npz` permanece como artefato histórico dos 24 vetores da base de
fábrica; não foi substituído pela exportação de 23 vetores do novo protocolo.
A base C histórica também foi preservada até existir validação do novo limiar.

Para gerar todos os artefatos novos em pasta separada, siga
[REPRODUCAO](../../docs/REPRODUCAO.md). O teste MFCC aceita opcionalmente o caminho
das fixtures com barra final: `./hosttest /caminho/tests/fixtures/`.
