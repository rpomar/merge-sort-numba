# Merge Sort com Numba: sequencial e MPI

Merge Sort em Python compilado com [Numba](https://numba.pydata.org/), em duas frentes:

1. **vários vetores aleatórios de tamanhos variados** (hackathon de programação paralela): uma
   execução **sequencial** de referência comparada com uma execução **paralela com MPI**
   ([mpi4py](https://mpi4py.readthedocs.io/)), que reparte os vetores entre processos;
2. **um único vetor** do `dados.csv` dos professores, só sequencial (seção
   [Merge Sort sequencial no dados.csv](#merge-sort-sequencial-no-dadoscsv)).

Trabalho da disciplina de Programação Paralela e Distribuída.

### Sequencial x MPI x Numba

- **Merge Sort sequencial**: um processo ordena todos os vetores, um depois do outro.
- **MPI**: `P` processos independentes (cada um com seu interpretador Python e sua memória)
  recebem lotes de vetores, ordenam cada vetor com **o mesmo** Merge Sort sequencial e devolvem
  o resultado ao processo 0. O paralelismo está **entre vetores**: cada vetor é ordenado
  inteiro por um único processo.
- **Numba** só compila o Merge Sort para código de máquina. Ele acelera as duas versões igualmente
  e **não** é paralelismo: não há threads, `prange` nem `multiprocessing` neste programa.

## Arquivos

| arquivo | função |
|---|---|
| `merge_mpi.py` | **programa do hackathon**: gera os cenários, mede sequencial x MPI, valida e grava o CSV |
| `vetores.py` | cenários, gerador de vetores, distribuição entre processos e validação (sem MPI) |
| `merge_numba.py` | núcleo do Merge Sort compilado com `@njit` (usado pelos dois programas) e medição no `dados.csv` |
| `merge_seq.py` | leitura do CSV e Merge Sort em Python puro (referência para `merge_numba.py --python`) |
| `tests/` | testes automatizados (`pytest`) |
| `requirements.txt` | dependências Python |

## Dependências

- Python 3.10 ou mais novo (testado com Python 3.14.6)
- NumPy, Numba, mpi4py e pytest (testado com NumPy 2.5.4, Numba 0.68.0, mpi4py 4.1.2, pytest 9.1.1)
- uma implementação de MPI que forneça o `mpiexec` (testado com Intel MPI 2021.18 no Windows)

## Instalação local

Tudo fica num ambiente virtual dentro da pasta do projeto (`.venv`, ignorado pelo git).

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Para o `mpiexec`, há duas opções:

- instalar o [Microsoft MPI](https://learn.microsoft.com/message-passing-interface/microsoft-mpi)
  (`msmpisetup.exe`), que coloca o `mpiexec` no PATH do sistema; ou
- instalar o runtime do Intel MPI **dentro do venv**, sem mexer no sistema (foi o que se usou
  nos testes deste README). O `mpiexec.exe` fica em `.venv\Library\bin`, que precisa entrar no
  PATH da sessão:

  ```powershell
  pip install impi_rt
  $env:PATH = "$PWD\.venv\Library\bin;$env:PATH"
  ```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O `mpiexec` vem do Open MPI ou do MPICH do sistema (`sudo apt install openmpi-bin` no
Debian/Ubuntu, `brew install open-mpi` no macOS). Sem acesso de administrador,
`pip install mpich` instala o MPICH dentro do venv. Nestes sistemas o programa **não foi
testado** aqui; os comandos são os mesmos do Windows.

Para conferir se o MPI funciona:

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)"
```

Deve imprimir `0 2` e `1 2` (em qualquer ordem).

## Como executar

Tudo roda numa única máquina: o `mpiexec -n P` abre `P` processos locais.

```bash
python merge_mpi.py --help
python merge_mpi.py --cenario bordas                       # 1 processo, sem mpiexec
mpiexec -n 4 python -m mpi4py merge_mpi.py                 # todos os cenários, 4 processos
mpiexec -n 2 python -m mpi4py merge_mpi.py --cenario variados --vetores 50 --semente 7
mpiexec -n 8 python -m mpi4py merge_mpi.py --cenario grandes --vetores 3   # processos sem trabalho
```

O `-m mpi4py` faz um erro em qualquer processo encerrar todos os outros, em vez de deixá-los
esperando para sempre numa comunicação.

| opção | padrão | significado |
|---|---|---|
| `--cenario` | `todos` | `pequenos`, `grandes`, `variados`, `bordas` ou `todos` |
| `--semente` | 42 | semente do gerador; a mesma semente gera os mesmos vetores |
| `--repeticoes` | 3 | medições de cada versão; speedup e eficiência usam a **menor** |
| `--vetores` | do cenário | quantidade de vetores |
| `--tam-min`, `--tam-max` | do cenário | tamanho mínimo e máximo de cada vetor (0 permite vazios) |
| `--vmin`, `--vmax` | do cenário | intervalo dos valores (dentro de `int32`, aceita negativos) |
| `--csv` | `resultados.csv` | arquivo em que as linhas são **acrescentadas** |

Se algum resultado estiver errado, o programa imprime os índices dos vetores errados e termina
com código **1**. Parâmetros inválidos terminam com código 2 e uma mensagem do `argparse`.

## Cenários

| cenário | vetores | tamanho | valores | para quê |
|---|---|---|---|---|
| `pequenos` | 5000 | 0 a 2.000 | ±1.000.000 | muitos vetores pequenos: pouca conta por mensagem |
| `grandes` | 8 | 1.000.000 a 2.000.000 | ±1.000.000 | poucos vetores grandes: difícil equilibrar com muitos processos |
| `variados` | 100 | 0 a 400.000 | ±1.000.000 | tamanhos bem diferentes entre si |
| `bordas` | 200 | 0 a 3 | −2 a 2 | vazios, unitários, repetidos e negativos |

Tamanhos e valores são sorteados com distribuição uniforme (`numpy.random.default_rng(semente)`).
Os vetores são `int32`, o tipo do núcleo Numba. Valores repetidos aparecem em todos os cenários
(o intervalo de valores é menor que o total de elementos) e os negativos também.

## O que cada medição inclui

A geração dos vetores e a distribuição dos índices acontecem **antes** das medições. A
compilação do Numba também: cada processo faz uma chamada de aquecimento.

| medição | começa | termina | inclui |
|---|---|---|---|
| sequencial (`tempo_seq_s`) | antes de ordenar o 1º vetor no processo 0 | depois do último | cópia de cada vetor + Merge Sort, um vetor por vez |
| MPI (`tempo_par_s`) | depois de uma barreira com todos os processos | quando o processo 0 tem todos os resultados na ordem original | montagem dos lotes, serialização (pickle), envio (`scatter`), cópia + Merge Sort em cada processo, espera pelo mais lento, coleta (`gather`) e remontagem na ordem |

A validação (comparar com `numpy.sort` e entre as versões) fica fora das duas medições.

## Distribuição do trabalho

`vetores.distribuir` usa a heurística gulosa do **maior primeiro** (LPT): os vetores são
percorridos do maior para o menor e cada um vai para o processo com **menos elementos
acumulados** até o momento (empate: o que tem menos vetores, para espalhar os vazios). O peso é o
tamanho do vetor, uma aproximação simples do custo do Merge Sort.

Cada vetor leva seu índice original. O processo 0 envia um lote para cada processo (inclusive para
si), recebe de volta pares `(índice, vetor ordenado)` e recoloca cada vetor na posição de origem.
Com mais processos do que vetores, os processos excedentes recebem lotes vazios, participam das
comunicações e não ordenam nada; o programa avisa quantos ficaram sem trabalho.

## CSV

Cada execução acrescenta uma linha por cenário em `resultados.csv`:

| coluna | conteúdo |
|---|---|
| `data_hora` | momento da gravação |
| `cenario`, `semente` | cenário e semente usados |
| `processos`, `processos_ociosos` | processos MPI e quantos ficaram sem vetores |
| `vetores`, `total_elementos` | quantidade de vetores e soma dos tamanhos |
| `tam_min`, `tam_max`, `vmin`, `vmax` | parâmetros do gerador |
| `repeticoes` | medições de cada versão |
| `tempo_seq_s`, `tempo_par_s` | menor tempo sequencial e menor tempo MPI, em segundos |
| `tempo_seq_medio_s`, `tempo_par_medio_s` | médias das repetições |
| `speedup` | `tempo_seq_s / tempo_par_s` |
| `eficiencia` | `speedup / processos` |
| `desequilibrio` | maior carga / carga média, em elementos (1,0 = cargas iguais) |
| `validacao` | `ok` ou `falhou` |

## Como interpretar

- **speedup > 1**: o MPI terminou antes do sequencial. O teto teórico é o número de processos.
- **eficiência**: quanto de cada processo foi aproveitado (1,0 = ideal). Ela cai quando a
  comunicação pesa, quando a carga fica desigual ou quando há mais processos do que núcleos.
- **desequilíbrio**: em `grandes` com 8 processos e 8 vetores cada processo recebe um vetor, e o
  tempo é ditado pelo maior. Nenhuma distribuição de vetores inteiros resolve isso.
- **MPI pode ser mais lento** em cargas pequenas: cada processo é um programa separado, e os
  vetores são serializados, copiados entre processos e copiados de volta. Quando ordenar leva
  menos que esse transporte (cenário `bordas`, ou qualquer cenário com 1 processo), o speedup fica
  abaixo de 1.
- Os números **dependem do computador**, do número de processos e da carga da máquina no momento.
  Processadores híbridos (núcleos de desempenho e de eficiência) e o modo de energia mudam bastante
  os tempos; repita as medições e compare execuções feitas nas mesmas condições.

### Exemplo de saída

Uma execução real com 4 processos (Windows 11, Intel Core Ultra 7 255H, 16 núcleos, Intel MPI
2021.18); na sua máquina os tempos serão outros:

```
4 processo(s) MPI, semente 42, menor de 3 medição(ões)

cenário   vetores   elementos   seq(s)   mpi(s) speedup  efic. deseq. validação
-------------------------------------------------------------------------------
pequenos    5,000   4,986,460   0.1451   0.1077   1.348  0.337  1.000        ok
grandes         8  12,031,583   0.6243   0.2694   2.318  0.579  1.041        ok
variados      100  21,069,636   0.9495   0.4275   2.221  0.555  1.001        ok
bordas        200         306   0.0001   0.0005   0.186  0.047  1.007        ok

resultados acrescentados em resultados.csv
```

## Testes

```bash
python -m pytest
```

| arquivo | o que testa |
|---|---|
| `tests/test_ordenacao.py` | Merge Sort sequencial: vazio, unitário, repetidos, negativos, extremos de `int32`, tamanhos em volta dos blocos de 16 |
| `tests/test_vetores.py` | gerador (semente, limites, tamanhos desiguais, parâmetros inválidos), distribuição (cada vetor uma vez, equilíbrio, mais processos que vetores, vazios) e validação |
| `tests/test_mpi.py` | ordem original dos resultados, execução com 1 processo, falha de validação com código 1, argumentos inválidos e, **se houver `mpiexec` no PATH**, execuções reais com 2 processos e com mais processos do que vetores |

Sem `mpiexec`, os dois testes com vários processos aparecem como `SKIPPED`; sem `mpi4py`, o
arquivo `test_mpi.py` inteiro é pulado. Os demais rodam normalmente.

## Limitações

- O paralelismo é só entre vetores: um vetor enorme continua sendo ordenado por um processo.
- O peso da distribuição é o tamanho `n`, não `n log n`, e não considera o custo de comunicação.
- `scatter`/`gather` com pickle são simples, mas copiam e serializam os dados; versões com buffer
  (`Scatterv`/`Gatherv`) seriam mais rápidas, ao custo de calcular deslocamentos.
- Todos os vetores são gerados e reunidos no processo 0, que precisa de memória para todos eles
  (entrada, resultado sequencial e resultado MPI).
- Vetores `int32` apenas, como no núcleo Numba.
- Testado em uma única máquina com Windows; não foi executado em cluster nem em Linux/macOS.

---

## Merge Sort sequencial no dados.csv

`merge_numba.py` aplica o mesmo núcleo ao vetor do `dados.csv` fornecido pelos professores
(1.000.000 de inteiros em uma única linha, separados por `;`). O `dados.csv` **não está no
repositório**: coloque o arquivo nesta pasta ou passe o caminho dele como argumento.

```bash
python merge_numba.py [dados.csv] [repeticoes] [--python]
```

| argumento | padrão | significado |
|---|---|---|
| `dados.csv` | `dados.csv` | caminho do arquivo dos professores |
| `repeticoes` | 5 | execuções por entrada; são mostrados o **menor** tempo e a média |
| `--python` | — | também mede o Merge Sort em Python puro (bem mais lento) |

As quatro entradas medidas, todas com os mesmos 1.000.000 de valores:

| entrada | o que é |
|---|---|
| aleatória | o próprio `dados.csv` |
| ordenada | os valores já em ordem crescente |
| invertida | os valores em ordem decrescente |
| PIOR CASO | a ordem que obriga **toda** intercalação a comparar até o último elemento |

O vetor ordenado é gravado em `dados_ordenados_numba.csv`.

### Exemplo de saída

```
1,000,000 valores lidos de dados.csv em 0.24s
compilação/carga do Numba: 0.46s

entrada               n  melhor(s)   media(s)  válido
-----------------------------------------------------
aleatória     1,000,000     0.0511     0.0821     sim
ordenada      1,000,000     0.0319     0.0395     sim
invertida     1,000,000     0.0350     0.0458     sim
PIOR CASO     1,000,000     0.0714     0.0842     sim

vetor ordenado (entrada aleatória) gravado em dados_ordenados_numba.csv
```

A primeira execução compila o código (cerca de 4 s, fora da medição). As seguintes carregam o
código compilado do cache (`__pycache__`) e começam em menos de 1 s. Com `--python`, as medições
do Python puro rodam entre as do Numba e deixam os tempos do Numba mais lentos e instáveis.

### Comparação com C e Python puro

Mesmo `dados.csv`, mesmo algoritmo (bottom-up) no C e no Numba, menor tempo de 7 execuções,
cada versão rodando sozinha:

| entrada | C (`gcc -O2`, WSL) | **Numba** | Python puro |
|---|---|---|---|
| aleatória | 0,053 s | **0,050 s** | 3,68 s |
| ordenada | 0,008 s | **0,025 s** | 0,60 s |
| invertida | 0,012 s | **0,029 s** | 0,63 s |
| PIOR CASO | 0,047 s | **0,049 s** | 3,88 s |

- Nas entradas **aleatória** e **pior caso**, em que o trabalho é intercalar, o Numba **empata com
  o C** e é ~75x mais rápido que o Python puro.
- Nas entradas **ordenada** e **invertida**, os atalhos transformam quase tudo em cópias de
  blocos: o C usa `memcpy`, mais rápido que a cópia por fatia do Numba (2-3x de diferença, em
  centésimos de segundo).
- As saídas do C, do Numba e do Python puro são idênticas byte a byte.

### Como funciona o núcleo

`@njit` compila as três funções do núcleo (`insercao`, `intercalar`, `merge_sort`) para código de
máquina via LLVM, sem o interpretador Python. Para o resultado ficar próximo de C, o código segue
o estilo de C, não o de Python:

1. **arrays NumPy `int32` contíguos** (como o `int[]` do C), não listas de objetos;
2. **bottom-up, sem recursão**, com dois buffers alternados ("ping-pong"): cada passada lê de um e
   escreve no outro, sem copiar de volta;
3. **intercalação sem desvio condicional** (o LLVM gera `cmov`), o que ajuda no pior caso, em que
   a próxima escolha é imprevisível;
4. **blocos de 16 elementos ordenados por inserção** antes de intercalar;
5. **atalhos O(1)** para pares de blocos já em ordem ou em ordem inversa (viram cópias);
6. a **compilação é feita antes da medição**, com uma chamada de aquecimento.

O **pior caso** é construído para as divisões exatas do bottom-up: em cada intercalação, o maior e
o segundo maior valor ficam em lados diferentes (L + R − 1 comparações, o máximo), e os blocos de
16 ficam em ordem inversa (pior caso da inserção). Assim nenhum atalho dispara.

### Validação

Cada resultado é comparado elemento a elemento com `numpy.sort`. A leitura confere se todos os
valores cabem em `int32` e avisa com uma mensagem clara se o arquivo não existir.
