# Merge Sort com Numba: sequencial e MPI

Ordenar milhares de vetores um atrás do outro leva tempo, mesmo com um Merge Sort rápido. Neste
projeto, nós dividimos esses vetores entre vários processos com MPI e medimos, com números reais,
quando isso compensa e quando não compensa.

O projeto nasceu na disciplina de Programação Paralela e Distribuída e tem duas partes:

- **Vários vetores, sequencial x MPI** (hackathon): `merge_mpi.py` gera vetores aleatórios de
  tamanhos variados, ordena tudo em um processo e depois reparte o trabalho entre `P` processos.
- **Um vetor grande, só sequencial**: `merge_numba.py` ordena o `dados.csv` dos professores
  (1.000.000 de inteiros) e compara com C e com Python puro
  ([veja abaixo](#merge-sort-sequencial-no-dadoscsv)).

## Antes de começar: o que é paralelo aqui

Vale separar três coisas que costumam se misturar:

- **Merge Sort sequencial**: um processo ordena todos os vetores, um de cada vez.
- **MPI**: `P` processos independentes, cada um com sua memória, recebem lotes de vetores e
  ordenam cada vetor com **o mesmo** Merge Sort. O paralelismo está **entre os vetores**: cada
  vetor é ordenado inteiro por um único processo.
- **Numba**: só compila o Merge Sort para código de máquina. Ele deixa as duas versões mais
  rápidas, mas **não** é paralelismo. Não usamos threads, `prange` nem `multiprocessing`.

## O que tem no repositório

| arquivo | para que serve |
|---|---|
| `merge_mpi.py` | programa do hackathon: mede sequencial x MPI, valida e grava o CSV |
| `vetores.py` | cenários, gerador de vetores, divisão entre processos e validação (não usa MPI) |
| `merge_numba.py` | o Merge Sort compilado com `@njit`, usado pelos dois programas |
| `merge_seq.py` | leitura do CSV e Merge Sort em Python puro, para comparação |
| `tests/` | testes automatizados com `pytest` |
| `resultados.csv` | medições que fizemos na nossa máquina |

## Instalação

Você vai precisar de:

- Python 3.10 ou mais novo (testamos com o 3.14.6);
- as bibliotecas do `requirements.txt`: NumPy, Numba, mpi4py e pytest;
- um MPI que forneça o comando `mpiexec` (testamos com o Intel MPI 2021.18 no Windows).

Tudo fica num ambiente virtual dentro da pasta (`.venv`), sem mexer no Python do sistema.

### Windows (PowerShell)

```powershell
python -m venv .venv
```
```powershell
.venv\Scripts\Activate.ps1
```
```powershell
pip install -r requirements.txt
```

Se a ativação reclamar que "a execução de scripts foi desabilitada", libere só para a janela
atual e tente de novo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Para ter o `mpiexec`, você pode escolher:

- instalar o [Microsoft MPI](https://learn.microsoft.com/message-passing-interface/microsoft-mpi), que coloca o `mpiexec` no PATH do sistema; ou
- instalar o Intel MPI **dentro do venv** (foi o que usamos). Ele fica em `.venv\Library\bin`, que
  precisa entrar no PATH de cada janela nova:

```powershell
pip install impi_rt
```
```powershell
$env:PATH = "$PWD\.venv\Library\bin;$env:PATH"
```

### Linux / macOS

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

O `mpiexec` vem do Open MPI ou do MPICH: `sudo apt install openmpi-bin` no Debian/Ubuntu ou
`brew install open-mpi` no macOS. Sem acesso de administrador, `pip install mpich` instala o
MPICH dentro do venv.

> Nós só testamos no Windows. Os comandos para Linux e macOS são os mesmos, mas ainda não
> rodamos nesses sistemas.

### Funcionou?

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)"
```

Você deve ver `0 2` e `1 2`, em qualquer ordem.

## Como rodar

Tudo roda em uma máquina só: o `mpiexec -n P` abre `P` processos locais.

Ver todas as opções:
```bash
python merge_mpi.py --help
```

Rodar com 1 processo, sem `mpiexec`:
```bash
python merge_mpi.py --cenario bordas
```

Rodar todos os cenários com 4 processos:
```bash
mpiexec -n 4 python -m mpi4py merge_mpi.py
```

Escolher cenário, quantidade de vetores e semente:
```bash
mpiexec -n 2 python -m mpi4py merge_mpi.py --cenario variados --vetores 50 --semente 7
```

Ver o que acontece quando há mais processos do que vetores:
```bash
mpiexec -n 8 python -m mpi4py merge_mpi.py --cenario grandes --vetores 3
```

Usamos `python -m mpi4py` porque, se um processo der erro, ele encerra todos os outros. Sem isso,
os demais podem ficar esperando uma mensagem que nunca chega.

### Opções

| opção | padrão | o que faz |
|---|---|---|
| `--cenario` | `todos` | `pequenos`, `grandes`, `variados`, `bordas` ou `todos` |
| `--semente` | 42 | a mesma semente gera os mesmos vetores |
| `--repeticoes` | 3 | quantas vezes medir cada versão; o speedup usa o **menor** tempo |
| `--vetores` | do cenário | quantidade de vetores |
| `--tam-min`, `--tam-max` | do cenário | tamanho mínimo e máximo de cada vetor (0 permite vetores vazios) |
| `--vmin`, `--vmax` | do cenário | intervalo dos valores (dentro de `int32`, negativos incluídos) |
| `--csv` | `resultados.csv` | arquivo onde as linhas são **acrescentadas** |

Se algum vetor sair errado, o programa mostra quais foram e termina com código **1**. Se uma
opção for inválida, ele explica o problema e termina com código **2**.

## Cenários

| cenário | vetores | tamanho de cada um | valores | o que queremos ver |
|---|---|---|---|---|
| `pequenos` | 5000 | 0 a 2.000 | ±1.000.000 | muitas mensagens com pouco trabalho em cada uma |
| `grandes` | 8 | 1 a 2 milhões | ±1.000.000 | poucos vetores grandes, difíceis de equilibrar |
| `variados` | 100 | 0 a 400.000 | ±1.000.000 | tamanhos bem diferentes entre si |
| `bordas` | 200 | 0 a 3 | −2 a 2 | vetores vazios, unitários, repetidos e negativos |

Tamanhos e valores são sorteados de forma uniforme com `numpy.random.default_rng(semente)`. Os
vetores são `int32`, o tipo que o Merge Sort em Numba usa. Todos os cenários têm valores repetidos
e negativos.

## O que entra em cada tempo

Gerar os vetores, decidir quem ordena o quê e compilar o Numba acontece **antes** das medições.

- **Tempo sequencial** (`tempo_seq_s`): o processo 0 copia e ordena cada vetor, um de cada vez.
- **Tempo MPI** (`tempo_par_s`): começa depois que todos os processos se sincronizam e termina
  quando o processo 0 tem todos os vetores de volta, na ordem original. Inclui:
  - montar os lotes e serializá-los;
  - enviar (`scatter`) e receber de volta (`gather`);
  - ordenar em cada processo;
  - esperar o processo mais lento;
  - recolocar cada vetor no lugar.

A conferência dos resultados fica fora dos dois tempos.

## Como dividimos o trabalho

A ideia é simples: **o maior vetor primeiro, para quem está mais livre.**

1. Ordenamos os vetores do maior para o menor.
2. Cada vetor vai para o processo com **menos elementos** acumulados até ali.
3. Em caso de empate, ganha quem tem menos vetores. Assim os vetores vazios também se espalham.

Cada vetor viaja com seu índice original. O processo 0 manda um lote para cada processo
(inclusive para si mesmo), recebe pares `(índice, vetor ordenado)` e devolve tudo para a posição
de origem. Se houver mais processos do que vetores, os que sobram recebem um lote vazio, e o
programa avisa quantos ficaram sem trabalho.

## O CSV de resultados

Cada execução acrescenta uma linha por cenário em `resultados.csv`:

| coluna | o que guarda |
|---|---|
| `data_hora` | quando a linha foi gravada |
| `cenario`, `semente` | cenário e semente usados |
| `processos`, `processos_ociosos` | processos MPI e quantos ficaram sem vetores |
| `vetores`, `total_elementos` | quantidade de vetores e soma dos tamanhos |
| `tam_min`, `tam_max`, `vmin`, `vmax` | parâmetros do gerador |
| `repeticoes` | quantas vezes cada versão foi medida |
| `tempo_seq_s`, `tempo_par_s` | menor tempo sequencial e menor tempo MPI, em segundos |
| `tempo_seq_medio_s`, `tempo_par_medio_s` | média das repetições |
| `speedup` | `tempo_seq_s / tempo_par_s` |
| `eficiencia` | `speedup / processos` |
| `desequilibrio` | maior carga dividida pela carga média (1,0 = todos com o mesmo tanto) |
| `validacao` | `ok` ou `falhou` |

## Como ler os números

- **Speedup maior que 1**: o MPI terminou antes do sequencial. O máximo teórico é o número de
  processos.
- **Eficiência**: quanto cada processo rendeu (1,0 seria o ideal). Ela cai quando a comunicação
  pesa, quando a carga fica desigual ou quando há mais processos do que núcleos.
- **Desequilíbrio**: no cenário `grandes` com 8 processos, cada um recebe um vetor, e quem pegou o
  maior define o tempo de todos. Dividindo vetores inteiros, não tem como evitar isso.
- **MPI mais lento que o sequencial?** Acontece, e é esperado em cargas pequenas. Cada processo é
  um programa separado, e os vetores precisam ser empacotados, enviados e devolvidos. Quando
  ordenar leva menos tempo que esse transporte, o speedup fica abaixo de 1. É o que vemos no
  cenário `bordas` e em qualquer cenário com 1 processo.
- **Seus números serão outros.** O resultado depende do computador, do número de processos e do
  que mais estiver rodando. Processadores com núcleos de tipos diferentes (desempenho e eficiência)
  e o modo de energia mudam bastante os tempos. Repita as medições e compare execuções feitas nas
  mesmas condições.

### Uma execução nossa

Com 4 processos, em um Windows 11 com Intel Core Ultra 7 255H (16 núcleos) e Intel MPI 2021.18:

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

> **Acentos estranhos no Windows?** Ao rodar pelo `mpiexec`, você pode ver "cenßrio" em vez de
> "cenário". É só a exibição no console; o CSV é gravado certo. Para corrigir na janela atual:
>
> ```powershell
> $env:PYTHONIOENCODING = "utf-8"; [Console]::OutputEncoding = [Text.Encoding]::UTF8
> ```

## Testes

Com o venv ativado e o `mpiexec` no PATH:

```bash
python -m pytest
```

| arquivo | o que confere |
|---|---|
| `tests/test_ordenacao.py` | o Merge Sort com vetor vazio, unitário, repetidos, negativos, extremos de `int32` e tamanhos perto dos blocos de 16 |
| `tests/test_vetores.py` | gerador (semente, limites, tamanhos desiguais, parâmetros inválidos), divisão entre processos e validação |
| `tests/test_mpi.py` | ordem original, execução com 1 processo, falha de validação com código 1, opções inválidas e, **se houver `mpiexec`**, execuções reais com 2 processos e com mais processos do que vetores |

Se o `mpiexec` não estiver no PATH, os dois testes com vários processos aparecem como `SKIPPED`.
Sem `mpi4py`, o `test_mpi.py` inteiro é pulado. O resto roda normalmente.

## Limitações conhecidas

- Paralelizamos **entre** vetores. Um vetor enorme continua sendo ordenado por um processo só.
- A divisão usa o tamanho `n` como peso, não `n log n`, e ignora o custo de enviar os dados.
- `scatter` e `gather` com pickle são simples de usar, mas copiam e serializam tudo. Versões com
  buffer (`Scatterv`/`Gatherv`) seriam mais rápidas, só que exigem calcular deslocamentos.
- O processo 0 guarda todos os vetores na memória: a entrada, o resultado sequencial e o
  resultado MPI.
- Só aceitamos vetores `int32`, como o Merge Sort em Numba.
- Testamos em uma máquina com Windows. Ainda não rodamos em cluster, Linux ou macOS.

Achou um problema ou tem uma ideia? Abra uma issue, ficaremos felizes em conversar.

---

## Merge Sort sequencial no dados.csv

Aqui a pergunta é outra: **quanto o Numba aproxima o Python do C** em um único vetor de 1.000.000
de inteiros (o `dados.csv` dos professores, com os números separados por `;`).

O `dados.csv` **não está no repositório**. Coloque o arquivo nesta pasta ou passe o caminho dele:

```bash
python merge_numba.py [dados.csv] [repeticoes] [--python]
```

| argumento | padrão | o que faz |
|---|---|---|
| `dados.csv` | `dados.csv` | caminho do arquivo dos professores |
| `repeticoes` | 5 | execuções por entrada; mostramos o **menor** tempo e a média |
| `--python` | — | mede também o Merge Sort em Python puro (bem mais lento) |

Medimos quatro entradas, todas com os mesmos valores:

- **aleatória**: o próprio `dados.csv`;
- **ordenada**: os valores já em ordem crescente;
- **invertida**: os valores em ordem decrescente;
- **pior caso**: uma ordem que obriga cada intercalação a comparar até o último elemento.

O vetor ordenado é salvo em `dados_ordenados_numba.csv`.

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

Na primeira vez, o Numba leva cerca de 4 s para compilar (fora da medição). Depois, o código
compilado fica em cache no `__pycache__` e o programa começa em menos de 1 s. Com `--python`, as
medições em Python puro rodam intercaladas com as do Numba e deixam os tempos do Numba mais lentos
e instáveis.

### Numba x C x Python puro

Mesmo `dados.csv` e mesmo algoritmo (bottom-up) no C e no Numba. Cada versão rodou sozinha, e
anotamos o menor tempo de 7 execuções:

| entrada | C (`gcc -O2`, WSL) | **Numba** | Python puro |
|---|---|---|---|
| aleatória | 0,053 s | **0,050 s** | 3,68 s |
| ordenada | 0,008 s | **0,025 s** | 0,60 s |
| invertida | 0,012 s | **0,029 s** | 0,63 s |
| pior caso | 0,047 s | **0,049 s** | 3,88 s |

- Na entrada **aleatória** e no **pior caso**, em que o trabalho de verdade é intercalar, o Numba
  **empata com o C** e é cerca de 75 vezes mais rápido que o Python puro.
- Nas entradas **ordenada** e **invertida**, quase tudo vira cópia de blocos. O C usa `memcpy` e
  fica 2 a 3 vezes à frente, numa diferença de centésimos de segundo.
- As saídas do C, do Numba e do Python puro são idênticas, byte a byte.

### Por dentro do Merge Sort

O `@njit` compila as três funções principais (`insercao`, `intercalar` e `merge_sort`) para
código de máquina, sem passar pelo interpretador. Para chegar perto do C, escrevemos o código do
jeito que escreveríamos em C:

1. **arrays NumPy `int32` contíguos**, como um `int[]`, em vez de listas de objetos;
2. **bottom-up, sem recursão**, com dois buffers que se alternam: cada passada lê de um e escreve
   no outro, sem copiar de volta;
3. **intercalação sem `if` na escolha do próximo elemento** (o compilador gera `cmov`), o que
   ajuda quando essa escolha é imprevisível;
4. **blocos de 16 elementos ordenados por inserção** antes de começar a intercalar;
5. **atalhos** para pares de blocos que já estão em ordem ou em ordem inversa (viram cópias);
6. **compilação antes da medição**, com uma chamada de aquecimento.

O **pior caso** foi montado para as divisões exatas do bottom-up. Em cada intercalação, o maior e
o segundo maior valor ficam em lados diferentes, o que força L + R − 1 comparações (o máximo). Os
blocos de 16 ficam em ordem inversa, que é o pior caso da inserção. Assim, nenhum atalho entra em
ação.

### Como sabemos que está certo

Cada resultado é comparado, elemento por elemento, com o `numpy.sort`. Na leitura, conferimos se
todos os valores cabem em `int32` e avisamos com uma mensagem clara se o arquivo não existir.
