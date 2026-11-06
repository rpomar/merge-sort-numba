# Merge Sort com Numba e MPI

Trabalho da disciplina de Programação Paralela e Distribuída. O programa ordena vários vetores aleatórios de tamanhos variados, primeiro em um único processo e depois repartindo os vetores entre processos MPI. No fim, compara os resultados e os tempos das duas execuções.

O Merge Sort é escrito em Python e compilado com Numba. As duas execuções usam o mesmo código compilado, e o paralelismo vem só do MPI: cada processo ordena vetores inteiros, sem threads.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `merge_mpi.py` | execução sequencial, execução MPI, validação e gravação do CSV |
| `vetores.py` | cenários, gerador, distribuição dos vetores e comparação de resultados |
| `merge_numba.py` | Merge Sort compilado com `@njit` e medição no `dados.csv` |
| `merge_seq.py` | leitura do CSV e Merge Sort em Python puro |
| `tests/` | testes com `pytest` |
| `resultados.csv` | medições feitas em uma máquina com Windows |

## Instalação

São necessários Python 3.10 ou mais novo, as bibliotecas do `requirements.txt` e uma implementação de MPI com o comando `mpiexec`. Os testes foram feitos com Python 3.14.6 e Intel MPI 2021.18, no Windows.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install impi_rt
$env:PATH = "$PWD\.venv\Library\bin;$env:PATH"
```

O `impi_rt` instala o Intel MPI dentro do ambiente virtual. O Microsoft MPI também fornece o `mpiexec`, mas não foi testado. Se o PowerShell bloquear a ativação, rode antes `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

Linux ou macOS (não testado):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Linux e no macOS, o `mpiexec` vem do Open MPI ou do MPICH: `sudo apt install openmpi-bin`, `brew install open-mpi` ou `pip install mpich`.

## Execução

```bash
python merge_mpi.py --help
python merge_mpi.py --cenario bordas
mpiexec -n 4 python -m mpi4py merge_mpi.py
mpiexec -n 2 python -m mpi4py merge_mpi.py --cenario variados --vetores 50 --semente 7
mpiexec -n 8 python -m mpi4py merge_mpi.py --cenario grandes --vetores 3
```

O `mpiexec -n P` abre `P` processos na mesma máquina. Com `-m mpi4py`, um erro em um processo encerra os demais.

| opção | padrão | significado |
|---|---|---|
| `--cenario` | `todos` | `pequenos`, `grandes`, `variados`, `bordas` ou `todos` |
| `--semente` | 42 | semente do gerador |
| `--repeticoes` | 3 | medições de cada versão |
| `--vetores` | do cenário | quantidade de vetores |
| `--tam-min`, `--tam-max` | do cenário | tamanho mínimo e máximo (0 permite vetores vazios) |
| `--vmin`, `--vmax` | do cenário | intervalo dos valores, dentro de `int32` |
| `--csv` | `resultados.csv` | arquivo em que as linhas são acrescentadas |

O programa termina com código 1 se a validação falhar e com código 2 se uma opção for inválida.

No Windows, os acentos podem sair errados na saída do `mpiexec`. Isso não afeta o CSV. Para ajustar a janela atual:

```powershell
$env:PYTHONIOENCODING = "utf-8"; [Console]::OutputEncoding = [Text.Encoding]::UTF8
```

## Cenários

| cenário | vetores | tamanho | valores |
|---|---|---|---|
| `pequenos` | 5000 | 0 a 2.000 | -1.000.000 a 1.000.000 |
| `grandes` | 8 | 1.000.000 a 2.000.000 | -1.000.000 a 1.000.000 |
| `variados` | 100 | 0 a 400.000 | -1.000.000 a 1.000.000 |
| `bordas` | 200 | 0 a 3 | -2 a 2 |

Tamanhos e valores são sorteados com distribuição uniforme. O cenário `bordas` cobre vetores vazios, unitários, com repetidos e com negativos.

## Funções principais

| função | arquivo | o que faz |
|---|---|---|
| `insercao` | `merge_numba.py` | ordena um bloco de até 16 elementos por inserção |
| `intercalar` | `merge_numba.py` | intercala dois blocos ordenados |
| `merge_sort` | `merge_numba.py` | ordena um vetor `int32` no lugar, usando um vetor auxiliar |
| `gerar_vetores` | `vetores.py` | gera os vetores a partir da semente |
| `distribuir` | `vetores.py` | define quais vetores cada processo ordena |
| `desequilibrio` | `vetores.py` | calcula a maior carga dividida pela carga média |
| `validar` | `vetores.py` | devolve os índices dos vetores que diferem do esperado |
| `ordenar_lote` | `merge_mpi.py` | ordena uma cópia de cada vetor de um lote e mantém os índices |
| `sequencial` | `merge_mpi.py` | ordena todos os vetores em um processo e mede o tempo |
| `paralelo` | `merge_mpi.py` | envia os lotes, ordena, coleta os resultados e mede o tempo |
| `executar` | `merge_mpi.py` | roda um cenário, valida e monta a linha do CSV |
| `gravar_csv` | `merge_mpi.py` | acrescenta uma linha ao arquivo de resultados |

O `merge_sort` é bottom-up e não usa recursão. Primeiro ordena blocos de 16 elementos por inserção. Depois intercala blocos cada vez maiores, alternando entre o vetor original e o auxiliar. Pares de blocos já em ordem ou em ordem inversa são apenas copiados.

## Execução com MPI

1. O processo 0 gera os vetores e calcula a distribuição.
2. O processo 0 ordena todos os vetores sozinho, para obter a referência sequencial.
3. Todos os processos se sincronizam com `Barrier`.
4. O `scatter` envia um lote de pares `(índice, vetor)` a cada processo, inclusive ao processo 0.
5. Cada processo ordena o seu lote com `ordenar_lote`.
6. O `gather` traz os lotes de volta ao processo 0, que usa os índices para restaurar a ordem original.
7. O resultado sequencial é comparado com `numpy.sort`, e o resultado MPI é comparado com o sequencial.

A distribuição percorre os vetores do maior para o menor e entrega cada um ao processo com menos elementos acumulados. No empate, escolhe o processo com menos vetores. Se houver mais processos do que vetores, os processos excedentes recebem lotes vazios e o programa informa quantos ficaram sem trabalho.

## Medições

A geração dos vetores, o cálculo da distribuição, a compilação do Numba e a validação ficam fora dos tempos.

O tempo sequencial inclui a cópia e a ordenação de cada vetor no processo 0. O tempo MPI começa depois da barreira e termina com todos os vetores de volta na ordem original, incluindo a serialização, a comunicação e a espera pelo processo mais lento.

| métrica | cálculo |
|---|---|
| speedup | menor tempo sequencial / menor tempo MPI |
| eficiência | speedup / número de processos |
| desequilíbrio | maior carga / carga média, em elementos |

Speedup abaixo de 1 significa que o MPI foi mais lento. Isso é esperado em cargas pequenas, em que enviar e receber os vetores custa mais do que ordená-los, e também com 1 processo. Os tempos mudam com o computador, o número de processos e a carga da máquina.

Cada execução acrescenta ao CSV uma linha por cenário, com data, cenário, semente, processos, processos ociosos, vetores, total de elementos, parâmetros do gerador, repetições, menores tempos, tempos médios, speedup, eficiência, desequilíbrio e validação.

Exemplo com 4 processos, em um Intel Core Ultra 7 255H:

```
cenário   vetores   elementos   seq(s)   mpi(s) speedup  efic. deseq. validação
-------------------------------------------------------------------------------
pequenos    5,000   4,986,460   0.1451   0.1077   1.348  0.337  1.000        ok
grandes         8  12,031,583   0.6243   0.2694   2.318  0.579  1.041        ok
variados      100  21,069,636   0.9495   0.4275   2.221  0.555  1.001        ok
bordas        200         306   0.0001   0.0005   0.186  0.047  1.007        ok
```

## Testes

```bash
python -m pytest
```

Os testes cobrem a ordenação, o gerador, a distribuição, a validação e a execução com 1 processo. Os dois testes com vários processos exigem `mpiexec` no PATH e são pulados sem ele. Sem `mpi4py`, todo o `tests/test_mpi.py` é pulado.

## Limitações

- Cada vetor é ordenado por um único processo, então um vetor muito grande limita o tempo total.
- A distribuição usa o tamanho do vetor como peso e ignora o custo de comunicação.
- `scatter` e `gather` com pickle serializam e copiam todos os dados.
- O processo 0 mantém na memória os vetores de entrada e os dois resultados.
- O código só aceita vetores `int32`.
- Só foi testado em uma máquina com Windows.

## Merge Sort no dados.csv

O `merge_numba.py` ordena o vetor de 1.000.000 de inteiros do `dados.csv` fornecido pelos professores. Esse arquivo não está no repositório.

```bash
python merge_numba.py [dados.csv] [repeticoes] [--python]
```

O programa mede quatro entradas com os mesmos valores (aleatória, ordenada, invertida e pior caso), confere cada resultado com `numpy.sort` e grava o vetor ordenado em `dados_ordenados_numba.csv`. A opção `--python` mede também o Merge Sort em Python puro.

Menor tempo de 7 execuções, com o mesmo algoritmo no C e no Numba:

| entrada | C (`gcc -O2`, WSL) | Numba | Python puro |
|---|---|---|---|
| aleatória | 0,053 s | 0,050 s | 3,68 s |
| ordenada | 0,008 s | 0,025 s | 0,60 s |
| invertida | 0,012 s | 0,029 s | 0,63 s |
| pior caso | 0,047 s | 0,049 s | 3,88 s |
