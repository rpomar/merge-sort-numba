# Merge Sort sequencial com Numba

Merge Sort **sequencial** em Python compilado com [Numba](https://numba.pydata.org/), aplicado ao
vetor do `dados.csv` fornecido pelos professores (1.000.000 de inteiros em uma única linha,
separados por `;`). O programa mede quatro entradas montadas com esses mesmos valores e,
opcionalmente, compara com o Merge Sort em Python puro.

Trabalho da disciplina de Programação Paralela e Distribuída.

## Arquivos

| arquivo | função |
|---|---|
| `merge_numba.py` | **programa principal**: núcleo do Merge Sort compilado com `@njit`, construção do pior caso, medição e relatório |
| `merge_seq.py` | leitura do CSV e Merge Sort em Python puro (referência para `--python`) |
| `requirements.txt` | dependências (`numpy`, `numba`) |

O `dados.csv` **não está no repositório**: coloque o arquivo dos professores nesta pasta ou passe
o caminho dele como argumento.

## Requisitos

- Python 3.10 ou mais novo (testado com Python 3.14)
- NumPy e Numba (testado com NumPy 2.5.3 e Numba 0.68.0)

```bash
pip install -r requirements.txt
```

## Como executar

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

## Exemplo de saída

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
código compilado do cache (`__pycache__`) e começam em menos de 1 s. Os tempos variam com a
máquina e a carga do sistema: para comparar versões, rode cada uma sozinha e com mais
repetições. Com `--python`, as medições do Python puro rodam entre as do Numba e deixam os
tempos do Numba mais lentos e instáveis.

## Comparação com C e Python puro

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

## Como funciona

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

## Validação

Cada resultado é comparado elemento a elemento com `numpy.sort`. A leitura confere se todos os
valores cabem em `int32` e avisa com uma mensagem clara se o arquivo não existir.
