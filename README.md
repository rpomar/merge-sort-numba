# Merge Sort sequencial com Numba

Merge Sort **sequencial** em Python compilado com [Numba](https://numba.pydata.org/), aplicado a
200 vetores aleatórios de tamanhos variados (distribuição de Pareto: muitos vetores pequenos e
poucos gigantes). A mesma execução mede o Merge Sort em Python puro, para comparar.

Trabalho da disciplina de Programação Paralela e Distribuída.

## Arquivos

| arquivo | função |
|---|---|
| `merge_numba.py` | **programa principal**: núcleo do Merge Sort compilado com `@njit`, medição e relatório |
| `merge_seq.py` | Merge Sort em Python puro (referência para a comparação com `--python`) |
| `ordenacao_paralela.py` | gerador dos vetores (seed fixa 42, reprodutível); o `merge_numba.py` usa `gerar_vetores` e `salvar` |
| `requirements.txt` | dependências (`numpy`, `numba`) |

O arquivo `vetores.txt` (~60 MB, um vetor por linha) é **gerado na primeira execução** e
reaproveitado nas seguintes; por isso ele não está no repositório.

## Requisitos

- Python 3.10 ou mais novo (testado com Python 3.14)
- NumPy e Numba (testado com NumPy 2.5.3 e Numba 0.68.0)

```bash
pip install -r requirements.txt
```

## Como executar

```bash
python merge_numba.py [qtd_vetores] [repeticoes] [--python]
```

| argumento | padrão | significado |
|---|---|---|
| `qtd_vetores` | 200 | quantos vetores gerar/ler |
| `repeticoes` | 3 | execuções por vetor; vale o **menor** tempo |
| `--python` | — | também mede o Merge Sort em Python puro (bem mais lento) |

Exemplo:

```bash
python merge_numba.py 200 3 --python
```

## Exemplo de saída

```
200 vetores | 6,324,066 elementos | menor=5,133 maior=565,145
compilação/carga do Numba: 4.12s

faixa (n)          vetores   elementos   numba(s)  python(s)   ganho
0 - 9,999               99     683,283     0.0472     0.8239   17.5x
10,000 - 49,999         81   1,693,999     0.1210     2.1793   18.0x
50,000 - 199,999        14   1,532,767     0.1317     2.5410   19.3x
>= 200,000               6   2,414,017     0.2084     4.6523   22.3x
TOTAL                  200   6,324,066     0.5084    10.1965   20.1x
numba: 80.4 ns por elemento | válido: sim
python: 1612.3 ns por elemento | válido: sim
```

Os tempos dependem da máquina e da carga do sistema; o que importa são as proporções.
A primeira execução compila o código (alguns segundos, fora da medição); as seguintes carregam
o código compilado do cache (`__pycache__`) e começam em menos de 1 s.

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
6. **um único buffer auxiliar**, do tamanho do maior vetor, reaproveitado por todos;
7. a **compilação é feita antes da medição**, com uma chamada de aquecimento.

## Validação

Cada vetor ordenado é comparado elemento a elemento com `numpy.sort`. A linha `válido: sim` só
aparece se **todos** os 200 vetores estiverem corretos, nas duas versões.
