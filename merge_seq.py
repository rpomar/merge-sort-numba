"""merge_seq.py - Merge Sort sequencial em Python, ajustado para o pior caso.

Uso:  python merge_seq.py [dados.csv] [repeticoes] [--casos]

  sem --casos: ordena o arquivo e grava dados_ordenados_py.csv
  com --casos: mede aleatória (o arquivo), ordenada, invertida e PIOR CASO

Também é o módulo base de merge_sort.py, merge_numba.py e analise_merge.py.

Em Python pesa o número de operações do INTERPRETADOR, então as escolhas
diferem das de merge_seq.c (bottom-up e inserção ficaram mais lentos aqui):
  1. top-down com fatiamento: as fatias são copiadas em C;
  2. intercalação com append/extend, testando o fim de UMA lista por volta;
  3. caso base com 2 elementos: corta o nível mais numeroso da recursão;
  4. atalhos O(1) para listas já em ordem ou em ordem inversa.
"""
import sys
from time import perf_counter

CAMINHO_PADRAO = "dados.csv"
SAIDA = "dados_ordenados_py.csv"


def ler_vetor(caminho=CAMINHO_PADRAO):
    with open(caminho, encoding="utf-8-sig") as f:   # -sig: tolera BOM
        texto = f.read()
    for sep in ",\r\n":                               # aceita ; , e quebras
        texto = texto.replace(sep, ";")
    return [int(v) for v in texto.split(";") if v.strip()]


def salvar_vetor(vetor, caminho=SAIDA):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(";".join(map(str, vetor)))


def intercalar(a, b):
    """Intercala duas listas ordenadas e não vazias."""
    if a[-1] <= b[0]:            # atalho: já em ordem
        return a + b
    if b[-1] < a[0]:             # atalho: ordem inversa
        return b + a
    res = []
    adiciona = res.append        # evita buscar o método a cada volta
    i = j = 0
    na, nb = len(a), len(b)
    x, y = a[0], b[0]
    while True:
        if x <= y:               # '<=' mantém a estabilidade
            adiciona(x)
            i += 1
            if i == na:          # 'a' acabou: o resto de 'b' vai direto
                res.extend(b[j:])
                return res
            x = a[i]
        else:
            adiciona(y)
            j += 1
            if j == nb:          # 'b' acabou: o resto de 'a' vai direto
                res.extend(a[i:])
                return res
            y = b[j]


def merge_sort(vetor):
    n = len(vetor)
    if n <= 2:
        if n == 2 and vetor[0] > vetor[1]:
            return [vetor[1], vetor[0]]
        return list(vetor)
    meio = n // 2
    return intercalar(merge_sort(vetor[:meio]), merge_sort(vetor[meio:]))


# --------------------------------------------------------------------------
# Pior caso e ferramentas de medição
# --------------------------------------------------------------------------
def pior_caso(ordenado):
    """Entrada que faz merge_sort comparar o máximo possível.

    Uma intercalação de L + R elementos faz L + R - 1 comparações quando o
    maior e o segundo maior ficam em lados diferentes. Alternar os valores
    entre esquerda e direita garante isso em todos os níveis, e nenhum
    atalho dispara. As divisões são as de merge_sort (meio = n // 2).
    """
    n = len(ordenado)
    if n <= 2:
        return ordenado[::-1]
    esq, dir_ = ordenado[0::2], ordenado[1::2]
    if len(esq) > n // 2:        # n ímpar: a esquerda é a metade menor
        esq, dir_ = dir_, esq
    return pior_caso(esq) + pior_caso(dir_)


def como_lido(vetor):
    """Recria os inteiros na ordem da lista, como se viessem do arquivo.

    A lista guarda PONTEIROS para objetos int: reordená-la espalha os objetos
    pela memória e cada comparação vira falta de cache (~2,5x mais lento),
    o que não tem relação com o algoritmo.
    """
    return [int(str(x)) for x in vetor]


def comparacoes(vetor):
    """Conta as comparações do PRÓPRIO merge_sort com um int instrumentado."""
    total = [0]

    class Int(int):
        pass

    for op in ("__lt__", "__le__", "__gt__"):
        def contar(a, b, _op=getattr(int, op)):
            total[0] += 1
            return _op(a, b)
        setattr(Int, op, contar)

    merge_sort([Int(x) for x in vetor])
    return total[0]


def medir(nome, entrada, referencia, reps):
    """Imprime o melhor e o tempo médio de 'reps' execuções e a validação."""
    tempos = []
    for _ in range(reps):
        t0 = perf_counter()
        res = merge_sort(entrada)
        tempos.append(perf_counter() - t0)
    ok = "sim" if res == referencia else "NÃO"
    print(f"{nome:<12} {len(entrada):>10,} {min(tempos):>10.4f} "
          f"{sum(tempos) / reps:>10.4f} {ok:>7}")
    return res


def main():
    args = [a for a in sys.argv[1:] if a != "--casos"]
    caminho = args[0] if args else CAMINHO_PADRAO
    reps = max(1, int(args[1])) if len(args) > 1 else 3

    t0 = perf_counter()
    dados = ler_vetor(caminho)
    print(f"{len(dados):,} valores lidos em {perf_counter() - t0:.3f}s\n")
    ordenado = sorted(dados)     # gabarito (Timsort nativo)

    print(f"{'entrada':<12} {'n':>10} {'melhor(s)':>10} {'media(s)':>10} "
          f"{'válido':>7}\n" + "-" * 55)
    res = medir("aleatória", dados, ordenado, reps)
    if "--casos" not in sys.argv:
        salvar_vetor(res)
        print(f"\nvetor ordenado gravado em {SAIDA}")
        return

    for nome, v in (("ordenada", ordenado), ("invertida", ordenado[::-1]),
                    ("PIOR CASO", pior_caso(ordenado))):
        medir(nome, como_lido(v), ordenado, reps)

    m = 10_000                   # prova do pior caso numa amostra menor
    print(f"\ncomparações com n = {m:,}: "
          f"aleatória {comparacoes(dados[:m]):,} | "
          f"pior caso {comparacoes(pior_caso(sorted(dados[:m]))):,}")


if __name__ == "__main__":
    main()
