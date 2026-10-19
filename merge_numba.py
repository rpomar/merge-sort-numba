"""merge_numba.py - Merge Sort SEQUENCIAL com Numba sobre o dados.csv.

Uso:  python merge_numba.py [dados.csv] [repeticoes] [--python]

  Mede 4 entradas montadas com os valores do arquivo: aleatória (o próprio
  arquivo), ordenada, invertida e PIOR CASO. Grava a saída ordenada em
  dados_ordenados_numba.csv.
  --python  também mede merge_seq.merge_sort (Python puro, lento)
"""
import sys
from time import perf_counter

import numpy as np
from numba import njit

import merge_seq

RUN = 16                 # tamanho dos blocos ordenados por inserção
SAIDA = "dados_ordenados_numba.csv"


# --------------------------------------------------------------------------
# Núcleo compilado
# --------------------------------------------------------------------------
@njit(cache=True)
def insercao(v, ini, fim):                 # intervalo [ini, fim)
    for i in range(ini + 1, fim):
        x = v[i]
        j = i - 1
        while j >= ini and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x


@njit(cache=True)
def intercalar(src, dst, ini, meio, fim):
    """Intercala src[ini,meio) com src[meio,fim) em dst[ini,fim)."""
    if src[meio - 1] <= src[meio]:         # atalho: já em ordem
        dst[ini:fim] = src[ini:fim]
        return
    if src[fim - 1] < src[ini]:            # atalho: ordem inversa
        dst[ini:ini + fim - meio] = src[meio:fim]
        dst[ini + fim - meio:fim] = src[ini:meio]
        return
    i, j, k = ini, meio, ini
    while i < meio and j < fim:
        a = src[i]
        b = src[j]
        pega_a = 1 if a <= b else 0        # '<=' mantém a estabilidade
        dst[k] = a if pega_a else b        # seleção sem desvio (cmov)
        i += pega_a
        j += 1 - pega_a
        k += 1
    dst[k:k + meio - i] = src[i:meio]      # sobra (só um lado tem)
    dst[k + meio - i:fim] = src[j:fim]


@njit(cache=True)
def merge_sort(v, aux):
    """Ordena v no lugar. aux precisa ter pelo menos len(v) elementos."""
    n = v.shape[0]
    for ini in range(0, n, RUN):
        insercao(v, ini, min(ini + RUN, n))
    src, dst = v, aux                      # aux é só indexado em [0, n)
    no_aux = False                         # o resultado está em aux?
    largura = RUN
    while largura < n:
        for ini in range(0, n, 2 * largura):
            meio = min(ini + largura, n)
            fim = min(ini + 2 * largura, n)
            if meio >= fim:                # bloco sem par: só copia
                dst[ini:fim] = src[ini:fim]
            else:
                intercalar(src, dst, ini, meio, fim)
        src, dst = dst, src                # troca os papéis
        no_aux = not no_aux
        largura *= 2
    if no_aux:
        v[:] = aux[:n]


# --------------------------------------------------------------------------
# Dados de entrada
# --------------------------------------------------------------------------
def ler(caminho):
    """dados.csv -> array int32 (mesma leitura de merge_seq.py)."""
    try:
        valores = merge_seq.ler_vetor(caminho)
    except FileNotFoundError:
        sys.exit(f"arquivo não encontrado: {caminho}\n"
                 "coloque o dados.csv dos professores nesta pasta ou passe "
                 "o caminho como argumento")
    if not valores:
        sys.exit(f"nenhum número encontrado em {caminho}")
    limite = np.iinfo(np.int32)
    if min(valores) < limite.min or max(valores) > limite.max:
        sys.exit("há valores fora do intervalo de int32")
    return np.array(valores, dtype=np.int32)


def pior_caso(ordenado):
    """Entrada que faz ESTE merge_sort (bottom-up) comparar o máximo.

    Mesma construção de merge_seq.c: em cada intercalação o maior e o
    segundo maior ficam em lados diferentes (L + R - 1 comparações) e os
    blocos de RUN ficam invertidos (pior caso da inserção). As divisões são
    as do bottom-up (meio = ini + largura), então nenhum atalho dispara.
    """
    v = ordenado.copy()

    def montar(ini, n, largura):           # v[ini:ini+n] está crescente
        seg = v[ini:ini + n]
        if n <= RUN:
            seg[:] = seg[::-1].copy()
            return
        if n <= largura:                   # sem par neste nível
            montar(ini, n, largura // 2)
            return
        L, R = largura, n - largura        # R <= L
        # os L - R menores já estão no início da esquerda; os 2R maiores
        # alternam de lado, começando pelo maior na esquerda
        topo = seg[L - R:].copy()
        seg[L - R:L] = topo[1::2]          # maior, 3º maior, ... -> esquerda
        seg[L:] = topo[0::2]               # 2º maior, 4º maior, ... -> direita
        montar(ini, L, largura // 2)
        montar(ini + L, R, largura // 2)

    largura = RUN
    while 2 * largura < len(v):            # largura da última passada
        largura *= 2
    montar(0, len(v), largura)
    return v


# --------------------------------------------------------------------------
# Medição e relatório
# --------------------------------------------------------------------------
def medir(ordenar, reps):
    """Menor e média de 'reps' execuções; devolve também o último resultado."""
    tempos = []
    for _ in range(reps):
        t, res = ordenar()
        tempos.append(t)
    return min(tempos), sum(tempos) / reps, res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    caminho = args[0] if args else "dados.csv"
    reps = max(1, int(args[1])) if len(args) > 1 else 5
    com_python = "--python" in sys.argv

    t0 = perf_counter()
    dados = ler(caminho)
    print(f"{len(dados):,} valores lidos de {caminho} em "
          f"{perf_counter() - t0:.2f}s")
    ordenado = np.sort(dados)              # gabarito
    trab, aux = np.empty_like(dados), np.empty_like(dados)

    t0 = perf_counter()
    merge_sort(dados.copy(), aux)          # compilação fora da medição
    print(f"compilação/carga do Numba: {perf_counter() - t0:.2f}s\n")

    def numba(entrada):
        trab[:] = entrada                  # cópia fora da medição (como no C)
        t0 = perf_counter()
        merge_sort(trab, aux)
        return perf_counter() - t0, trab

    def python(lista):
        t0 = perf_counter()
        res = merge_seq.merge_sort(lista)
        return perf_counter() - t0, res

    cab = (f"{'entrada':<12} {'n':>10} {'melhor(s)':>10} {'media(s)':>10} "
           f"{'válido':>7}")
    if com_python:
        cab += f" {'python(s)':>10} {'ganho':>7}"
    print(cab + "\n" + "-" * len(cab))

    casos = (("aleatória", dados), ("ordenada", ordenado),
             ("invertida", ordenado[::-1].copy()),
             ("PIOR CASO", pior_caso(ordenado)))
    for nome, entrada in casos:
        melhor, media, res = medir(lambda: numba(entrada), reps)
        ok = np.array_equal(res, ordenado)
        if nome == "aleatória":
            merge_seq.salvar_vetor(res.tolist(), SAIDA)
        linha = (f"{nome:<12} {len(entrada):>10,} {melhor:>10.4f} "
                 f"{media:>10.4f} {'sim' if ok else 'NÃO':>7}")
        if com_python:
            lista = merge_seq.como_lido(entrada.tolist())
            tp, _, res_py = medir(lambda: python(lista), reps)
            ok_py = res_py == ordenado.tolist()
            linha += (f" {tp:>10.4f} {tp / melhor:>6.1f}x"
                      + ("" if ok_py else "  <-- python INVÁLIDO"))
        print(linha)

    print(f"\nvetor ordenado (entrada aleatória) gravado em {SAIDA}")


if __name__ == "__main__":
    main()
