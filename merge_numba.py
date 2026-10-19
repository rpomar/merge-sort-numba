"""merge_numba.py - Merge Sort SEQUENCIAL com Numba em vetores aleatórios
de tamanhos variados.

Uso:  python merge_numba.py [qtd_vetores] [repeticoes] [--python]

"""
import sys
from time import perf_counter

import numpy as np
from numba import njit

import merge_seq
from ordenacao_paralela import ARQUIVO, gerar_vetores, salvar

RUN = 16                 # tamanho dos blocos ordenados por inserção


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
# Dados, medição e relatório
# --------------------------------------------------------------------------
def ler():
    """Um vetor por linha -> arrays int32 (os valores vão até 10^9)."""
    try:
        with open(ARQUIVO, encoding="utf-8") as f:
            return [np.array(linha.split(), dtype=np.int32)
                    for linha in f if linha.strip()]
    except FileNotFoundError:
        return []


def carregar(qtd):
    vetores = ler()
    if len(vetores) != qtd:
        salvar(gerar_vetores(qtd))
        vetores = ler()
    return vetores


def medir(ordenar, entradas, gabaritos, reps):
    """Melhor tempo de cada vetor + validação de todos."""
    tempos, ok = [], True
    for v, g in zip(entradas, gabaritos):
        melhor = float("inf")
        for _ in range(reps):
            t0 = perf_counter()
            res = ordenar(v)
            melhor = min(melhor, perf_counter() - t0)
        ok &= bool(np.array_equal(res, g))
        tempos.append(melhor)
    return tempos, ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    qtd = int(args[0]) if args else 200
    reps = max(1, int(args[1])) if len(args) > 1 else 3
    com_python = "--python" in sys.argv

    vetores = carregar(qtd)
    tam = [len(v) for v in vetores]
    total = sum(tam)
    print(f"{qtd} vetores | {total:,} elementos | menor={min(tam):,} "
          f"maior={max(tam):,}")
    gabaritos = [np.sort(v) for v in vetores]

    aux = np.empty(max(tam), dtype=np.int32)

    def ordenar_nb(v):                     # devolve uma cópia ordenada,
        t = v.copy()                       # como merge_seq.merge_sort
        merge_sort(t, aux)
        return t

    t0 = perf_counter()
    ordenar_nb(vetores[0])                 # compilação fora da medição
    print(f"compilação/carga do Numba: {perf_counter() - t0:.2f}s\n")

    medidas = {"numba": medir(ordenar_nb, vetores, gabaritos, reps)}
    if com_python:
        listas = [v.tolist() for v in vetores]
        medidas["python"] = medir(merge_seq.merge_sort, listas, gabaritos,
                                  reps)

    print(f"{'faixa (n)':<18} {'vetores':>7} {'elementos':>11}"
          + "".join(f" {k + '(s)':>10}" for k in medidas)
          + (f" {'ganho':>7}" if com_python else ""))
    faixas = [(0, 10**4), (10**4, 5 * 10**4), (5 * 10**4, 2 * 10**5),
              (2 * 10**5, 10**9), (0, 10**9)]
    for lo, hi in faixas:
        idx = [i for i in range(qtd) if lo <= tam[i] < hi]
        if not idx:
            continue
        rotulo = ("TOTAL" if (lo, hi) == (0, 10**9) else
                  f">= {lo:,}" if hi == 10**9 else f"{lo:,} - {hi - 1:,}")
        soma = {k: sum(t[i] for i in idx) for k, (t, _) in medidas.items()}
        print(f"{rotulo:<18} {len(idx):>7} {sum(tam[i] for i in idx):>11,}"
              + "".join(f" {s:>10.4f}" for s in soma.values())
              + (f" {soma['python'] / soma['numba']:>6.1f}x"
                 if com_python else ""))

    for k, (t, ok) in medidas.items():
        print(f"{k}: {sum(t) / total * 1e9:.1f} ns por elemento | "
              f"válido: {'sim' if ok else 'NÃO'}")


if __name__ == "__main__":
    main()
