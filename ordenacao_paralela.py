"""Ordenação paralela de múltiplos vetores de tamanhos variados.

Uso:
    python ordenacao_paralela.py [qtd_vetores] [repeticoes]
"""
import math
import random
import statistics
import sys
from collections import Counter
from multiprocessing import Pool, cpu_count
from time import perf_counter

ARQUIVO = "vetores.txt"


# --------------------------------------------------------------------------
# PASSO 1 - Gerar dados com tamanhos de "cauda longa"
# --------------------------------------------------------------------------
def gerar_vetores(qtd, seed=42, tam_min=5000, tam_max=1_000_000):
    rng = random.Random(seed)              # seed fixa = experimento reproduzível
    vetores = []
    for _ in range(qtd):
        # Pareto: muitos vetores pequenos e poucos gigantes
        n = int(tam_min * rng.paretovariate(1.1))
        n = min(n, tam_max)
        vetores.append([rng.randint(0, 10**9) for _ in range(n)])
    return vetores


# --------------------------------------------------------------------------
# PASSO 2 - Salvar e ler o arquivo (um vetor por linha)
# --------------------------------------------------------------------------
def salvar(vetores, caminho=ARQUIVO):
    with open(caminho, "w", encoding="utf-8") as f:
        for v in vetores:
            f.write(" ".join(map(str, v)) + "\n")


def ler(caminho=ARQUIVO):
    with open(caminho, encoding="utf-8") as f:
        return [list(map(int, linha.split())) for linha in f if linha.strip()]


# --------------------------------------------------------------------------
# PASSO 3 - Validação independente
# --------------------------------------------------------------------------
def ordenacao_valida(original, resultado):
    if len(original) != len(resultado):
        return False
    # (a) está em ordem crescente?
    if any(resultado[i] > resultado[i + 1] for i in range(len(resultado) - 1)):
        return False
    # (b) é uma permutação da entrada? (nada sumiu, nada duplicou)
    return Counter(original) == Counter(resultado)


# --------------------------------------------------------------------------
# PASSO 4 - A tarefa: a unidade de trabalho enviada a cada processo
# --------------------------------------------------------------------------
def ordenar_tarefa(item):
    idx, vetor = item
    t0 = perf_counter()
    resultado = sorted(vetor)
    return idx, resultado, perf_counter() - t0


# --------------------------------------------------------------------------
# PASSO 5 - Versão sequencial (a linha de base do speedup)
# --------------------------------------------------------------------------
def sequencial(vetores):
    return [ordenar_tarefa((i, v)) for i, v in enumerate(vetores)]


# --------------------------------------------------------------------------
# PASSO 6 - Versões paralelas: três estratégias de distribuição
# --------------------------------------------------------------------------
def paralelo(pool, procs, vetores, estrategia):
    itens = list(enumerate(vetores))

    if estrategia == "estatica":
        # cada processo recebe um bloco fixo com ~len/procs vetores
        bloco = math.ceil(len(itens) / procs)
        resultados = pool.map(ordenar_tarefa, itens, chunksize=bloco)

    elif estrategia == "dinamica":
        # fila: quem termina pega o próximo vetor
        resultados = list(pool.imap_unordered(ordenar_tarefa, itens,
                                              chunksize=1))

    elif estrategia == "lpt":
        # Longest Processing Time first: maiores primeiro, depois a fila
        itens.sort(key=lambda it: len(it[1]), reverse=True)
        resultados = list(pool.imap_unordered(ordenar_tarefa, itens,
                                              chunksize=1))
    else:
        raise ValueError(estrategia)

    resultados.sort(key=lambda r: r[0])    # devolve na ordem original
    return resultados


# --------------------------------------------------------------------------
# PASSO 7 - Medição com repetições
# --------------------------------------------------------------------------
def medir(funcao, repeticoes):
    funcao()                                # aquecimento (descartado)
    tempos = []
    for _ in range(repeticoes):
        t0 = perf_counter()
        resultado = funcao()
        tempos.append(perf_counter() - t0)
    desvio = statistics.stdev(tempos) if len(tempos) > 1 else 0.0
    return statistics.mean(tempos), desvio, resultado


# --------------------------------------------------------------------------
# PASSO 8 - Programa principal
# --------------------------------------------------------------------------
def main():
    qtd = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    salvar(gerar_vetores(qtd))
    vetores = ler()
    tamanhos = [len(v) for v in vetores]
    total = sum(tamanhos)
    maior = max(tamanhos)
    print(f"{qtd} vetores | {total:,} elementos | menor={min(tamanhos)} "
          f"maior={maior:,} ({100 * maior / total:.1f}% do total)")

    # linha de base
    t_seq, d_seq, res_seq = medir(lambda: sequencial(vetores), reps)
    assert all(ordenacao_valida(vetores[i], r) for i, r, _ in res_seq)
    trabalho = sum(dt for _, _, dt in res_seq)  # tempo só ordenando
    print(f"\nsequencial: {t_seq:.3f}s ± {d_seq:.3f}")

    # limite teórico imposto pelo maior vetor (Amdahl por tarefa)
    t_maior = max(dt for _, _, dt in res_seq)
    print(f"speedup máximo possível (maior vetor): {trabalho / t_maior:.2f}x\n")

    print(f"{'procs':>5} {'estratégia':>10} {'tempo(s)':>9} "
          f"{'±':>6} {'speedup':>8} {'efic.':>6}")
    procs_lista = sorted({1, 2, 4, cpu_count()})
    for p in procs_lista:
        # o Pool é criado fora da medição: medimos só a ordenação + envio
        with Pool(p) as pool:
            for est in ("estatica", "dinamica", "lpt"):
                t, d, res = medir(lambda: paralelo(pool, p, vetores, est),
                                  reps)
                ok = all(ordenacao_valida(vetores[i], r) for i, r, _ in res)
                s = t_seq / t
                print(f"{p:>5} {est:>10} {t:>9.3f} {d:>6.3f} "
                      f"{s:>7.2f}x {s / p:>6.2f}"
                      + ("" if ok else "  <-- RESULTADO INVÁLIDO"))


if __name__ == "__main__":     # obrigatório no Windows (método spawn)
    main()
