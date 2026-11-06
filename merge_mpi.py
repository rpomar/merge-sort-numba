"""merge_mpi.py - Merge Sort de vários vetores: sequencial x MPI.

Exemplos:
  python merge_mpi.py --cenario bordas
  mpiexec -n 4 python -m mpi4py merge_mpi.py
  mpiexec -n 4 python -m mpi4py merge_mpi.py --cenario variados --vetores 50
"""
import argparse
import csv
import os
import sys
from datetime import datetime
from time import perf_counter

import numpy as np

from merge_numba import merge_sort
from vetores import (CENARIOS, checar_parametros, desequilibrio, distribuir,
                     gerar_vetores, validar)

try:
    from mpi4py import MPI
except ImportError:
    sys.exit("mpi4py não encontrado: rode 'pip install -r requirements.txt' "
             "e instale uma implementação de MPI (veja o README)")


def ordenar_lote(lote):
    """Recebe pares (índice, vetor) e devolve pares (índice, vetor ordenado)."""
    maior = max((len(vetor) for _, vetor in lote), default=0)
    aux = np.empty(maior, dtype=np.int32)
    ordenados = []
    for indice, vetor in lote:
        copia = vetor.copy()
        merge_sort(copia, aux)
        ordenados.append((indice, copia))
    return ordenados


def sequencial(vetores):
    inicio = perf_counter()
    pares = ordenar_lote(list(enumerate(vetores)))
    tempo = perf_counter() - inicio
    return tempo, [vetor for _, vetor in pares]


def paralelo(comm, vetores, lotes):
    """Roda em todos os processos; só o processo 0 recebe os resultados."""
    comm.Barrier()
    inicio = perf_counter()

    envio = None
    if comm.rank == 0:
        envio = [[(i, vetores[i]) for i in lote] for lote in lotes]
    meu_lote = comm.scatter(envio, root=0)
    ordenados = ordenar_lote(meu_lote)
    partes = comm.gather(ordenados, root=0)

    if comm.rank != 0:
        return None, None
    resultado = [None] * len(vetores)
    for parte in partes:
        for indice, vetor in parte:
            resultado[indice] = vetor
    return perf_counter() - inicio, resultado


def medir(comm, vetores, lotes, repeticoes):
    tempos_seq, tempos_par = [], []
    res_seq = res_par = None
    for _ in range(repeticoes):
        if comm.rank == 0:
            tempo, res_seq = sequencial(vetores)
            tempos_seq.append(tempo)
        tempo, res_par = paralelo(comm, vetores, lotes)
        tempos_par.append(tempo)
    return tempos_seq, tempos_par, res_seq, res_par


def conferir(nome, vetores, res_seq, res_par):
    """Mostra os erros encontrados e diz se os dois resultados estão certos."""
    erros_seq = validar(res_seq, [np.sort(v) for v in vetores])
    erros_par = validar(res_par, res_seq)
    if erros_seq:
        print(f"ERRO [{nome}]: resultado sequencial incorreto nos vetores "
              f"{erros_seq[:10]}", file=sys.stderr)
    if erros_par:
        print(f"ERRO [{nome}]: resultado MPI incorreto nos vetores "
              f"{erros_par[:10]}", file=sys.stderr)
    return not erros_seq and not erros_par


def parametros(nome, args):
    """Valores do cenário, trocados pelas opções da linha de comando."""
    opcoes = (args.vetores, args.tam_min, args.tam_max, args.vmin, args.vmax)
    return tuple(padrao if opcao is None else opcao
                 for padrao, opcao in zip(CENARIOS[nome], opcoes))


def executar(comm, nome, args):
    """Roda um cenário. O processo 0 devolve a linha do CSV."""
    qtd, tam_min, tam_max, vmin, vmax = parametros(nome, args)
    vetores = lotes = None
    if comm.rank == 0:
        vetores = gerar_vetores(qtd, tam_min, tam_max, vmin, vmax,
                                args.semente)
        lotes = distribuir([len(v) for v in vetores], comm.size)

    tempos_seq, tempos_par, res_seq, res_par = medir(comm, vetores, lotes,
                                                     args.repeticoes)
    if comm.rank != 0:
        return None

    tamanhos = [len(v) for v in vetores]
    speedup = min(tempos_seq) / min(tempos_par)
    ok = conferir(nome, vetores, res_seq, res_par)
    return {
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "cenario": nome,
        "semente": args.semente,
        "processos": comm.size,
        "processos_ociosos": sum(1 for lote in lotes if not lote),
        "vetores": qtd,
        "total_elementos": sum(tamanhos),
        "tam_min": tam_min,
        "tam_max": tam_max,
        "vmin": vmin,
        "vmax": vmax,
        "repeticoes": args.repeticoes,
        "tempo_seq_s": round(min(tempos_seq), 6),
        "tempo_par_s": round(min(tempos_par), 6),
        "tempo_seq_medio_s": round(sum(tempos_seq) / len(tempos_seq), 6),
        "tempo_par_medio_s": round(sum(tempos_par) / len(tempos_par), 6),
        "speedup": round(speedup, 3),
        "eficiencia": round(speedup / comm.size, 3),
        "desequilibrio": round(desequilibrio(tamanhos, lotes), 3),
        "validacao": "ok" if ok else "falhou",
    }


def gravar_csv(caminho, linha):
    novo = not os.path.exists(caminho)
    with open(caminho, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=linha.keys())
        if novo:
            escritor.writeheader()
        escritor.writerow(linha)


def imprimir_cabecalho(comm, args):
    print(f"{comm.size} processo(s) MPI, semente {args.semente}, "
          f"menor de {args.repeticoes} medição(ões)\n")
    titulo = (f"{'cenário':<9} {'vetores':>7} {'elementos':>11} "
              f"{'seq(s)':>8} {'mpi(s)':>8} {'speedup':>7} {'efic.':>6} "
              f"{'deseq.':>6} {'validação':>9}")
    print(titulo)
    print("-" * len(titulo))


def imprimir_linha(linha):
    print(f"{linha['cenario']:<9} {linha['vetores']:>7,} "
          f"{linha['total_elementos']:>11,} "
          f"{linha['tempo_seq_s']:>8.4f} {linha['tempo_par_s']:>8.4f} "
          f"{linha['speedup']:>7.3f} {linha['eficiencia']:>6.3f} "
          f"{linha['desequilibrio']:>6.3f} {linha['validacao']:>9}")
    if linha["processos_ociosos"]:
        print(f"  ({linha['processos_ociosos']} processo(s) sem vetores "
              "para ordenar)")


def ler_argumentos(argv=None):
    parser = argparse.ArgumentParser(
        description="Compara o Merge Sort sequencial com a distribuição dos "
                    "vetores entre processos MPI.",
        epilog="Para usar P processos: mpiexec -n P python -m mpi4py "
               "merge_mpi.py. As opções de vetores, tamanhos e valores "
               "substituem os valores do cenário.")
    parser.add_argument("--cenario", choices=[*CENARIOS, "todos"],
                        default="todos", help="padrão: todos")
    parser.add_argument("--semente", type=int, default=42,
                        help="semente do gerador (padrão: 42)")
    parser.add_argument("--repeticoes", type=int, default=3,
                        help="medições por cenário (padrão: 3)")
    parser.add_argument("--vetores", type=int, help="quantidade de vetores")
    parser.add_argument("--tam-min", type=int, help="tamanho mínimo")
    parser.add_argument("--tam-max", type=int, help="tamanho máximo")
    parser.add_argument("--vmin", type=int, help="menor valor possível")
    parser.add_argument("--vmax", type=int, help="maior valor possível")
    parser.add_argument("--csv", default="resultados.csv",
                        help="arquivo de resultados (padrão: resultados.csv)")
    args = parser.parse_args(argv)

    if args.repeticoes < 1:
        parser.error("--repeticoes precisa ser pelo menos 1")
    args.cenarios = list(CENARIOS)
    if args.cenario != "todos":
        args.cenarios = [args.cenario]
    for nome in args.cenarios:
        try:
            checar_parametros(*parametros(nome, args))
        except ValueError as erro:
            parser.error(f"cenário {nome}: {erro}")
    return args


def main(argv=None):
    args = ler_argumentos(argv)
    comm = MPI.COMM_WORLD

    # a primeira chamada compila o Numba; assim isso fica fora das medições
    merge_sort(np.array([2, 1], dtype=np.int32), np.empty(2, dtype=np.int32))

    if comm.rank == 0:
        imprimir_cabecalho(comm, args)

    falhou = False
    for nome in args.cenarios:
        linha = executar(comm, nome, args)
        if comm.rank == 0:
            imprimir_linha(linha)
            gravar_csv(args.csv, linha)
            if linha["validacao"] != "ok":
                falhou = True

    falhou = comm.bcast(falhou, root=0)
    if comm.rank == 0:
        if falhou:
            print("\nvalidação FALHOU: veja as mensagens de erro acima",
                  file=sys.stderr)
        else:
            print(f"\nresultados acrescentados em {args.csv}")
    if falhou:
        sys.exit(1)


if __name__ == "__main__":
    main()
