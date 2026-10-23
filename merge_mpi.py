"""merge_mpi.py - Merge Sort de vários vetores: sequencial x MPI.

Exemplos:
  python merge_mpi.py --cenario bordas                   (1 processo)
  mpiexec -n 4 python -m mpi4py merge_mpi.py             (todos os cenários)
  mpiexec -n 4 python -m mpi4py merge_mpi.py --cenario variados --vetores 50

O processo 0 gera os vetores (fora das medições), ordena todos sozinho
(referência sequencial) e depois reparte o trabalho entre os processos MPI.
Os dois resultados são validados e a linha de métricas vai para o CSV.
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

CAMPOS = ["data_hora", "cenario", "semente", "processos", "processos_ociosos",
          "vetores", "total_elementos", "tam_min", "tam_max", "vmin", "vmax",
          "repeticoes", "tempo_seq_s", "tempo_par_s", "tempo_seq_medio_s",
          "tempo_par_medio_s", "speedup", "eficiencia", "desequilibrio",
          "validacao"]


def ordenar_lote(lote):
    """[(id, vetor), ...] -> [(id, cópia ordenada), ...]"""
    aux = np.empty(max((len(v) for _, v in lote), default=0), dtype=np.int32)
    ordenados = []
    for i, v in lote:
        w = v.copy()
        merge_sort(w, aux)
        ordenados.append((i, w))
    return ordenados


def sequencial(vetores):
    t0 = perf_counter()
    ordenados = [w for _, w in ordenar_lote(list(enumerate(vetores)))]
    return perf_counter() - t0, ordenados


def paralelo(comm, vetores, lotes):
    """Chamado por todos os processos; só o 0 recebe (tempo, resultados).

    O tempo vai do envio dos lotes até os resultados reunidos na ordem
    original: inclui serialização, comunicação e espera pelo processo mais
    lento.
    """
    comm.Barrier()
    t0 = perf_counter()
    envio = None
    if comm.rank == 0:
        envio = [[(i, vetores[i]) for i in lote] for lote in lotes]
    # scatter/gather com pickle aceitam lotes de tamanhos diferentes; versões
    # com buffer (Scatterv) evitariam a serialização, mas exigem deslocamentos
    partes = comm.gather(ordenar_lote(comm.scatter(envio, root=0)), root=0)
    if comm.rank != 0:
        return None
    ordenados = [None] * len(vetores)
    for parte in partes:
        for i, w in parte:
            ordenados[i] = w
    return perf_counter() - t0, ordenados


def parametros(nome, args):
    padrao = CENARIOS[nome]
    escolhidos = (args.vetores, args.tam_min, args.tam_max, args.vmin,
                  args.vmax)
    return tuple(p if a is None else a for p, a in zip(padrao, escolhidos))


def executar(comm, nome, args):
    """Roda um cenário em todos os processos; o 0 devolve a linha do CSV."""
    qtd, tam_min, tam_max, vmin, vmax = parametros(nome, args)
    vetores = lotes = None
    if comm.rank == 0:
        vetores = gerar_vetores(qtd, tam_min, tam_max, vmin, vmax,
                                args.semente)
        lotes = distribuir([len(v) for v in vetores], comm.size)

    tempos_seq, tempos_par = [], []
    for _ in range(args.repeticoes):
        if comm.rank == 0:
            t, res_seq = sequencial(vetores)
            tempos_seq.append(t)
        r = paralelo(comm, vetores, lotes)
        if comm.rank == 0:
            tempos_par.append(r[0])
            res_par = r[1]
    if comm.rank != 0:
        return None

    erros_seq = validar(res_seq, [np.sort(v) for v in vetores])
    erros_par = validar(res_par, res_seq)
    for rotulo, erros in (("sequencial", erros_seq), ("MPI", erros_par)):
        if erros:
            print(f"ERRO [{nome}]: resultado {rotulo} incorreto nos vetores "
                  f"{erros[:10]}{' ...' if len(erros) > 10 else ''}",
                  file=sys.stderr)

    seq, par = min(tempos_seq), min(tempos_par)
    speedup = seq / par if par > 0 else float("nan")
    tamanhos = [len(v) for v in vetores]
    return {
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "cenario": nome, "semente": args.semente, "processos": comm.size,
        "processos_ociosos": sum(not lote for lote in lotes),
        "vetores": qtd, "total_elementos": sum(tamanhos),
        "tam_min": tam_min, "tam_max": tam_max, "vmin": vmin, "vmax": vmax,
        "repeticoes": args.repeticoes,
        "tempo_seq_s": f"{seq:.6f}", "tempo_par_s": f"{par:.6f}",
        "tempo_seq_medio_s": f"{np.mean(tempos_seq):.6f}",
        "tempo_par_medio_s": f"{np.mean(tempos_par):.6f}",
        "speedup": f"{speedup:.3f}",
        "eficiencia": f"{speedup / comm.size:.3f}",
        "desequilibrio": f"{desequilibrio(tamanhos, lotes):.3f}",
        "validacao": "falhou" if erros_seq or erros_par else "ok",
    }


def gravar_csv(caminho, linha):
    novo = not os.path.exists(caminho) or os.path.getsize(caminho) == 0
    with open(caminho, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            escritor.writeheader()
        escritor.writerow(linha)


def ler_argumentos(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge Sort (Numba) de vários vetores aleatórios: "
                    "execução sequencial de referência x distribuição entre "
                    "processos MPI. Rode com 'mpiexec -n P python -m mpi4py "
                    "merge_mpi.py' para usar P processos.",
        epilog="Cenários (vetores, tam. mín., tam. máx., vmin, vmax): "
               + "; ".join(f"{n} {p}" for n, p in CENARIOS.items())
               + ". As opções --vetores, --tam-min, --tam-max, --vmin e "
                 "--vmax substituem os valores do cenário.")
    parser.add_argument("--cenario", choices=[*CENARIOS, "todos"],
                        default="todos", help="padrão: todos")
    parser.add_argument("--semente", type=int, default=42,
                        help="semente do gerador (padrão: 42)")
    parser.add_argument("--repeticoes", type=int, default=3,
                        help="medições por cenário; vale a menor "
                             "(padrão: 3)")
    parser.add_argument("--vetores", type=int, help="quantidade de vetores")
    parser.add_argument("--tam-min", type=int, help="tamanho mínimo")
    parser.add_argument("--tam-max", type=int, help="tamanho máximo")
    parser.add_argument("--vmin", type=int, help="menor valor possível")
    parser.add_argument("--vmax", type=int, help="maior valor possível")
    parser.add_argument("--csv", default="resultados.csv",
                        help="arquivo onde as linhas são acrescentadas "
                             "(padrão: resultados.csv)")
    args = parser.parse_args(argv)
    if args.repeticoes < 1:
        parser.error("--repeticoes precisa ser pelo menos 1")
    args.cenarios = list(CENARIOS) if args.cenario == "todos" else [args.cenario]
    for nome in args.cenarios:
        try:
            checar_parametros(*parametros(nome, args))
        except ValueError as e:
            parser.error(f"cenário {nome}: {e}")
    return args


def main(argv=None):
    args = ler_argumentos(argv)
    comm = MPI.COMM_WORLD
    merge_sort(np.arange(3, 0, -1, dtype=np.int32),       # compila/carrega o
               np.empty(3, dtype=np.int32))                # Numba fora do tempo

    if comm.rank == 0:
        print(f"{comm.size} processo(s) MPI, semente {args.semente}, "
              f"menor de {args.repeticoes} medição(ões)\n")
        cab = (f"{'cenário':<9} {'vetores':>7} {'elementos':>11} "
               f"{'seq(s)':>8} {'mpi(s)':>8} {'speedup':>7} {'efic.':>6} "
               f"{'deseq.':>6} {'validação':>9}")
        print(cab + "\n" + "-" * len(cab))

    falhou = False
    for nome in args.cenarios:
        linha = executar(comm, nome, args)
        if comm.rank == 0:
            print(f"{nome:<9} {linha['vetores']:>7,} "
                  f"{linha['total_elementos']:>11,} "
                  f"{float(linha['tempo_seq_s']):>8.4f} "
                  f"{float(linha['tempo_par_s']):>8.4f} "
                  f"{linha['speedup']:>7} {linha['eficiencia']:>6} "
                  f"{linha['desequilibrio']:>6} {linha['validacao']:>9}")
            if linha["processos_ociosos"]:
                print(f"  ({linha['processos_ociosos']} processo(s) sem "
                      f"vetores para ordenar)")
            gravar_csv(args.csv, linha)
            falhou |= linha["validacao"] != "ok"

    if comm.bcast(falhou, root=0):
        if comm.rank == 0:
            print("\nvalidação FALHOU: veja as mensagens de erro acima",
                  file=sys.stderr)
        sys.exit(1)
    if comm.rank == 0:
        print(f"\nresultados acrescentados em {args.csv}")


if __name__ == "__main__":
    main()
