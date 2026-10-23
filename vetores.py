"""vetores.py - geração, distribuição e validação dos vetores do hackathon.

Não depende de MPI: tudo aqui é testável com um único processo.
"""
import heapq

import numpy as np

INT32 = np.iinfo(np.int32)

# (vetores, tamanho mínimo, tamanho máximo, menor valor, maior valor)
CENARIOS = {
    "pequenos": (5000, 0, 2_000, -1_000_000, 1_000_000),
    "grandes": (8, 1_000_000, 2_000_000, -1_000_000, 1_000_000),
    "variados": (100, 0, 400_000, -1_000_000, 1_000_000),
    # vazios, unitários, muitos repetidos e negativos
    "bordas": (200, 0, 3, -2, 2),
}


def checar_parametros(qtd, tam_min, tam_max, vmin, vmax):
    if qtd < 0 or not 0 <= tam_min <= tam_max:
        raise ValueError("exige vetores >= 0 e 0 <= tam-min <= tam-max")
    if not INT32.min <= vmin <= vmax <= INT32.max:
        raise ValueError(f"exige {INT32.min} <= vmin <= vmax <= {INT32.max}")


def gerar_vetores(qtd, tam_min, tam_max, vmin, vmax, semente):
    """Lista de 'qtd' vetores int32 com tamanhos e valores uniformes."""
    checar_parametros(qtd, tam_min, tam_max, vmin, vmax)
    rng = np.random.default_rng(semente)
    tamanhos = rng.integers(tam_min, tam_max, size=qtd, endpoint=True)
    return [rng.integers(vmin, vmax, size=n, dtype=np.int32, endpoint=True)
            for n in tamanhos]


def distribuir(tamanhos, processos):
    """Divide os índices dos vetores entre 'processos' listas.

    Guloso do maior para o menor: cada vetor vai para o processo com menos
    elementos até o momento. O empate é decidido pela quantidade de vetores,
    para que vetores vazios também se espalhem.
    """
    if processos < 1:
        raise ValueError("exige pelo menos 1 processo")
    lotes = [[] for _ in range(processos)]
    heap = [(0, 0, p) for p in range(processos)]
    for i in sorted(range(len(tamanhos)), key=lambda i: -tamanhos[i]):
        carga, qtd, p = heapq.heappop(heap)
        lotes[p].append(i)
        heapq.heappush(heap, (carga + tamanhos[i], qtd + 1, p))
    return lotes


def desequilibrio(tamanhos, lotes):
    """Maior carga / carga média, em elementos (1.0 = perfeito)."""
    cargas = [sum(tamanhos[i] for i in lote) for lote in lotes]
    media = sum(cargas) / len(cargas)
    return max(cargas) / media if media else 1.0


def validar(obtidos, esperados):
    """Índices dos vetores em que 'obtidos' difere de 'esperados'."""
    if len(obtidos) != len(esperados):
        raise ValueError(f"{len(obtidos)} resultados para "
                         f"{len(esperados)} vetores")
    return [i for i, (a, b) in enumerate(zip(obtidos, esperados))
            if not np.array_equal(a, b)]
