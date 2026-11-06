"""vetores.py - cenários, geração, distribuição e validação dos vetores."""
import numpy as np

INT32 = np.iinfo(np.int32)

# (vetores, tamanho mínimo, tamanho máximo, menor valor, maior valor)
CENARIOS = {
    "pequenos": (5000, 0, 2_000, -1_000_000, 1_000_000),
    "grandes": (8, 1_000_000, 2_000_000, -1_000_000, 1_000_000),
    "variados": (100, 0, 400_000, -1_000_000, 1_000_000),
    "bordas": (200, 0, 3, -2, 2),
}


def checar_parametros(qtd, tam_min, tam_max, vmin, vmax):
    if qtd < 0 or not 0 <= tam_min <= tam_max:
        raise ValueError("exige vetores >= 0 e 0 <= tam-min <= tam-max")
    if not INT32.min <= vmin <= vmax <= INT32.max:
        raise ValueError(f"exige {INT32.min} <= vmin <= vmax <= {INT32.max}")


def gerar_vetores(qtd, tam_min, tam_max, vmin, vmax, semente):
    checar_parametros(qtd, tam_min, tam_max, vmin, vmax)
    rng = np.random.default_rng(semente)
    tamanhos = rng.integers(tam_min, tam_max, size=qtd, endpoint=True)
    return [rng.integers(vmin, vmax, size=n, dtype=np.int32, endpoint=True)
            for n in tamanhos]


def distribuir(tamanhos, processos):
    """Devolve, para cada processo, a lista de índices que ele vai ordenar.

    Do maior vetor para o menor, cada um vai para o processo com menos
    elementos. No empate, vai para o que tem menos vetores, o que espalha
    também os vetores vazios.
    """
    if processos < 1:
        raise ValueError("exige pelo menos 1 processo")
    lotes = [[] for _ in range(processos)]
    cargas = [0] * processos
    indices = sorted(range(len(tamanhos)), key=lambda i: tamanhos[i],
                     reverse=True)
    for i in indices:
        p = min(range(processos), key=lambda p: (cargas[p], len(lotes[p])))
        lotes[p].append(i)
        cargas[p] += tamanhos[i]
    return lotes


def desequilibrio(tamanhos, lotes):
    """Maior carga dividida pela carga média (1.0 = cargas iguais)."""
    cargas = [sum(tamanhos[i] for i in lote) for lote in lotes]
    media = sum(cargas) / len(cargas)
    return max(cargas) / media if media else 1.0


def validar(obtidos, esperados):
    """Devolve os índices dos vetores que diferem do esperado."""
    if len(obtidos) != len(esperados):
        raise ValueError(f"{len(obtidos)} resultados para "
                         f"{len(esperados)} vetores")
    erros = []
    for i in range(len(esperados)):
        if not np.array_equal(obtidos[i], esperados[i]):
            erros.append(i)
    return erros
