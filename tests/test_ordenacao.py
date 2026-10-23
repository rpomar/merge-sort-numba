import numpy as np
import pytest

from merge_numba import merge_sort

CASOS = {
    "vazio": [],
    "unitario": [7],
    "repetidos": [3, 3, 3, 3],
    "negativos": [-5, 0, -1, 4, -5, 2],
    "extremos_int32": [2**31 - 1, -2**31, 0, -2**31],
    "invertido": list(range(100, 0, -1)),
}


def ordenar(valores):
    v = np.array(valores, dtype=np.int32)
    merge_sort(v, np.empty_like(v))
    return v


@pytest.mark.parametrize("valores", CASOS.values(), ids=CASOS.keys())
def test_casos_de_borda(valores):
    assert ordenar(valores).tolist() == sorted(valores)


# tamanhos em volta dos blocos de 16 e das passadas do bottom-up
@pytest.mark.parametrize("n", [2, 15, 16, 17, 31, 33, 100, 1000, 4097])
def test_tamanhos_aleatorios(n):
    valores = np.random.default_rng(n).integers(-50, 50, size=n)
    assert np.array_equal(ordenar(valores), np.sort(valores))


def test_aux_maior_que_o_vetor():
    v = np.array([3, 1, 2] * 20, dtype=np.int32)
    merge_sort(v, np.empty(1000, dtype=np.int32))
    assert np.array_equal(v, np.sort(v))
