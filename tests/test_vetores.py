import numpy as np
import pytest

from vetores import (CENARIOS, desequilibrio, distribuir, gerar_vetores,
                     validar)


# ---------------------------------------------------------------- gerador
def test_gerador_reproduzivel_pela_semente():
    a = gerar_vetores(20, 0, 50, -10, 10, semente=1)
    b = gerar_vetores(20, 0, 50, -10, 10, semente=1)
    c = gerar_vetores(20, 0, 50, -10, 10, semente=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_gerador_respeita_limites():
    vetores = gerar_vetores(200, 3, 40, -7, 9, semente=0)
    assert len(vetores) == 200
    assert all(v.dtype == np.int32 for v in vetores)
    assert all(3 <= len(v) <= 40 for v in vetores)
    tudo = np.concatenate(vetores)
    assert tudo.min() >= -7 and tudo.max() <= 9


def test_gerador_extremos_int32():
    v = gerar_vetores(1, 1000, 1000, -2**31, 2**31 - 1, semente=0)[0]
    assert v.min() < 0 < v.max()


def test_gerador_tamanhos_desiguais():
    tamanhos = {len(v) for v in gerar_vetores(50, 0, 10_000, 0, 9, 3)}
    assert len(tamanhos) > 40


def test_gerador_sem_vetores_e_vetores_vazios():
    assert gerar_vetores(0, 0, 10, 0, 1, semente=0) == []
    assert all(len(v) == 0 for v in gerar_vetores(5, 0, 0, 0, 1, 0))


@pytest.mark.parametrize("params", [
    (-1, 0, 1, 0, 1),            # quantidade negativa
    (1, 5, 2, 0, 1),             # tam_min > tam_max
    (1, -1, 2, 0, 1),            # tamanho negativo
    (1, 0, 1, 5, 2),             # vmin > vmax
    (1, 0, 1, 0, 2**31),         # fora de int32
])
def test_gerador_rejeita_parametros_invalidos(params):
    with pytest.raises(ValueError):
        gerar_vetores(*params, semente=0)


def test_cenario_bordas_tem_os_casos_especiais():
    vetores = gerar_vetores(*CENARIOS["bordas"], semente=42)
    tamanhos = [len(v) for v in vetores]
    assert 0 in tamanhos and 1 in tamanhos
    assert any(len(np.unique(v)) < len(v) for v in vetores)   # repetidos
    assert any((v < 0).any() for v in vetores)


# ------------------------------------------------------------ distribuição
def test_distribuir_cobre_cada_vetor_uma_vez():
    tamanhos = [5, 0, 9, 1, 1, 7, 0, 3]
    lotes = distribuir(tamanhos, 3)
    assert len(lotes) == 3
    assert sorted(i for lote in lotes for i in lote) == list(range(8))


def test_distribuir_considera_o_tamanho():
    lotes = distribuir([100, 1, 1, 1, 1], 2)
    assert sorted(map(sorted, lotes)) == [[0], [1, 2, 3, 4]]
    tamanhos = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert desequilibrio(tamanhos, distribuir(tamanhos, 2)) < 1.05


def test_distribuir_mais_processos_que_vetores():
    lotes = distribuir([4, 2], 5)
    assert sorted(len(lote) for lote in lotes) == [0, 0, 0, 1, 1]


def test_distribuir_espalha_vetores_vazios():
    assert all(len(lote) == 1 for lote in distribuir([0, 0, 0, 0], 4))


def test_distribuir_exige_processo():
    with pytest.raises(ValueError):
        distribuir([1], 0)


def test_desequilibrio():
    assert desequilibrio([3, 3], [[0], [1]]) == 1.0
    assert desequilibrio([6, 0], [[0], [1]]) == 2.0
    assert desequilibrio([0, 0], [[0], [1]]) == 1.0


# --------------------------------------------------------------- validação
def test_validar_aceita_resultados_iguais():
    esperados = [np.array([], dtype=np.int32), np.array([-1, 2])]
    assert validar([v.copy() for v in esperados], esperados) == []


def test_validar_aponta_vetores_errados():
    esperados = [np.array([1, 2]), np.array([3]), np.array([4, 5])]
    obtidos = [np.array([1, 2]), np.array([]), np.array([5, 4])]
    assert validar(obtidos, esperados) == [1, 2]


def test_validar_quantidade_diferente():
    with pytest.raises(ValueError):
        validar([np.array([1])], [])
