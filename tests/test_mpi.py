"""Testes do programa principal: com um processo e, se houver mpiexec, com
vários processos de verdade."""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mpi4py")
import merge_mpi  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
MPIEXEC = shutil.which("mpiexec")


def ler_csv(caminho):
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_ordenar_lote_preserva_ids_e_entrada():
    lote = [(4, np.array([3, -1, 3], dtype=np.int32)),
            (0, np.array([], dtype=np.int32))]
    ordenados = merge_mpi.ordenar_lote(lote)
    assert [i for i, _ in ordenados] == [4, 0]
    assert ordenados[0][1].tolist() == [-1, 3, 3]
    assert lote[0][1].tolist() == [3, -1, 3]


def test_sequencial_mantem_a_ordem_original():
    vetores = [np.array([2, 1], dtype=np.int32), np.array([9], dtype=np.int32)]
    _, ordenados = merge_mpi.sequencial(vetores)
    assert [v.tolist() for v in ordenados] == [[1, 2], [9]]


def test_main_um_processo(tmp_path):
    saida = tmp_path / "r.csv"
    merge_mpi.main(["--cenario", "bordas", "--repeticoes", "1",
                    "--csv", str(saida)])
    [linha] = ler_csv(saida)
    assert linha["validacao"] == "ok"
    assert linha["processos"] == "1"
    assert linha["vetores"] == "200"


def test_main_falha_na_validacao(tmp_path, monkeypatch):
    monkeypatch.setattr(merge_mpi, "merge_sort", lambda v, aux: None)
    saida = tmp_path / "r.csv"
    with pytest.raises(SystemExit) as erro:
        merge_mpi.main(["--cenario", "bordas", "--repeticoes", "1",
                        "--csv", str(saida)])
    assert erro.value.code == 1
    assert ler_csv(saida)[0]["validacao"] == "falhou"


def test_argumentos_invalidos():
    with pytest.raises(SystemExit) as erro:
        merge_mpi.ler_argumentos(["--tam-min", "10", "--tam-max", "2"])
    assert erro.value.code == 2


def rodar_mpi(processos, csv_saida, *args):
    comando = [MPIEXEC, "-n", str(processos), sys.executable, "-m", "mpi4py",
               "merge_mpi.py", "--repeticoes", "1", "--csv", str(csv_saida),
               *args]
    return subprocess.run(comando, cwd=RAIZ, capture_output=True, text=True,
                          timeout=300)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec não encontrado no PATH")
def test_mpi_dois_processos(tmp_path):
    saida = tmp_path / "r.csv"
    r = rodar_mpi(2, saida, "--cenario", "variados", "--vetores", "30",
                  "--tam-max", "5000")
    assert r.returncode == 0, r.stderr
    [linha] = ler_csv(saida)
    assert (linha["processos"], linha["validacao"]) == ("2", "ok")


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec não encontrado no PATH")
def test_mpi_mais_processos_que_vetores(tmp_path):
    saida = tmp_path / "r.csv"
    r = rodar_mpi(4, saida, "--cenario", "bordas", "--vetores", "2")
    assert r.returncode == 0, r.stderr
    [linha] = ler_csv(saida)
    assert linha["validacao"] == "ok"
    assert linha["processos_ociosos"] == "2"
