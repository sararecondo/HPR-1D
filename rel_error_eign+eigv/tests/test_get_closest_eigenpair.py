import numpy as np
import pytest
from types import SimpleNamespace

from dolfinx.fem import Function, FunctionSpace
from mpi4py import MPI
from dolfinx.mesh import create_unit_interval

# --------- IMPORTA A FUNCIÓN A TESTEAR -----------
from piecewise_module import get_closest_eigenpair


# ============================================================
# 1. Mock do solver e interface PETSc necesarias
# ============================================================

class MockSolver:
    def __init__(self, eigenvalues):
        """
        eigenvalues: lista de números complexos simulando autovalores λ_k
        """
        self._eigs = eigenvalues
        self._n = len(eigenvalues)

    def getConverged(self):
        return self._n

    def getOperators(self):
        # devolve un obxecto cun método getVecs(), pero non fai falta que sexa real
        class MockOperator:
            def getVecs(self):
                return np.zeros(self._n, dtype=complex), None
        return [MockOperator()]

    def getEigenpair(self, k, vrA):
        """
        Simula escribir o autovector en vrA e devolver λ_k.
        """
        vrA[:] = 0.0  # calquera cousa
        vrA[k] = 1.0  # marcamos posición para recoñecer o autovector
        return self._eigs[k]


# ============================================================
# 2. Fixture para crear Q, V e estruturas mínimas
# ============================================================

@pytest.fixture
def function_spaces():
    mesh = create_unit_interval(MPI.COMM_WORLD, 5)
    Q = FunctionSpace(mesh, ("CG", 1))
    V = FunctionSpace(mesh, ("CG", 1))
    dof = {"p": Q.dofmap.index_map.size_global,
           "v": V.dofmap.index_map.size_global}
    return Q, V, dof


# ============================================================
# 3. TEST: selección do autovalor máis próximo
# ============================================================

def test_get_closest_eigenpair(function_spaces):
    Q, V, dof = function_spaces

    # autovalores simulados para o mock solver
    eigs = [
        10 + 1j,
        20 - 3j,
        30 + 0.5j,
        40 - 1j,
        50 + 2j,
    ]

    solver = MockSolver(eigs)

    # obxectivo: 22 - 3j → o máis próximo é 20 - 3j (índice 1)
    omega_target = 22 - 3j

    lam_best, p_h, v_h = get_closest_eigenpair(
        solver,
        omega_target,
        Q, V, dof,
        verbose=False
    )

    # --- Comprobacións ---
    assert lam_best == eigs[1]      # selecciona o correcto
    assert isinstance(p_h, Function)
    assert isinstance(v_h, Function)

    # O autovector debe ter un 1.0 na posición do modo seleccionado
    # (segundo o comportamento que definimos no mock)
    assert np.isclose(p_h.x.array).max() > 0.9
    assert np.isclose(v_h.x.array).max() > 0.9
