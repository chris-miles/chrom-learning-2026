import numpy as np

from chromlearn.io.trajectory import TrimmedCell
from chromlearn.model_fitting.basis import HatBasis
from chromlearn.model_fitting.features import build_design_matrix


def make_simple_trimmed_cell(T: int = 20, N: int = 4) -> TrimmedCell:
    centrioles = np.zeros((T, 3, 2))
    centrioles[:, 0, 0] = -5.0
    centrioles[:, 0, 1] = 5.0

    chromosomes = np.zeros((T, 3, N))
    positions = np.array(
        [[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]],
        dtype=float,
    )
    for chrom_index in range(N):
        chromosomes[:, :, chrom_index] = positions[chrom_index]

    return TrimmedCell(
        cell_id="test_001",
        condition="test",
        centrioles=centrioles,
        chromosomes=chromosomes,
        tracked=N,
        dt=5.0,
        start_frame=0,
        end_frame=T - 1,
    )


def test_design_matrix_shape() -> None:
    cell = make_simple_trimmed_cell(T=20, N=4)
    G, V = build_design_matrix(
        [cell],
        HatBasis(0.0, 10.0, n_basis=5),
        HatBasis(0.0, 15.0, n_basis=5),
    )
    assert G.shape == (4 * 19 * 3, 10)
    assert V.shape == (4 * 19 * 3,)


def test_stationary_particles_zero_response() -> None:
    cell = make_simple_trimmed_cell(T=20, N=4)
    _, V = build_design_matrix(
        [cell],
        HatBasis(0.0, 10.0, n_basis=5),
        HatBasis(0.0, 15.0, n_basis=5),
    )
    np.testing.assert_allclose(V, 0.0, atol=1e-15)


def test_design_matrix_nonzero_features() -> None:
    cell = make_simple_trimmed_cell(T=20, N=4)
    G, _ = build_design_matrix(
        [cell],
        HatBasis(0.0, 10.0, n_basis=5),
        HatBasis(0.0, 15.0, n_basis=5),
    )
    assert np.any(np.abs(G) > 1e-10)


def test_multiple_cells_stacked() -> None:
    cell1 = make_simple_trimmed_cell(T=10, N=3)
    cell2 = make_simple_trimmed_cell(T=15, N=3)
    G, _ = build_design_matrix(
        [cell1, cell2],
        HatBasis(0.0, 10.0, n_basis=4),
        HatBasis(0.0, 15.0, n_basis=4),
    )
    assert G.shape[0] == 81 + 126


def test_nan_handling() -> None:
    cell = make_simple_trimmed_cell(T=20, N=4)
    cell.chromosomes[5, :, 0] = np.nan
    G, V = build_design_matrix(
        [cell],
        HatBasis(0.0, 10.0, n_basis=5),
        HatBasis(0.0, 15.0, n_basis=5),
    )
    assert G.shape[0] < 4 * 19 * 3
    assert not np.any(np.isnan(G))
    assert not np.any(np.isnan(V))
