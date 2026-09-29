# cython: language_level=3, cdivision=True
cimport cython
import numpy as np
from scipy.spatial.distance import cdist


@cython.boundscheck(False)
@cython.wraparound(False)
def dtw_calc_accu_cost(double[:, :] C, double[:, :] D, int[:, :] steps,
                       Py_ssize_t[:, :] step_sizes_sigma, double[:] weights_mul,
                       double[:] weights_add, Py_ssize_t max_0, Py_ssize_t max_1):
    cdef Py_ssize_t n_steps = step_sizes_sigma.shape[0]
    cdef Py_ssize_t cur_n, cur_m, cur_step_idx
    cdef double cur_D, cur_C, cur_cost

    for cur_n in range(max_0, D.shape[0]):
        for cur_m in range(max_1, D.shape[1]):
            for cur_step_idx in range(n_steps):
                cur_D = D[cur_n - step_sizes_sigma[cur_step_idx, 0],
                          cur_m - step_sizes_sigma[cur_step_idx, 1]]
                cur_C = weights_mul[cur_step_idx] * C[cur_n - max_0, cur_m - max_1]
                cur_C += weights_add[cur_step_idx]
                cur_cost = cur_D + cur_C

                if cur_cost < D[cur_n, cur_m]:
                    D[cur_n, cur_m] = cur_cost
                    steps[cur_n, cur_m] = cur_step_idx

    return np.asarray(D), np.asarray(steps)


@cython.boundscheck(False)
@cython.wraparound(False)
def dtw_backtracking(int[:, :] steps, Py_ssize_t[:, :] step_sizes_sigma, bint subseq, start=None):
    cdef Py_ssize_t cur_0, cur_1, cur_step_idx
    cdef list wp = []

    if start is None:
        cur_0 = steps.shape[0] - 1
        cur_1 = steps.shape[1] - 1
    else:
        cur_0 = steps.shape[0] - 1
        cur_1 = start

    wp.append((cur_0, cur_1))

    while (subseq and cur_0 > 0) or (not subseq and (cur_0 != 0 or cur_1 != 0)):
        cur_step_idx = steps[cur_0, cur_1]
        cur_0 = cur_0 - step_sizes_sigma[cur_step_idx, 0]
        cur_1 = cur_1 - step_sizes_sigma[cur_step_idx, 1]

        if cur_0 < 0 or cur_1 < 0:
            break

        wp.append((cur_0, cur_1))

    return wp


def fill_off_diagonal(x, *, radius, value=0):
    nx, ny = x.shape
    radius = int(np.round(radius * np.min(x.shape)))
    offset = np.abs(x.shape[0] - x.shape[1])

    if nx < ny:
        idx_u = np.triu_indices_from(x, k=radius + offset)
        idx_l = np.tril_indices_from(x, k=-radius)
    else:
        idx_u = np.triu_indices_from(x, k=radius)
        idx_l = np.tril_indices_from(x, k=-radius - offset)

    x[idx_u] = value
    x[idx_l] = value


def dtw(X=None, Y=None, *, C=None, metric="euclidean", step_sizes_sigma=None,
        weights_add=None, weights_mul=None, subseq=False, backtrack=True,
        global_constraints=False, band_rad=0.25, return_steps=False):
    default_steps = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.uint32)
    default_weights_add = np.zeros(3, dtype=np.float64)
    default_weights_mul = np.ones(3, dtype=np.float64)

    if step_sizes_sigma is None:
        step_sizes_sigma = default_steps
        if weights_add is None:
            weights_add = default_weights_add
        if weights_mul is None:
            weights_mul = default_weights_mul
    else:
        if weights_add is None:
            weights_add = np.zeros(len(step_sizes_sigma), dtype=np.float64)
        if weights_mul is None:
            weights_mul = np.ones(len(step_sizes_sigma), dtype=np.float64)

        default_weights_add.fill(np.inf)
        default_weights_mul.fill(np.inf)

        step_sizes_sigma = np.concatenate((default_steps, step_sizes_sigma))
        weights_add = np.concatenate((default_weights_add, weights_add))
        weights_mul = np.concatenate((default_weights_mul, weights_mul))

    if np.any(step_sizes_sigma < 0):
        raise ValueError("step_sizes_sigma cannot contain negative values")
    if len(step_sizes_sigma) != len(weights_add):
        raise ValueError("len(weights_add) must be equal to len(step_sizes_sigma)")
    if len(step_sizes_sigma) != len(weights_mul):
        raise ValueError("len(weights_mul) must be equal to len(step_sizes_sigma)")

    if C is None and (X is None or Y is None):
        raise ValueError("If C is not supplied, both X and Y must be supplied")
    if C is not None and (X is not None or Y is not None):
        raise ValueError("If C is supplied, both X and Y must not be supplied")

    c_is_transposed = False
    C_local = False
    if C is None:
        C_local = True
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        X = np.swapaxes(X, -1, 0)
        Y = np.swapaxes(Y, -1, 0)

        X = X.reshape((X.shape[0], -1), order="F")
        Y = Y.reshape((Y.shape[0], -1), order="F")

        try:
            C = cdist(X, Y, metric=metric)
        except ValueError as exc:
            raise ValueError(
                "scipy.spatial.distance.cdist returned an error. Please provide your input in "
                "the form X.shape=(K, N) and Y.shape=(K, M). 1-dimensional sequences should "
                "be reshaped to X.shape=(1, N) and Y.shape=(1, M)."
            ) from exc

        if subseq and (X.shape[0] > Y.shape[0]):
            C = C.T
            c_is_transposed = True

    C = np.atleast_2d(C)

    if np.array_equal(step_sizes_sigma, np.array([[1, 1]])) and (C.shape[0] > C.shape[1]):
        raise ValueError("For diagonal matching: Y.shape[-1] >= X.shape[-11] (C.shape[1] >= C.shape[0])")

    max_0 = step_sizes_sigma[:, 0].max()
    max_1 = step_sizes_sigma[:, 1].max()

    if np.any(np.isnan(C)):
        raise ValueError("DTW cost matrix C has NaN values. ")

    if global_constraints:
        if not C_local:
            C = np.copy(C)
        fill_off_diagonal(C, radius=band_rad, value=np.inf)

    D = np.ones(C.shape + np.array([max_0, max_1])) * np.inf
    D[max_0, max_1] = C[0, 0]

    if subseq:
        D[max_0, max_1:] = C[0, :]

    steps = np.zeros(D.shape, dtype=np.int32)
    steps[0, :] = 1
    steps[:, 0] = 2

    D, steps = dtw_calc_accu_cost(
        np.ascontiguousarray(C, dtype=np.float64), D, steps,
        np.ascontiguousarray(step_sizes_sigma, dtype=np.intp),
        np.ascontiguousarray(weights_mul, dtype=np.float64),
        np.ascontiguousarray(weights_add, dtype=np.float64), max_0, max_1,
    )

    D = D[max_0:, max_1:]
    steps = steps[max_0:, max_1:]

    return_values = []
    if backtrack:
        step_sizes = np.ascontiguousarray(step_sizes_sigma, dtype=np.intp)
        steps_c = np.ascontiguousarray(steps, dtype=np.int32)
        if subseq:
            if np.all(np.isinf(D[-1])):
                raise ValueError(
                    "No valid sub-sequence warping path could be constructed with the given step sizes."
                )
            start = np.argmin(D[-1, :])
            _wp = dtw_backtracking(steps_c, step_sizes, subseq, int(start))
        else:
            if np.isinf(D[-1, -1]):
                raise ValueError(
                    "No valid sub-sequence warping path could be constructed with the given step sizes."
                )
            _wp = dtw_backtracking(steps_c, step_sizes, subseq)
            if _wp[-1] != (0, 0):
                raise ValueError(
                    "Unable to compute a full DTW warping path. You may want to try again with subseq=True."
                )

        wp = np.asarray(_wp, dtype=int)

        if subseq and (
            (X is not None and Y is not None and X.shape[0] > Y.shape[0])
            or c_is_transposed
            or C.shape[0] > C.shape[1]
        ):
            wp = np.fliplr(wp)
        return_values = [D, wp]
    else:
        return_values = [D]

    if return_steps:
        return_values.append(steps)

    if len(return_values) > 1:
        return tuple(return_values)
    return return_values[0]
