from __future__ import division

import numpy as np
from numba import jit


@jit(nopython=True)
def searchsorted(a, v):
    lo = -1
    hi = len(a)
    while(lo < hi-1):
        m = (lo + hi) // 2
        if v < a[m]:
            hi = m
        else:
            lo = m
    return hi


@jit
def _replicate_markov_chain(P_cdfs, T, num_reps, init_states,
                            random_state=None):
    """
    Main body of MarkovChain.replicate.

    Parameters
    ----------
    P_cdfs : ndarray(float, ndim=2)
        Array containing as rows the CDFs of the state transition.

    num_reps : scalar(int)
        Number of replication.

    init : ndarray(int, ndim=1)
        Array of length num_reps containing the initial states.

    random_state : scalar(int) or np.random.RandomState,
                   optional(default=None)
        Random seed (int) or np.random.RandomState instance.

    Returns
    -------
    out : ndarray(int, ndim=1)
        Array containing the num_reps observations of the state at
        time T.

    Notes
    -----
    This routine is jit-complied if the module Numba is vailable.

    """
    out = np.empty(num_reps, dtype=int)

    if random_state is None or isinstance(random_state, int):
        _random_state = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        _random_state = random_state
    else:
        raise ValueError
    u = _random_state.random_sample(size=(num_reps, T))
    # u = np.random.random(size=(num_reps, T))

    for i in range(num_reps):
        x_current = init_states[i]
        for t in range(T):
            x_next = searchsorted(P_cdfs[x_current], u[i, t])
            x_current = x_next
        out[i] = x_current

    return out


if __name__ == '__main__':
    P = [[0.4, 0.6], [0.2, 0.8]]
    P_cdfs = np.cumsum(P, axis=-1)
    T = 100
    num_reps = 10**3
    init_states = np.zeros(num_reps, dtype=int)
    prng = np.random.RandomState(0)
    X_Ts = _replicate_markov_chain(P_cdfs, T, num_reps, init_states,
                                   random_state=prng)
    print X_Ts.sum() / num_reps
