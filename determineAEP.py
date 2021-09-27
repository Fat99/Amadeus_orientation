import numpy as np

r0 = 1.0e+07 * np.array([1.715573066233160, 2.289071031477311, 0.892592547997959])
V0 = 1.0e+04 * np.array([1.701250362854970, -1.101411738217119, -0.469847228124423])
mug = 6.809e15


def determineAEP(r, V, mug):
    h = np.cross(r, V)
    nhat = np.cross(np.array([0, 0, 1]), h)

    evec = ((np.linalg.norm(V) ** 2 - mug / np.linalg.norm(r)) * r - np.dot(r, V) * V) / mug
    e = np.linalg.norm(evec)

    energy = np.linalg.norm(V) ** 2 / 2 - mug / np.linalg.norm(r)

    if np.abs(e - 1.0) > 1e-10:
        a = -mug / (2 * energy)
        p = a * (1 - e ** 2)
    else:
        p = np.linalg.norm(h) ** 2 / mug
        a = np.inf
    return a, e, p


def w_ref_t(p, t, t_0, r):
    return np.sqrt(mug * p) / (np.linalg.norm(r) ** 2)


a, e, p = determineAEP(r0, V0, mug)

