import numpy as np
from scipy.integrate import odeint
from equats import traj_and_speed

T = 5400  # тут фигня
M_0 = 0

# Нептун
nu = 6.809e15

# Земля
# nu = 398600.4415 * 10 ** 9


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


r0 = 1.0e+07 * np.array([1.715573066233160, 2.289071031477311, 0.892592547997959])
v0 = 1.0e+04 * np.array([1.701250362854970, -1.101411738217119, -0.469847228124423])

a, e, p = determineAEP(r0, v0, nu)


def M(t, t_0):
    return M_0 + 2 * np.pi / T * (t - t_0)


def E_anamal(e, t, t_0):
    E = M(t, t_0)
    # print(M(t, t_0))
    # print('E', E)
    while E - e * np.sin(E) - M(t, t_0) > 10 ** (-4):
        E = e * np.sin(E) + M(t, t_0)

    return E


def teta(e, t, t_0):
    return 2 * np.arctan(np.tan(E_anamal(e, t, t_0) / 2) * np.sqrt((1 + e) / (1 - e)))


# def w_ref_t(e, t, t_0, r):
#     return np.sqrt(nu * (1 + e * np.cos(teta(e, t, t_0)))) / np.linalg.norm(r) ** (3 / 2)


def w_ref_t(p, t, t_0, r):
    return np.sqrt(nu * p) / (np.linalg.norm(r) ** 2)


def w_ref(p, t, t_0, r):
    return np.array([0, 0, w_ref_t(p, t, t_0, r)])


def w_ref_dif(r, v, p, t, t_0):
    drdt = np.dot(r, v) / np.linalg.norm(r)
    return -2 * w_ref(p, t, t_0, r) * drdt / np.linalg.norm(r)
