import numpy as np
from numpy import linalg as LA


def traj_and_speed(y, t, nu):
    r, v = y[:3], y[3:]
    dydt = np.zeros(6)
    dydt[:3] = v
    dydt[3:] = - nu / np.linalg.norm(r) ** 3 * r
    return dydt


def w_rel_and_quat(y, t, j_tenzor, k_w, k_q):
    w_rel, quat = y[:3], y[3:]
    dydt = np.zeros(7)
    dydt[:3] = -(k_w * w_rel + k_q * quat[:3])
    dydt[:3] = np.matmul(LA.inv(j_tenzor), dydt[:3])
    dydt[3:6] = (quat[3] * w_rel + np.cross(quat[:3], w_rel)) / 2
    dydt[6] = -np.dot(w_rel, quat[:3]) / 2
    return dydt
