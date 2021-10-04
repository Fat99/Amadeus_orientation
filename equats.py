import numpy as np
from numpy import linalg as LA
from ext_moments import mom_gravit


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


def euler_and_kinematic(y, t, j_tenzor, nu, control_mom):
    r, v = y[:3], y[3:6]
    w_abs, quat = y[6:9], y[9:]
    dydt = np.zeros(13)
    dydt[:3] = v
    dydt[3:6] = - nu / np.linalg.norm(r) ** 3 * r

    mom_imp = np.matmul(j_tenzor, w_abs)

    mom_gr = mom_gravit(quat, r, j_tenzor, nu)

    dydt[6:9] = mom_gr - np.cross(w_abs, mom_imp) + control_mom
    dydt[6:9] = np.matmul(LA.inv(j_tenzor), dydt[6:9])

    dydt[9:12] = (quat[3] * w_abs + np.cross(quat[:3], w_abs)) / 2
    dydt[12] = -np.dot(w_abs, quat[:3]) / 2

    return dydt
