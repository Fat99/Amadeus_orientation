import numpy as np
from scipy.integrate import odeint
from equats import traj_and_speed, w_rel_and_quat
import matplotlib.pyplot as plt
from calculate_w_ref import w_ref, w_ref_dif, determineAEP
from ext_moments import mom_gravit
from utils import rotation_to_connected

# для Нептуна
t_0, nu = 0, 6.809e15

# для Земли
# t_0, e, nu = 0, 0.01, 398600.4415 * 10 ** 9

# t_final = 250000
t_final = 500000

t = np.linspace(0, t_final, t_final + 1)  # для Нептуна
# t = np.linspace(0, 5400, 5401)  # для Земли

# Для Нептуна
r0 = 1.0e+07 * np.array([1.715573066233160, 2.289071031477311, 0.892592547997959])
v0 = 1.0e+04 * np.array([1.701250362854970, -1.101411738217119, -0.469847228124423])
r_and_v0 = np.concatenate([r0, v0])

a, e, p = determineAEP(r0, v0, nu)

# Для Земли
# r_and_v0 = np.array([6800000, 0, 0, 0, 7600, 0])

r_and_v = odeint(traj_and_speed, r_and_v0, t, args=(nu,))

r, v = np.zeros((t.size, 3)), np.zeros((t.size, 3))
r[:], v[:] = r_and_v[:, :3], r_and_v[:, 3:]


j_tenzor = np.array([[3348, 0, 0],
                     [0, 1836, 0],
                     [0, 0, 4548]])

# k_w_and_q = [[0.7, 0.01], [0.65, 0.008], [0.75, 0.02], [0.6, 0.02]]
w_rel_init, quat0 = np.array([0, 0, 0]), np.array([0, 0, 0, 1])
