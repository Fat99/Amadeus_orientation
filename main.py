import numpy as np
from scipy.integrate import odeint
from equats import traj_and_speed, w_rel_and_quat
import matplotlib.pyplot as plt
from calculate_w_ref import w_ref, w_ref_dif, determineAEP
from ext_moments import mom_gravit
from utils import rotation_to_connected


t_0, nu = 0, 6.809e15

t_final = 500000

t = np.linspace(0, t_final, t_final + 1)  # для Нептуна

r0 = 1.0e+07 * np.array([1.715573066233160, 2.289071031477311, 0.892592547997959])
v0 = 1.0e+04 * np.array([1.701250362854970, -1.101411738217119, -0.469847228124423])
r_and_v0 = np.concatenate([r0, v0])

a, e, p = determineAEP(r0, v0, nu)

r_and_v = odeint(traj_and_speed, r_and_v0, t, args=(nu,))

r, v = np.zeros((t.size, 3)), np.zeros((t.size, 3))
r[:], v[:] = r_and_v[:, :3], r_and_v[:, 3:]

j_tenzor_t = np.array([[3348, 0, 0],
                       [0, 1836, 0],
                       [0, 0, 4548]])

j_tenzor_c = np.array([[6216, 0, 0],
                       [0, 6582, 0],
                       [0, 0, 5509]])

a_t = 0.73
j_tenzor = j_tenzor_c + a_t * j_tenzor_t

init_angle = 5 * np.pi / 180
w_rel_init, quat0 = np.array([0, 0, 0]), np.array([0, 0, np.sin(init_angle / 2), np.cos(init_angle / 2)])
w_rel_and_quat0 = np.concatenate([w_rel_init, quat0])

k_w, k_q = 3, 0.04

w_rel_and_quat = odeint(w_rel_and_quat, w_rel_and_quat0, t, args=(j_tenzor, k_w, k_q))

w_rel, quat = np.zeros((t.size, 3)), np.zeros((t.size, 4))
w_rel[:], quat[:] = w_rel_and_quat[:, :3], w_rel_and_quat[:, 3:]

M_ctrl, w_abs, mom_imp = np.zeros((t.size, 3)), np.zeros((t.size, 3)), np.zeros((t.size, 3))

for i in range(0, t.size):
    w_abs[i] = w_rel[i] + rotation_to_connected(quat[i], w_ref(p, t[i], t_0, r[i]))

w_ref_t = np.zeros((t.size, 3))

for i in range(0, t.size):
    mom_imp[i] = np.matmul(j_tenzor, w_abs[i])

    M_ctrl[i] = -mom_gravit(quat[i], r[i], j_tenzor, nu) + np.cross(w_abs[i], mom_imp[i]) \
                - np.matmul(j_tenzor, np.cross(w_rel[i], rotation_to_connected(quat[i], w_ref(p, t[i], t_0, r[i])))) \
                + np.matmul(j_tenzor, rotation_to_connected(quat[i], w_ref_dif(r[i], v[i], p, t[i], t_0))) \
                - k_w * w_rel[i] - k_q * quat[i, :3]

    w_ref_t[i] = w_ref(p, t[i], t_0, r[i])

plt.plot(t / 3600, M_ctrl[:, 0], label='M_ctrl_x')
plt.plot(t / 3600, M_ctrl[:, 1], label='M_ctrl_y')
plt.plot(t / 3600, M_ctrl[:, 2], label='M_ctrl_z')
plt.legend(loc='best')
plt.title('График зав-ти управляющего момента M_ctrl от времени')
plt.xlabel('t, час')
plt.ylabel('M_ctrl, Н*м')
plt.grid()
plt.show()

plt.plot(t / 3600, w_rel[:, 0], label='w_rel_x')
plt.plot(t / 3600, w_rel[:, 1], label='w_rel_y')
plt.plot(t / 3600, w_rel[:, 2], label='w_rel_z')
plt.legend(loc='best')
plt.title('График зав-ти w_rel от времени')
plt.xlabel('t, час')
plt.ylabel('w_rel, c^-1')
plt.grid()
plt.show()

plt.plot(t / 3600, quat[:, 0], label='quat_x')
plt.plot(t / 3600, quat[:, 1], label='quat_y')
plt.plot(t / 3600, quat[:, 2], label='quat_z')
plt.plot(t / 3600, quat[:, 3], label='quat_scalar')
plt.legend(loc='best')
plt.title('График зав-ти quat от времени')
plt.xlabel('t, час')
plt.ylabel('quat')
plt.grid()
plt.show()

plt.plot(t / 3600, w_abs[:, 0], label='w_abs_x')
plt.plot(t / 3600, w_abs[:, 1], label='w_abs_y')
plt.plot(t / 3600, w_abs[:, 2], label='w_abs_z')
plt.legend(loc='best')
plt.title('График зав-ти w_abs от времени')
plt.xlabel('t, час')
plt.ylabel('w_abs, c^-1')
plt.grid()
plt.show()
#
plt.plot(t / 3600, w_ref_t[:, 0], label='w_ref_x')
plt.plot(t / 3600, w_ref_t[:, 1], label='w_ref_y')
plt.plot(t / 3600, w_ref_t[:, 2], label='w_ref_z')
plt.legend(loc='best')
plt.title('График зав-ти w_ref от времени')
plt.xlabel('t, час')
plt.ylabel('w_ref, c^-1')
plt.grid()
plt.show()

# уравнение на моменты импульса маховиков
# def h_machs(y, t):
#     h_mach = y[:3]
#     dh_dt = np.zeros(3)
#     # print('mom_ctrl_check = ', M_ctrl[int(t / t_final * t_len - 1), :])
#     # print('cross = ', np.cross(w_abs[int(t / t_final * t_len - 1), :], h_mach))
#     print(int(t / t_final * t_len) - 1)
#     dh_dt[:] = -M_ctrl[int(t / t_final * t_len) - 1, :] - np.cross(w_abs[int(t / t_final * t_len) - 1, :], h_mach)
#     return dh_dt
#
#
#
# h_mach_init = np.zeros(3)
#
# h_mach = odeint(h_machs, h_mach_init, t)

# t_len = t.size
t_len, dt = t.size, t[1] - t[0]
h_mach_init = np.zeros(3)
H_mach = np.zeros((t_len, 3))
h_mach = h_mach_init
for i in range(0, t.size):
    dhdt = -M_ctrl[i] - np.cross(w_abs[i], h_mach)
    h_mach = h_mach + dhdt * dt
    H_mach[i][:] = h_mach

plt.plot(t / 3600, H_mach[:, 0], label='h_mach_x')
plt.plot(t / 3600, H_mach[:, 1], label='h_mach_y')
plt.plot(t / 3600, H_mach[:, 2], label='h_mach_z')
plt.legend(loc='best')
plt.title('График зав-ти h_mach от времени')
plt.xlabel('t, час')
plt.ylabel('H_mach, м^2·кг/с')
plt.grid()
plt.show()

moment_machoviks = np.zeros((t.size, 3))
for i in range(1, t.size):
    moment_machoviks[i] = -(H_mach[i] - H_mach[i - 1]) / dt

plt.plot(t / 3600, moment_machoviks[:, 0], label='moment_machoviks_x')
plt.plot(t / 3600, moment_machoviks[:, 1], label='moment_machoviks_y')
plt.plot(t / 3600, moment_machoviks[:, 2], label='moment_machoviks_z')
plt.legend(loc='best')
plt.title('График зав-ти момента, создаваемого маховиками, от времени')
plt.xlabel('t, час')
plt.ylabel('moment_machoviks, H*m')
plt.grid()
plt.show()

j_tenzor = np.array([[3348, 0, 0],
                     [0, 1836, 0],
                     [0, 0, 4548]])

# k_w_and_q = [[0.7, 0.01], [0.65, 0.008], [0.75, 0.02], [0.6, 0.02]]
w_rel_init, quat0 = np.array([0, 0, 0]), np.array([0, 0, 0, 1])
w_rel_and_quat0 = np.concatenate([w_rel_init, quat0])

# k_w, k_q = 0.025, 0.0015
k_w, k_q = 1.2, 0.01


w_rel_and_quat = odeint(w_rel_and_quat, w_rel_and_quat0, t, args=(j_tenzor, k_w, k_q))

w_rel, quat = np.zeros((t.size, 3)), np.zeros((t.size, 4))
w_rel[:], quat[:] = w_rel_and_quat[:, :3], w_rel_and_quat[:, 3:]

M_ctrl, w_abs, mom_imp = np.zeros((t.size, 3)), np.zeros((t.size, 3)), np.zeros((t.size, 3))

for i in range(0, t.size):
    w_abs[i] = w_rel[i] + rotation_to_connected(quat[i], w_ref(p, t[i], t_0, r[i]))

w_ref_t = np.zeros((t.size, 3))

for i in range(0, t.size):
    mom_imp[i] = np.matmul(j_tenzor, w_abs[i])

    M_ctrl[i] = -mom_gravit(quat[i], r[i], j_tenzor, nu) + np.cross(w_abs[i], mom_imp[i]) \
                - np.matmul(j_tenzor, np.cross(w_rel[i], rotation_to_connected(quat[i], w_ref(p, t[i], t_0, r[i])))) \
                + np.matmul(j_tenzor, rotation_to_connected(quat[i], w_ref_dif(r[i], v[i], p, t[i], t_0))) \
                - k_w * w_rel[i] - k_q * quat[i, :3]

    w_ref_t[i] = w_ref(p, t[i], t_0, r[i])


# графики
plt.plot(t / 3600, M_ctrl[:, 0], label='M_ctrl_x')
plt.plot(t / 3600, M_ctrl[:, 1], label='M_ctrl_y')
plt.plot(t / 3600, M_ctrl[:, 2], label='M_ctrl_z')
plt.legend(loc='best')
plt.title('График зав-ти управляющего момента M_ctrl от времени')
plt.xlabel('t, час')
plt.ylabel('M_ctrl, Н*м')
plt.grid()
plt.show()

# plt.plot(t / 3600, w_rel[:, 0], label='w_rel_x')
# plt.plot(t / 3600, w_rel[:, 1], label='w_rel_y')
# plt.plot(t / 3600, w_rel[:, 2], label='w_rel_z')
# plt.legend(loc='best')
# plt.title('График зав-ти w_rel от времени')
# plt.xlabel('t, час')
# plt.ylabel('w_rel, c^-1')
# plt.grid()
# plt.show()


# plt.plot(t / 3600, quat[:, 0], label='quat_x')
# plt.plot(t / 3600, quat[:, 1], label='quat_y')
# plt.plot(t / 3600, quat[:, 2], label='quat_z')
# plt.plot(t / 3600, quat[:, 3], label='quat_scalar')
# plt.legend(loc='best')
# plt.title('График зав-ти quat от времени')
# plt.xlabel('t, час')
# plt.ylabel('quat')
# plt.grid()
# plt.show()


plt.plot(t / 3600, w_abs[:, 0], label='w_abs_x')
plt.plot(t / 3600, w_abs[:, 1], label='w_abs_y')
plt.plot(t / 3600, w_abs[:, 2], label='w_abs_z')
plt.legend(loc='best')
plt.title('График зав-ти w_abs от времени')
plt.xlabel('t, час')
plt.ylabel('w_abs, c^-1')
plt.grid()
plt.show()

plt.plot(t / 3600, w_ref_t[:, 0], label='w_ref_x')
plt.plot(t / 3600, w_ref_t[:, 1], label='w_ref_y')
plt.plot(t / 3600, w_ref_t[:, 2], label='w_ref_z')
plt.legend(loc='best')
plt.title('График зав-ти w_ref от времени')
plt.xlabel('t, час')
plt.ylabel('w_ref, c^-1')
plt.grid()
plt.show()


# уравнение на моменты импульса маховиков
# def h_machs(y, t):
#     h_mach = y[:3]
#     dh_dt = np.zeros(3)
#     # print('mom_ctrl_check = ', M_ctrl[int(t / t_final * t_len - 1), :])
#     # print('cross = ', np.cross(w_abs[int(t / t_final * t_len - 1), :], h_mach))
#     print(int(t / t_final * t_len) - 1)
#     dh_dt[:] = -M_ctrl[int(t / t_final * t_len) - 1, :] - np.cross(w_abs[int(t / t_final * t_len) - 1, :], h_mach)
#     return dh_dt
#
#
#
# h_mach_init = np.zeros(3)
#
# h_mach = odeint(h_machs, h_mach_init, t)

# t_len = t.size
t_len, dt = t.size, t[1] - t[0]
h_mach_init = np.zeros(3)
H_mach = np.zeros((t_len, 3))
h_mach = h_mach_init
for i in range(0, t.size):
    dhdt = -M_ctrl[i] - np.cross(w_abs[i], h_mach)
    h_mach = h_mach + dhdt * dt
    H_mach[i][:] = h_mach

plt.plot(t / 3600, H_mach[:, 0], label='h_mach_x')
plt.plot(t / 3600, H_mach[:, 1], label='h_mach_y')
plt.plot(t / 3600, H_mach[:, 2], label='h_mach_z')
plt.legend(loc='best')
plt.title('График зав-ти h_mach от времени')
plt.xlabel('t, час')
plt.ylabel('H_mach, м^2·кг/с')
plt.grid()
plt.show()
