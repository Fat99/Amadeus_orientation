import numpy as np
from scipy.integrate import odeint
from equats import traj_and_speed, w_rel_and_quat, euler_and_kinematic
import matplotlib.pyplot as plt
from calculate_w_ref import w_ref, w_ref_dif, determineAEP
from ext_moments import mom_gravit
from utils import rotate_between_quaternions, multiply_quaternions
from scipy.spatial.transform import Rotation as R

t_0, nu = 0, 6.809e15

# t_final = 8000
t_final = 800
t = np.linspace(0, t_final, 5 * t_final + 1)  # для Нептуна
dt = t[1] - t[0]

t_len = t.size

# задание начального и конечного кватерниона при 1-ом повороте

angle1 = 0.074358628256879
quat_init1, quat_final1 = np.array([0, 0, 0, 1]), np.array([0, 0, np.sin(angle1 / 2), np.cos(angle1 / 2)])

# задание начального и конечного кватерниона при 1-ом повороте
angle2 = 0.015383401780595
quat_rotate = np.array([0, np.sin(angle2 / 2), 0, np.cos(angle2 / 2)])
quat_init2 = np.array([0, 0, np.sin(angle1 / 2), np.cos(angle1 / 2)])
quat_final2 = multiply_quaternions(quat_rotate, quat_init2)

# quat_rotate = np.array([0, np.sin(angle2 / 2), 0, np.cos(angle2 / 2)])
# ch = multiply_quaternions(quat_rotate, quat_init2)
# print(ch)
# print(rotate_between_quaternions(quat_init2, ch))


# рядом с апоцентром
# r0 = 1.0e+07 * np.array([1.715573066233160, 2.289071031477311, 0.892592547997959])
# v0 = 1.0e+04 * np.array([1.701250362854970, -1.101411738217119, -0.469847228124423])

# рядом с перицентром
r0 = np.array([-3.52839483e+08, -4.51136343e+08, -1.75679247e+08])
v0 = np.array([-766.20297685, 659.56780191, 277.32758496])
r_and_v0 = np.concatenate([r0, v0])

a, e, p = determineAEP(r0, v0, nu)

# r_and_v = odeint(traj_and_speed, r_and_v0, t, args=(nu,))
#
# r, v = np.zeros((t.size, 3)), np.zeros((t.size, 3))
# r[:], v[:] = r_and_v[:, :3], r_and_v[:, 3:]

j_tenzor = 0.5 * np.array([[3348, 0, 0],
                           [0, 1836, 0],
                           [0, 0, 4548]])

w_abs_init = np.zeros(3)

init_data = np.concatenate([r_and_v0, w_abs_init, quat_init1])

# непосредтсвенно сами коэффициенты пид
k_prop, k_dif = 300, 200

# simulate with ODEINT

quat_rotate = np.zeros((t_len, 4))
quat_rotate[0] = quat_init1

# check rotate
# data = odeint(euler_and_kinematic, init_data, t, args=(j_tenzor, nu))
# plt.plot(t[:] / 3600, data[:, 9], label='quat_x')
# plt.plot(t[:] / 3600, data[:, 10], label='quat_y')
# plt.plot(t[:] / 3600, data[:, 11], label='quat_z')
# plt.plot(t[:] / 3600, data[:, 12], label='quat_scalar')
# plt.legend(loc='best')
# plt.show()


# начальная ошибка диф члена
error_dif = np.zeros(3)
M_ctrl = np.zeros((t_len, 3))
W_abs = np.zeros((t_len, 3))

# проверка
# t2 = 50
# t2 = np.linspace(0, t2, t2 + 1)
# quat_check = np.zeros((t2.size, 4))
# control_mom = np.array([0, 0, 2])
# data2 = odeint(euler_and_kinematic, init_data, t2, args=(j_tenzor, nu, control_mom))
# quat_check[:] = data2[:, 9:]
# plt.plot(t2, quat_check[:, 0])
# plt.plot(t2, quat_check[:, 1])
# plt.plot(t2, quat_check[:, 2])
# plt.plot(t2, quat_check[:, 3])
# plt.show()

for i in range(t_len - 1):
    # тут будет расчет ошибок(интеграл, диф и пропорционал) (или после ограничений на ошибку по выдаваемому моменту)

    r1 = R.from_quat(rotate_between_quaternions(quat_rotate[i], quat_final1))
    error_proportion = r1.as_rotvec()
    # print('error_prop = ', error_proportion)
    if i > 0:
        r2 = R.from_quat(rotate_between_quaternions(quat_rotate[i],
                                                    quat_rotate[i - 1]))

        error_dif = r2.as_rotvec() / dt

    control_mom = k_prop * error_proportion + k_dif * error_dif

    # тут будут ограничения на ошибку
    if np.amax(np.fabs(control_mom)) > 0.025:
        max_index = np.argmax(np.fabs(control_mom))
        dif = np.amax(np.fabs(control_mom)) / 0.025
        control_mom = control_mom / dif

    # print('control_mom = ', control_mom)
    M_ctrl[i + 1] = control_mom

    data = odeint(euler_and_kinematic, init_data, [0, dt], args=(j_tenzor, nu, control_mom))
    r, v, w_abs, quat = data[-1, 0:3], data[-1, 3:6], data[-1, 6:9], data[-1, 9:]  # take the last value
    init_data = np.concatenate([r, v, w_abs, quat])
    quat_rotate[i + 1] = quat
    W_abs[i + 1] = w_abs

# отрисовка
# кватернион
plt.plot(t[:t_len - 1] / 3600, quat_rotate[:t_len - 1, 0], label='quat_x')
plt.plot(t[:t_len - 1] / 3600, quat_rotate[:t_len - 1, 1], label='quat_y')
plt.plot(t[:t_len - 1] / 3600, quat_rotate[:t_len - 1, 2], label='quat_z')
plt.plot([0, t[t_len - 2] / 3600], [quat_final2[2], quat_final2[2]], label='quat_target')

plt.legend(loc='best')
plt.title('График зав-ти quat от времени')
plt.xlabel('t, час')
plt.ylabel('quat')
plt.grid()
plt.show()

# управляющий момент
plt.plot(t / 3600, M_ctrl[:, 0], label='M_ctrl_x')
plt.plot(t / 3600, M_ctrl[:, 1], label='M_ctrl_y')
plt.plot(t / 3600, M_ctrl[:, 2], label='M_ctrl_z')
plt.legend(loc='best')
plt.title('График зав-ти управляющего момента M_ctrl от времени')
plt.xlabel('t, час')
plt.ylabel('M_ctrl, Н*м')
plt.grid()
plt.show()

# момент импульса и силы маховиков
h_mach_init = np.zeros(3)
H_mach = np.zeros((t_len, 3))
h_mach = h_mach_init
for i in range(0, t.size):
    dhdt = -M_ctrl[i] - np.cross(W_abs[i], h_mach)
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
