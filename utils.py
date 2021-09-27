"""
Файл с функциями, нужными для основных модулей
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def conjugate_quater(quat):
    quat_conj = np.zeros(4)
    quat_conj[:3] = -quat[:3]
    quat_conj[3] = quat[3]
    return quat_conj


def multiply_quaternions(quat1, quat2):
    quat_result = np.zeros(4)
    quat_result[:3] = quat1[3] * quat2[:3] + quat2[3] * quat1[:3] + np.cross(quat1[:3], quat2[:3])
    # print('check1: ', quat1[3] * quat2[3] - np.dot(quat1[:3], quat2[:3]))
    quat_result[3] = quat1[3] * quat2[3] - np.dot(quat1[:3], quat2[:3])
    # print('quat3 = ', quat_result[3])
    return quat_result


def rotate_between_quaternions(quat_init, quat_final):
    # print('check2: ', quat_final, conjugate_quater(quat_init))
    return multiply_quaternions(quat_final, conjugate_quater(quat_init))


def quat_from_angle(axis, angle):
    quat_vec = axis * np.sin(angle / 2)
    quat_scal = np.cos(angle / 2)
    return np.concatenate([quat_vec, np.array([quat_scal])])


def rotation_to_connected(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Выполняет переход вектора из инерциальной системы отсчета в связную
    Args:
        vector - координаты вектора в начальной СО
        quat - кватернион поворота связной системы отсчета относительно начальной
    Return:
        Координаты вектора в связной системе отсчета
    """
    rotate = R.from_quat(quat)
    return rotate.inv().apply(vector)


# k_w_and_q = [[0.7, 0.01], [0.65, 0.008], [0.75, 0.02], [0.6, 0.02]]
#
# for i in k_w_and_q:
#     a, b = i[0], i[1]
#     print(a, b)