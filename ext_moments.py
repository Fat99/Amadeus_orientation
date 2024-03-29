import numpy as np
from utils import rotation_to_connected


def mom_gravit(quat: np.ndarray, vector: np.ndarray, j_spacecr: np.ndarray,
               gravitational_parameter):

    vector_norm = vector / np.linalg.norm(vector)
    e_r_sv = rotation_to_connected(quat, vector_norm)
    moment_gr = 3 * gravitational_parameter / np.linalg.norm(vector) ** 3 \
                * np.cross(e_r_sv, np.matmul(j_spacecr, e_r_sv))

    return moment_gr
