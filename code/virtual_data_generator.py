import os
import pickle
import random
import numpy as np
import matplotlib
from rToF_matching import *

# ============================================================
# read file saved in judge_trajectories_information
# ============================================================
POINT_PATH = './trajectories_detection/trajectories_infor_save.pkl'
SAVE_ROOT = './generate_data/'
N_GENERATE = 2000
CLOSE_THRESHOLD = 0.8
GRID_INTERVAL_M = 0.5
NEARBY_SEARCH_RADIUS_M = 0.5
TRAJECTORY_DT_S = 0.1
STATIONARY_DURATION_S = 2.0
STATIONARY_FRAMES = int(STATIONARY_DURATION_S / TRAJECTORY_DT_S)
INVALID_AOA = 1000
MU_V_MPS = 1.0
SIGMA_V_MPS = 0.1
V_MIN_MPS = 0.8
V_MAX_MPS = 1.2
DELTA_V_MPS = 0.05
DELTA_OMEGA_RADPS = 0.05
SIGMA_THETA_DEG = 3.0
SIGMA_T_NS = 1.0
THETA_RES_DEG = 20.0
P_C = 0.7
P_A = 0.5
P_F = 0.05
ABNORMAL_AOA_RANGE_DEG = (-90.0, 90.0)
ABNORMAL_TOF_RANGE_NS = (0, 60.0)


def sample_motion_speed_mps():
    while True:
        v = np.random.normal(MU_V_MPS, SIGMA_V_MPS)
        if V_MIN_MPS <= v <= V_MAX_MPS:
            return float(v)


def perturb_motion_speed_mps(v):
    v = v + np.random.uniform(-DELTA_V_MPS, DELTA_V_MPS)
    return float(np.clip(v, V_MIN_MPS, V_MAX_MPS))


def mkdir(path):
    path = path.strip().rstrip("\\")
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def calculate_angle_diff(point1, origin1, point2, origin2):
    vector1 = np.array(point1) - np.array(origin1)
    vector2 = np.array(point2) - np.array(origin2)
    angle1 = np.arctan2(vector1[1], vector1[0])
    angle2 = np.arctan2(vector2[1], vector2[0])
    angle_diff = np.degrees(angle2 - angle1)
    angle_diff = np.abs(angle_diff)
    if angle_diff > 110:
        angle_diff = 360 - angle_diff
    return angle_diff


def calculate_distance_between_points(point1, point2):
    return np.linalg.norm(np.array(point2) - np.array(point1))


def calculate_rtof_of_polyline(points):
    total_distance = 0.0
    for i in range(len(points) - 1):
        total_distance += calculate_distance_between_points(points[i], points[i + 1])
    direct_distance = calculate_distance_between_points(points[0], points[-1])
    generate_rtof = (total_distance - direct_distance) / 0.1
    return float(int(generate_rtof))


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def find_nearby_points(points, apoint, threshold=NEARBY_SEARCH_RADIUS_M):
    nearby_points = [point for point in points if distance(point, apoint) <= threshold]
    if len(nearby_points) == 0:
        raise ValueError(
            f'No grid point is found near {apoint} with threshold={threshold}. '
            'Please increase the threshold.'
        )
    return nearby_points


def round_to_nearest_half_integer(radius):
    floor_radius = np.floor(radius)
    candidates = [floor_radius, floor_radius + 0.5, floor_radius + 1.0]
    return min(candidates, key=lambda x: abs(radius - x))


def calculate_coordinate_seq(AoA_list_1, AoA_list_2, ToF_list_1, ToF_list_2):
    if not (len(AoA_list_1) == len(AoA_list_2) == len(ToF_list_1) == len(ToF_list_2)):
        raise ValueError(
            'AoA_list_1, AoA_list_2, ToF_list_1 and ToF_list_2 must have the same length.'
        )

    coordinate_seq = []

    for i in range(len(AoA_list_1)):
        combine_cors = get_position_combine(
            AoA_list_1[i], ToF_list_1[i],
            AoA_list_2[i], ToF_list_2[i]
        )
        if combine_cors is None:
            combine_cors = []

        coordinate_seq.append(combine_cors)

    return coordinate_seq



LINE_LABELS = {'horizontal_line', 'vertical_line', 'bias_line'}


def infer_scene_type(trajectories_information, close_threshold=CLOSE_THRESHOLD):
    labels = [item[1] for item in trajectories_information]
    n_person = len(trajectories_information)

    if n_person == 2 and all(label == 'circle' for label in labels):
        return 'circle'

    if n_person == 3:
        broken_num = sum(label == 'broken_line' for label in labels)
        straight_num = sum(label in LINE_LABELS for label in labels)
        if broken_num == 1 and straight_num == 2:
            return 'W'

    if n_person == 2:
        broken = [item for item in trajectories_information if item[1] == 'broken_line']
        straight = [item for item in trajectories_information if item[1] in LINE_LABELS]

        if len(broken) == 2:
            return 'line'

        if len(broken) == 1 and len(straight) == 1:
            poly = broken[0]
            line = straight[0]

            poly_start = np.array(poly[2], dtype=float)
            poly_end = np.array(poly[3], dtype=float)
            line_start = np.array(line[2], dtype=float)
            line_end = np.array(line[3], dtype=float)

            d1 = np.linalg.norm(poly_start - line_end)
            d2 = np.linalg.norm(poly_end - line_start)

            if d1 < close_threshold and d2 < close_threshold:
                return 'triangle'
            else:
                return 'N'

    raise ValueError(
        'Cannot infer scene type from trajectories_information. '
        f'labels={labels}. You can set SCENE_TYPE manually.'
    )


def reorder_two_person_information(info, scene_type):
    if scene_type in ('N', 'triangle'):
        poly = next(item for item in info if item[1] == 'broken_line')
        line = next(item for item in info if item[1] in LINE_LABELS)
        return poly, line
    return info[0], info[1]


def reorder_w_information(info):
    poly = next(item for item in info if item[1] == 'broken_line')
    lines = [item for item in info if item[1] in LINE_LABELS]

    if len(lines) != 2:
        raise ValueError('W scene must contain one broken line and two straight lines.')

    poly_start = np.array(poly[2], dtype=float)
    poly_end = np.array(poly[3], dtype=float)

    cost_a = np.linalg.norm(np.array(lines[0][3], dtype=float) - poly_start) + \
             np.linalg.norm(np.array(lines[1][2], dtype=float) - poly_end)
    cost_b = np.linalg.norm(np.array(lines[1][3], dtype=float) - poly_start) + \
             np.linalg.norm(np.array(lines[0][2], dtype=float) - poly_end)

    if cost_a <= cost_b:
        return lines[0], poly, lines[1]
    return lines[1], poly, lines[0]


# ============================================================
# Motion trajectory generators
# ============================================================
def generate_straight_trajectory(
        target_trajectory,
        num_steps):
    generate_trajectory_single = []
    velocity = sample_motion_speed_mps()

    start_point = np.array(target_trajectory[0], dtype=float)
    end_point = np.array(target_trajectory[1], dtype=float)

    direction = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
    vector_direction = end_point - start_point
    vector_direction = vector_direction / np.linalg.norm(vector_direction)

    point_position = np.array([start_point[0], start_point[1], direction, velocity], dtype=float)
    generate_trajectory_single.append(point_position.copy())

    for _ in range(num_steps):
        vector_to_end = end_point - point_position[:2]
        if np.dot(vector_to_end, vector_direction) < 0.1:
            break

        point_position[0] += velocity * TRAJECTORY_DT_S * np.cos(direction)
        point_position[1] += velocity * TRAJECTORY_DT_S * np.sin(direction)

        direction = np.arctan2(end_point[1] - point_position[1], end_point[0] - point_position[0])
        velocity = perturb_motion_speed_mps(velocity)

        point_position[2] = direction
        point_position[3] = velocity
        generate_trajectory_single.append(point_position.copy())

    generate_trajectory_true = []
    num_steps_true = len(generate_trajectory_single)
    x_start, y_start = np.array(target_trajectory[0], dtype=float)
    x_end, y_end = np.array(target_trajectory[1], dtype=float)
    x_step = (x_end - x_start) / num_steps_true
    y_step = (y_end - y_start) / num_steps_true

    for index in range(num_steps_true):
        x = x_start + x_step * index
        y = y_start + y_step * index
        generate_trajectory_true.append(np.array([x, y]))

    return np.array(generate_trajectory_single), np.array(generate_trajectory_true)


def generate_polyline_trajectory(
        target_trajectory,
        num_steps,
        turn_num_range=(2, 5)):
    generate_trajectory_single = []
    velocity = sample_motion_speed_mps()

    start_point = np.array(target_trajectory[0], dtype=float)
    first_turn_point = np.array(target_trajectory[1], dtype=float)
    end_point = first_turn_point.copy()

    turn_around_start_step = 0
    turn_around_end_step = 0

    direction = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
    vector_direction = end_point - start_point
    vector_direction = vector_direction / np.linalg.norm(vector_direction)
    direction_second = None

    point_position = np.array([start_point[0], start_point[1], direction, velocity], dtype=float)
    generate_trajectory_single.append(point_position.copy())

    for step in range(num_steps):
        vector_to_end = end_point - point_position[:2]

        if np.dot(vector_to_end, vector_direction) < 0.1:
            if direction_second is None:
                end_point = np.array(target_trajectory[2], dtype=float)

                turn_num = np.random.randint(turn_num_range[0], turn_num_range[1])
                turn_around_start_step = step
                turn_around_end_step = step + turn_num

                for _ in range(turn_num):
                    generate_trajectory_single.append(point_position.copy())
                    point_position[0] = first_turn_point[0]
                    point_position[1] = first_turn_point[1]

                direction = np.arctan2(end_point[1] - point_position[1], end_point[0] - point_position[0])
                direction_second = direction
                vector_direction = end_point - point_position[:2]
                vector_direction = vector_direction / np.linalg.norm(vector_direction)
            else:
                break

        point_position[0] += velocity * TRAJECTORY_DT_S * np.cos(direction)
        point_position[1] += velocity * TRAJECTORY_DT_S * np.sin(direction)

        direction = np.arctan2(end_point[1] - point_position[1], end_point[0] - point_position[0])
        velocity = perturb_motion_speed_mps(velocity)

        point_position[2] = direction
        point_position[3] = velocity
        generate_trajectory_single.append(point_position.copy())

    if turn_around_start_step <= 0 or turn_around_end_step >= len(generate_trajectory_single):
        raise RuntimeError('Polyline generation did not produce a valid turn point.')

    generate_trajectory_true = []
    num_steps_true = len(generate_trajectory_single)

    x_start, y_start = np.array(target_trajectory[0], dtype=float)
    x_turn, y_turn = np.array(target_trajectory[1], dtype=float)
    x_end, y_end = np.array(target_trajectory[2], dtype=float)

    x_step_1 = (x_turn - x_start) / turn_around_start_step
    y_step_1 = (y_turn - y_start) / turn_around_start_step
    x_step_2 = (x_end - x_turn) / (num_steps_true - turn_around_end_step)
    y_step_2 = (y_end - y_turn) / (num_steps_true - turn_around_end_step)

    for index in range(turn_around_start_step):
        generate_trajectory_true.append(np.array([
            x_start + x_step_1 * index,
            y_start + y_step_1 * index
        ]))

    for _ in range(turn_around_end_step - turn_around_start_step):
        generate_trajectory_true.append(np.array([x_turn, y_turn]))

    for index in range(num_steps_true - turn_around_end_step):
        generate_trajectory_true.append(np.array([
            x_turn + x_step_2 * index,
            y_turn + y_step_2 * index
        ]))

    return np.array(generate_trajectory_single), np.array(generate_trajectory_true)


def generate_circle_trajectory(target_trajectory, num_steps, center_point=None, radius=1):
    if center_point is None:
        center_point = [2, -2.5]

    generate_trajectory_single = []

    start_point = np.array(target_trajectory[0], dtype=float)
    end_point = np.array(target_trajectory[1], dtype=float)
    center_x, center_y = center_point

    start_vector = start_point - np.array([center_x, center_y])
    end_vector = end_point - np.array([center_x, center_y])

    theta_start = np.arctan2(start_point[1] - center_y, start_point[0] - center_x)

    dot_product = np.dot(start_vector, end_vector)
    norm_start = np.linalg.norm(start_vector)
    norm_end = np.linalg.norm(end_vector)
    cos_theta_start_end = dot_product / (norm_start * norm_end)
    angle_radians = np.arccos(np.clip(cos_theta_start_end, -1.0, 1.0))
    cross_product = np.cross(start_vector, end_vector)
    if cross_product < 0:
        angle_radians = 2 * np.pi - angle_radians

    v0 = sample_motion_speed_mps()
    omega = v0 / radius

    theta = theta_start
    point_position = np.array([start_point[0], start_point[1], 0, 0], dtype=float)
    generate_trajectory_single.append(point_position.copy())

    for _ in range(num_steps):
        position_vector = point_position[:2] - np.array([center_x, center_y])
        dot_product = np.dot(position_vector, end_vector)
        norm_position = np.linalg.norm(position_vector)
        cos_theta_position_end = dot_product / (norm_position * norm_end)
        theta_to_end = np.arccos(np.clip(cos_theta_position_end, -1.0, 1.0))

        cross_product = np.cross(position_vector, end_vector)
        if cross_product < 0:
            theta_to_end = 2 * np.pi - theta_to_end

        if theta_to_end < 0.05 or theta_to_end > 6:
            break

        omega += np.random.uniform(-DELTA_OMEGA_RADPS, DELTA_OMEGA_RADPS)
        omega = float(np.clip(omega, V_MIN_MPS / radius, V_MAX_MPS / radius))

        theta += omega * TRAJECTORY_DT_S
        point_position[0] = center_x + np.cos(theta) * radius
        point_position[1] = center_y + np.sin(theta) * radius
        generate_trajectory_single.append(point_position.copy())

    generate_trajectory_true = []
    num_steps_true = len(generate_trajectory_single)
    womiga_true = angle_radians / num_steps_true
    position_true = np.array(target_trajectory[0], dtype=float)

    for index in range(num_steps_true):
        theta_true = theta_start + womiga_true * index
        position_true[0] = center_x + np.cos(theta_true) * radius
        position_true[1] = center_y + np.sin(theta_true) * radius
        generate_trajectory_true.append(position_true.copy())

    return np.array(generate_trajectory_single), np.array(generate_trajectory_true)


def add_stationary_part(trajectory, trajectory_true, add_front, add_end):
    trajectory = np.concatenate((np.tile(trajectory[0], (add_front, 1)), trajectory))
    trajectory = np.concatenate((trajectory, np.tile(trajectory[-1], (add_end, 1))))

    trajectory_true = np.concatenate((np.tile(trajectory_true[0], (add_front, 1)), trajectory_true))
    trajectory_true = np.concatenate((trajectory_true, np.tile(trajectory_true[-1], (add_end, 1))))
    return trajectory, trajectory_true


def pad_trajectories(trajectories):
    max_length = max(len(t) for t in trajectories)
    result = []
    for t in trajectories:
        while len(t) < max_length:
            t = np.vstack([t, t[-1]])
        result.append(t)
    return result


RX1_COORD = np.array([0.0, 0.0])
RX2_COORD = np.array([4.0, 0.0])
TX_COORD = np.array([2.0, 0.0])


def is_valid_aoa(aoa):
    return np.isfinite(aoa) and aoa != INVALID_AOA


def is_same_receiver_ray(point1, point2, receiver, atol=1e-8):
    v1 = np.asarray(point1, dtype=float) - np.asarray(receiver, dtype=float)
    v2 = np.asarray(point2, dtype=float) - np.asarray(receiver, dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return False

    cross = np.abs(np.cross(v1, v2))
    collinear = cross / (norm1 * norm2) <= atol
    same_direction = np.dot(v1, v2) > 0
    return bool(collinear and same_direction)


def apply_receiver_occlusion(target_aoas, target_tofs, positions, receiver):
    aoas = list(target_aoas)
    tofs = list(target_tofs)
    n_target = len(aoas)

    for i in range(n_target):
        for j in range(i + 1, n_target):
            if not (is_valid_aoa(aoas[i]) and is_valid_aoa(aoas[j])):
                continue

            if is_same_receiver_ray(positions[i], positions[j], receiver):
                if tofs[i] <= tofs[j]:
                    aoas[j] = INVALID_AOA
                else:
                    aoas[i] = INVALID_AOA

    return aoas, tofs


def apply_angular_ambiguity(target_aoas, target_tofs):
    aoas = list(target_aoas)
    tofs = list(target_tofs)

    candidate_pairs = []
    for i in range(len(aoas)):
        for j in range(i + 1, len(aoas)):
            if is_valid_aoa(aoas[i]) and is_valid_aoa(aoas[j]):
                diff = abs(aoas[i] - aoas[j])
                if diff < THETA_RES_DEG:
                    candidate_pairs.append((diff, i, j))

    candidate_pairs.sort(key=lambda item: item[0])
    used_targets = set()

    for _, i, j in candidate_pairs:
        if i in used_targets or j in used_targets:
            continue
        if not (is_valid_aoa(aoas[i]) and is_valid_aoa(aoas[j])):
            continue

        # 1-P_C: both AoAs remain unchanged.
        if np.random.rand() >= P_C:
            continue

        original_i = aoas[i]
        original_j = aoas[j]
        aoas[i] = (original_i + original_j) / 2.0

        if np.random.rand() < P_A:
            aoas[j] = np.random.uniform(*ABNORMAL_AOA_RANGE_DEG)
        else:
            aoas[j] = INVALID_AOA

        used_targets.add(i)
        used_targets.add(j)

    return aoas, tofs


def apply_failure_event(target_aoas_rx1, target_tofs_rx1,
                        target_aoas_rx2, target_tofs_rx2):

    if np.random.rand() >= P_F:
        return (list(target_aoas_rx1), list(target_tofs_rx1),
                list(target_aoas_rx2), list(target_tofs_rx2))

    def replace_detectable(aoas, tofs):
        aoas_out = list(aoas)
        tofs_out = list(tofs)
        for k in range(len(aoas_out)):
            if is_valid_aoa(aoas_out[k]):
                aoas_out[k] = np.random.uniform(*ABNORMAL_AOA_RANGE_DEG)
                tofs_out[k] = np.random.uniform(*ABNORMAL_TOF_RANGE_NS)
        return aoas_out, tofs_out

    aoas1, tofs1 = replace_detectable(target_aoas_rx1, target_tofs_rx1)
    aoas2, tofs2 = replace_detectable(target_aoas_rx2, target_tofs_rx2)
    return aoas1, tofs1, aoas2, tofs2


def add_measurement_noise(direct_aoa, direct_tof, target_aoas, target_tofs):
    """Add zero-mean Gaussian noise to direct and detectable target measurements."""
    direct_aoa_noisy = direct_aoa + np.random.randn() * SIGMA_THETA_DEG
    direct_tof_noisy = direct_tof + np.random.randn() * SIGMA_T_NS

    aoas_noisy = list(target_aoas)
    tofs_noisy = list(target_tofs)
    for i in range(len(aoas_noisy)):
        if is_valid_aoa(aoas_noisy[i]):
            aoas_noisy[i] += np.random.randn() * SIGMA_THETA_DEG
            tofs_noisy[i] += np.random.randn() * SIGMA_T_NS

    return direct_aoa_noisy, direct_tof_noisy, aoas_noisy, tofs_noisy


def pack_receiver_measurements(direct_aoa, direct_tof, target_aoas, target_tofs):
    """
    Pack one receiver frame. INVALID_AOA targets are omitted.

    Valid positive AoAs are retained; this is required because abnormal AoAs
    are now sampled from [-90, 90] rather than only negative angles.
    """
    aoa_frame = [direct_aoa]
    tof_frame = [direct_tof]

    for aoa, tof in zip(target_aoas, target_tofs):
        if is_valid_aoa(aoa):
            aoa_frame.append(aoa)
            tof_frame.append(tof)

    return aoa_frame, tof_frame


def simulate_multi_person_measurements(trajectories, true_trajectories):

    trajectories = [np.asarray(t) for t in trajectories]
    true_trajectories = [np.asarray(t) for t in true_trajectories]

    if len(trajectories) not in (2, 3):
        raise ValueError('Only 2-person and 3-person measurement simulation is supported.')

    frame_lengths = [len(t) for t in trajectories]
    if len(set(frame_lengths)) != 1:
        raise ValueError('All generated trajectories must have the same frame length.')

    position_seq_array = np.concatenate(
        [t[np.newaxis, :, :2] for t in true_trajectories], axis=0
    )
    position_seq_list = position_seq_array.tolist()

    AoA_list_1, AoA_list_2 = [], []
    ToF_list_1, ToF_list_2 = [], []

    for point_index in range(frame_lengths[0]):
        positions = [t[point_index, :2] for t in trajectories]

        # Direct path is the reference measurement.
        AoA_direct_rx1 = 0.0
        AoA_direct_rx2 = 0.0
        ToF_direct_rx1 = calculate_rtof_of_polyline([RX1_COORD, TX_COORD])
        ToF_direct_rx2 = calculate_rtof_of_polyline([TX_COORD, RX2_COORD])

        # Ideal target AoAs/rToFs generated from the same virtual positions.
        target_aoas_rx1 = [
            calculate_angle_diff(pos, RX1_COORD, RX2_COORD, RX1_COORD) * -1
            for pos in positions
        ]
        target_aoas_rx2 = [
            calculate_angle_diff(pos, RX2_COORD, RX1_COORD, RX2_COORD) * -1
            for pos in positions
        ]

        target_tofs_rx1 = [
            calculate_rtof_of_polyline([RX1_COORD, pos, TX_COORD])
            for pos in positions
        ]
        target_tofs_rx2 = [
            calculate_rtof_of_polyline([TX_COORD, pos, RX2_COORD])
            for pos in positions
        ]


        target_aoas_rx1, target_tofs_rx1 = apply_receiver_occlusion(
            target_aoas_rx1, target_tofs_rx1, positions, RX1_COORD
        )
        target_aoas_rx2, target_tofs_rx2 = apply_receiver_occlusion(
            target_aoas_rx2, target_tofs_rx2, positions, RX2_COORD
        )


        target_aoas_rx1, target_tofs_rx1 = apply_angular_ambiguity(
            target_aoas_rx1, target_tofs_rx1
        )
        target_aoas_rx2, target_tofs_rx2 = apply_angular_ambiguity(
            target_aoas_rx2, target_tofs_rx2
        )


        target_aoas_rx1, target_tofs_rx1, target_aoas_rx2, target_tofs_rx2 = \
            apply_failure_event(
                target_aoas_rx1, target_tofs_rx1,
                target_aoas_rx2, target_tofs_rx2
            )

        AoA_direct_rx1, ToF_direct_rx1, target_aoas_rx1, target_tofs_rx1 = \
            add_measurement_noise(
                AoA_direct_rx1, ToF_direct_rx1,
                target_aoas_rx1, target_tofs_rx1
            )
        AoA_direct_rx2, ToF_direct_rx2, target_aoas_rx2, target_tofs_rx2 = \
            add_measurement_noise(
                AoA_direct_rx2, ToF_direct_rx2,
                target_aoas_rx2, target_tofs_rx2
            )

        aoa_frame_1, tof_frame_1 = pack_receiver_measurements(
            AoA_direct_rx1, ToF_direct_rx1, target_aoas_rx1, target_tofs_rx1
        )
        aoa_frame_2, tof_frame_2 = pack_receiver_measurements(
            AoA_direct_rx2, ToF_direct_rx2, target_aoas_rx2, target_tofs_rx2
        )

        AoA_list_1.append(aoa_frame_1)
        ToF_list_1.append(tof_frame_1)
        AoA_list_2.append(aoa_frame_2)
        ToF_list_2.append(tof_frame_2)

    return AoA_list_1, AoA_list_2, ToF_list_1, ToF_list_2, position_seq_list


def simulate_two_person_measurements(
        trajectory1, trajectory2,
        trajectory1_true, trajectory2_true):
    return simulate_multi_person_measurements(
        [trajectory1, trajectory2],
        [trajectory1_true, trajectory2_true]
    )


def simulate_three_person_measurements(
        trajectory1, trajectory2, trajectory3,
        trajectory1_true, trajectory2_true, trajectory3_true):
    return simulate_multi_person_measurements(
        [trajectory1, trajectory2, trajectory3],
        [trajectory1_true, trajectory2_true, trajectory3_true]
    )


def main():
    with open(POINT_PATH, 'rb') as file:
        points = pickle.load(file)

    trajectories_information = points[-1]

    n_person = len(trajectories_information)
    if n_person not in (2, 3):
        raise ValueError(
            f'Unsupported person count: {n_person}. ' 
            'The current code supports only 2-person or 3-person scenes.'
        )

    scene_type = infer_scene_type(trajectories_information)
    print('Detected person count:', n_person)
    print('Detected scene type:', scene_type)

    x_range = np.arange(0, 4.5, GRID_INTERVAL_M)
    y_range = np.arange(-4, 0.5, GRID_INTERVAL_M)
    points_all = [(x, y) for x in x_range for y in y_range]

    if scene_type == 'circle':
        circle_1_points, circle_2_points = trajectories_information[0], trajectories_information[1]

        center_point_read = [
            (circle_1_points[4][0] + circle_2_points[4][0]) / 2,
            (circle_1_points[4][1] + circle_2_points[4][1]) / 2
        ]
        radius_read = round_to_nearest_half_integer(
            (circle_1_points[5] + circle_2_points[5]) / 2
        )
        circle_1_start_point_read = [
            (circle_1_points[2][0] + circle_2_points[3][0]) / 2,
            (circle_1_points[2][1] + circle_2_points[3][1]) / 2
        ]
        circle_1_end_point_read = [
            (circle_1_points[3][0] + circle_2_points[2][0]) / 2,
            (circle_1_points[3][1] + circle_2_points[2][1]) / 2
        ]

        center_points = find_nearby_points(points_all, center_point_read)
        radius_sets = [radius_read - 0.2, radius_read, radius_read + 0.2]

    elif scene_type == 'line':
        polyline_1_points, polyline_2_points = trajectories_information[0], trajectories_information[1]

        polyline_1_start_points = find_nearby_points(points_all, polyline_1_points[2])
        polyline_1_turn_points = find_nearby_points(points_all, polyline_1_points[4])

        polyline_2_start_points = find_nearby_points(points_all, polyline_2_points[2])
        polyline_2_turn_points = find_nearby_points(points_all, polyline_2_points[4])

    elif scene_type == 'N':
        polyline_points, line_points = reorder_two_person_information(
            trajectories_information, scene_type
        )

        polyline_end_point_read = [
            (polyline_points[3][0] + line_points[2][0]) / 2,
            (polyline_points[3][1] + line_points[2][1]) / 2
        ]

        polyline_start_points = find_nearby_points(points_all, polyline_points[2])
        polyline_turn_points = find_nearby_points(points_all, polyline_points[4])
        polyline_end_points = find_nearby_points(points_all, polyline_end_point_read)
        line_end_points = find_nearby_points(points_all, line_points[3])

    elif scene_type == 'triangle':
        polyline_points, line_points = reorder_two_person_information(
            trajectories_information, scene_type
        )

        polyline_start_point_read = [
            (polyline_points[2][0] + line_points[3][0]) / 2,
            (polyline_points[2][1] + line_points[3][1]) / 2
        ]
        polyline_end_point_read = [
            (polyline_points[3][0] + line_points[2][0]) / 2,
            (polyline_points[3][1] + line_points[2][1]) / 2
        ]

        polyline_start_points = find_nearby_points(points_all, polyline_start_point_read)
        polyline_turn_points = find_nearby_points(points_all, polyline_points[4])
        polyline_end_points = find_nearby_points(points_all, polyline_end_point_read)

    elif scene_type == 'W':
        line_1_points, polyline_points, line_2_points = reorder_w_information(
            trajectories_information
        )

        polyline_start_point_read = [
            (polyline_points[2][0] + line_1_points[3][0]) / 2,
            (polyline_points[2][1] + line_1_points[3][1]) / 2
        ]
        polyline_end_point_read = [
            (polyline_points[3][0] + line_2_points[2][0]) / 2,
            (polyline_points[3][1] + line_2_points[2][1]) / 2
        ]

        line_1_start_points = find_nearby_points(points_all, line_1_points[2])
        polyline_start_points = find_nearby_points(points_all, polyline_start_point_read)
        polyline_turn_points = find_nearby_points(points_all, polyline_points[4])
        polyline_end_points = find_nearby_points(points_all, polyline_end_point_read)
        line_2_end_points = find_nearby_points(points_all, line_2_points[3])

    else:
        raise ValueError(f'Unsupported scene type: {scene_type}')

    # --------------------------------------------------------
    # Generate virtual samples
    # --------------------------------------------------------
    for N_index in range(N_GENERATE):
        num_steps = np.random.randint(200, 300)

        if scene_type == 'circle':
            circle_center = random.choice(center_points)
            circle_radius = random.choice(radius_sets)

            target1_start_angle_read = calculate_angle_diff(
                circle_1_start_point_read, circle_center, (0, 1), (0, 0)
            )
            # Use the detected start angle directly; no manual +/-5 deg perturbation.
            target1_start_angle = target1_start_angle_read
            target1_start_x = circle_center[0] - np.sin(np.deg2rad(target1_start_angle)) * circle_radius
            target1_start_y = circle_center[1] + np.cos(np.deg2rad(target1_start_angle)) * circle_radius

            target1_end_angle_read = calculate_angle_diff(
                circle_1_end_point_read, circle_center, (0, -1), (0, 0)
            )
            # Use the detected end angle directly; no manual +/-5 deg perturbation.
            target1_end_angle = target1_end_angle_read
            target1_end_x = circle_center[0] + np.sin(np.deg2rad(target1_end_angle)) * circle_radius
            target1_end_y = circle_center[1] - np.cos(np.deg2rad(target1_end_angle)) * circle_radius

            target1_trajectory = [
                [target1_start_x, target1_start_y],
                [target1_end_x, target1_end_y]
            ]
            trajectory1, trajectory1_true = generate_circle_trajectory(
                target1_trajectory, num_steps, circle_center, circle_radius
            )

            target2_start_angle = target1_end_angle
            target2_start_x = circle_center[0] + np.sin(np.deg2rad(target2_start_angle)) * circle_radius
            target2_start_y = circle_center[1] - np.cos(np.deg2rad(target2_start_angle)) * circle_radius

            target2_end_angle = target1_start_angle
            target2_end_x = circle_center[0] - np.sin(np.deg2rad(target2_end_angle)) * circle_radius
            target2_end_y = circle_center[1] + np.cos(np.deg2rad(target2_end_angle)) * circle_radius

            target2_trajectory = [
                [target2_start_x, target2_start_y],
                [target2_end_x, target2_end_y]
            ]
            trajectory2, trajectory2_true = generate_circle_trajectory(
                target2_trajectory, num_steps, circle_center, circle_radius
            )

            trajectory1, trajectory1_true = add_stationary_part(
                trajectory1, trajectory1_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )
            trajectory2, trajectory2_true = add_stationary_part(
                trajectory2, trajectory2_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )

            trajectory1, trajectory2 = pad_trajectories([trajectory1, trajectory2])
            trajectory1_true, trajectory2_true = pad_trajectories([trajectory1_true, trajectory2_true])

            AoA_list_1, AoA_list_2, ToF_list_1, ToF_list_2, position_seq_list = \
                simulate_two_person_measurements(
                    trajectory1, trajectory2,
                    trajectory1_true, trajectory2_true
                )

        elif scene_type == 'line':

            p1_start = random.choice(polyline_1_start_points)
            p1_turn = random.choice(polyline_1_turn_points)
            p1_end = p1_start

            p2_start = random.choice(polyline_2_start_points)
            p2_turn = random.choice(polyline_2_turn_points)
            p2_end = p2_start

            trajectory1, trajectory1_true = generate_polyline_trajectory(
                [p1_start, p1_turn, p1_end],
                num_steps,
                turn_num_range=(9, 14)
            )
            trajectory2, trajectory2_true = generate_polyline_trajectory(
                [p2_start, p2_turn, p2_end],
                num_steps,
                turn_num_range=(9, 14)
            )

            trajectory1, trajectory1_true = add_stationary_part(
                trajectory1, trajectory1_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )
            trajectory2, trajectory2_true = add_stationary_part(
                trajectory2, trajectory2_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )

            trajectory1, trajectory2 = pad_trajectories([trajectory1, trajectory2])
            trajectory1_true, trajectory2_true = pad_trajectories([trajectory1_true, trajectory2_true])

            AoA_list_1, AoA_list_2, ToF_list_1, ToF_list_2, position_seq_list = \
                simulate_two_person_measurements(
                    trajectory1, trajectory2,
                    trajectory1_true, trajectory2_true
                )

        elif scene_type == 'N':
            polyline_start_point = random.choice(polyline_start_points)
            polyline_turn_point = random.choice(polyline_turn_points)
            polyline_end_point = random.choice(polyline_end_points)
            line_start_point = polyline_end_point
            line_end_point = random.choice(line_end_points)

            trajectory1, trajectory1_true = generate_polyline_trajectory(
                [polyline_start_point, polyline_turn_point, polyline_end_point],
                num_steps,
                turn_num_range=(2, 5)
            )
            trajectory2, trajectory2_true = generate_straight_trajectory(
                [line_start_point, line_end_point],
                num_steps
            )

            trajectory1, trajectory1_true = add_stationary_part(
                trajectory1, trajectory1_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )
            trajectory2, trajectory2_true = add_stationary_part(
                trajectory2, trajectory2_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )

            trajectory1, trajectory2 = pad_trajectories([trajectory1, trajectory2])
            trajectory1_true, trajectory2_true = pad_trajectories([trajectory1_true, trajectory2_true])

            AoA_list_1, AoA_list_2, ToF_list_1, ToF_list_2, position_seq_list = \
                simulate_two_person_measurements(
                    trajectory1, trajectory2,
                    trajectory1_true, trajectory2_true
                )

        elif scene_type == 'triangle':
            polyline_start_point = random.choice(polyline_start_points)
            polyline_turn_point = random.choice(polyline_turn_points)
            polyline_end_point = random.choice(polyline_end_points)

            trajectory1, trajectory1_true = generate_polyline_trajectory(
                [polyline_start_point, polyline_turn_point, polyline_end_point],
                num_steps,
                turn_num_range=(3, 10)
            )
            trajectory2, trajectory2_true = generate_straight_trajectory(
                [polyline_end_point, polyline_start_point],
                num_steps
            )

            trajectory1, trajectory1_true = add_stationary_part(
                trajectory1, trajectory1_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )
            trajectory2, trajectory2_true = add_stationary_part(
                trajectory2, trajectory2_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )

            trajectory1, trajectory2 = pad_trajectories([trajectory1, trajectory2])
            trajectory1_true, trajectory2_true = pad_trajectories([trajectory1_true, trajectory2_true])

            AoA_list_1, AoA_list_2, ToF_list_1, ToF_list_2, position_seq_list = \
                simulate_two_person_measurements(
                    trajectory1, trajectory2,
                    trajectory1_true, trajectory2_true
                )

        elif scene_type == 'W':
            line_1_start_point = random.choice(line_1_start_points)
            polyline_start_point = random.choice(polyline_start_points)
            polyline_turn_point = random.choice(polyline_turn_points)
            polyline_end_point = random.choice(polyline_end_points)
            line_2_end_point = random.choice(line_2_end_points)

            trajectory1, trajectory1_true = generate_straight_trajectory(
                [line_1_start_point, polyline_start_point],
                num_steps
            )
            trajectory2, trajectory2_true = generate_polyline_trajectory(
                [polyline_start_point, polyline_turn_point, polyline_end_point],
                num_steps,
                turn_num_range=(2, 5)
            )
            trajectory3, trajectory3_true = generate_straight_trajectory(
                [polyline_end_point, line_2_end_point],
                num_steps
            )

            trajectory1, trajectory1_true = add_stationary_part(
                trajectory1, trajectory1_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )
            trajectory2, trajectory2_true = add_stationary_part(
                trajectory2, trajectory2_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )
            trajectory3, trajectory3_true = add_stationary_part(
                trajectory3, trajectory3_true,
                STATIONARY_FRAMES,STATIONARY_FRAMES
            )

            trajectory1, trajectory2, trajectory3 = pad_trajectories(
                [trajectory1, trajectory2, trajectory3]
            )
            trajectory1_true, trajectory2_true, trajectory3_true = pad_trajectories(
                [trajectory1_true, trajectory2_true, trajectory3_true]
            )

            AoA_list_1, AoA_list_2, ToF_list_1, ToF_list_2, position_seq_list = \
                simulate_three_person_measurements(
                    trajectory1, trajectory2, trajectory3,
                    trajectory1_true, trajectory2_true, trajectory3_true
                )


        if n_person == 2:
            generated_trajectories = [trajectory1, trajectory2]
        elif n_person == 3:
            generated_trajectories = [trajectory1, trajectory2, trajectory3]

        coordinate_seq = calculate_coordinate_seq(
            AoA_list_1, AoA_list_2, ToF_list_1, ToF_list_2
        )
        Position_Seq_all=[coordinate_seq,generated_trajectories]

        with open(SAVE_ROOT+'virtual_data_'+str(N_index).pkl, 'wb') as handle:
            pickle.dump(Position_Seq_all, handle)



if __name__ == '__main__':
    main()

