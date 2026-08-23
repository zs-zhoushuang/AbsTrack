import numpy as np
import pickle
import pylab
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
import os
from scipy.optimize import least_squares

def residual(params, x, y):
    """
    计算残差函数，用于最小二乘法优化
    params: 包含圆心坐标和半径的参数数组 [a, b, r]
    x, y: 数据点的x坐标和y坐标
    """
    a, b, r = params
    return (x - a)**2 + (y - b)**2 - r**2

def fit_circle(x, y):
    """
    使用最小二乘法拟合圆
    x, y: 数据点的x坐标和y坐标
    返回拟合结果：圆心坐标 (a, b) 和半径 r
    """
    # 初始猜测值，可以根据实际情况设置
    initial_guess = [np.mean(x), np.mean(y), np.std(x)]

    # 使用最小二乘法拟合圆
    result = least_squares(residual, initial_guess, args=(x, y))

    # 拟合结果为圆心坐标和半径
    a, b, r = result.x
    return (a, b), r


def get_center_radius(x,y):
    # 示例数据
    # x = np.array([1, 2, 3, 4, 5])
    # y = np.array([2, 3, 4, 5, 6])

    # 拟合圆
    center, radius = fit_circle(x, y)
    print("拟合结果：圆心坐标 =", center, "半径 =", radius)
    return center, radius



# 核心代码，求斜率w,截距b
def line_fit(data_x, data_y):
    m = len(data_y)
    x_bar = np.mean(data_x)
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0
    for i in range(m):
        x = data_x[i]
        y = data_y[i]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    # 根据公式计算w
    w = sum_yx / (sum_x2 - m * (x_bar ** 2))

    for i in range(m):
        x = data_x[i]
        y = data_y[i]
        sum_delta += (y - w * x)
    b = sum_delta / m
    return w, b


def get_distance(x,y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

threashold=0.2

path='one_person_walking_trajectory/'
temp_paths = os.listdir(path, )
temp_paths.sort(key=lambda x: int(x[17:-4]))

# temp_paths=temp_paths[3:]
for tra in range(len(temp_paths)):
    with open(path+temp_paths[tra], 'rb') as f:
        trajectories = pickle.load(f)
    Trajectory_Matrix = trajectories[1][:, :, 0:2]


    init_position = []
    end_position = []

    init_error_distance = []
    end_error_distance = []

    trajectories_information = []

    print(Trajectory_Matrix.shape)
    for i in range(len(Trajectory_Matrix)):
        init_position.append((Trajectory_Matrix[i, 0, 0], Trajectory_Matrix[i, 0, 1]))
        end_position.append((Trajectory_Matrix[i, -1, 0], Trajectory_Matrix[i, -1, 1]))
    print('init_position ', init_position)
    print('end_position ', end_position)


    '''judge is circle or line'''
    dis_var, traje_Lable = [], []
    for i in range(len(Trajectory_Matrix)):
        center, radius = get_center_radius(Trajectory_Matrix[i, :, 0], Trajectory_Matrix[i, :, 1])
        dis_var_temp = 0
        for j in range(len(Trajectory_Matrix[i])):
            dis = np.sqrt((Trajectory_Matrix[i, j, 0] - center[0]) ** 2 + (Trajectory_Matrix[i, j, 1] - center[1]) ** 2)
            # print(dis)
            dis_var_temp += abs(dis - radius)
        dis_var.append((center, radius, dis_var_temp / len(Trajectory_Matrix[i])))
        if center[0] < 1 or center[0] > 3 or center[1] > -1 or center[1] < -3:
            print('trajec ', i, ' is line')
            traje_Lable.append(1)
            trajectories_information.append([i, 'line', init_position[i], end_position[i]])
        elif dis_var_temp / len(Trajectory_Matrix[i]) < 0.05:
            print('trajec ', i, ' is circle')
            print(dis_var_temp / len(Trajectory_Matrix[i]))
            traje_Lable.append(0)
            trajectories_information.append([i, 'circle', init_position[i], end_position[i], center, radius])
            pylab.scatter(init_position[i][0], init_position[i][1])
            pylab.scatter(end_position[i][0], end_position[i][1])
            pylab.scatter(center[0], center[1])
            pylab.plot(Trajectory_Matrix[i, :, 0], Trajectory_Matrix[i, :, 1], '-.')
        else:
            print('trajec ', i, ' is line')
            print(dis_var_temp / len(Trajectory_Matrix[i]))
            traje_Lable.append(1)
            trajectories_information.append([i, 'line', init_position[i], end_position[i]])

    print('radius dis var is ', dis_var)
    print('traje_Lable is ', traje_Lable)

    '''judge the straight line or broken line, find the broken points'''
    start_end_flag = []
    for i in range(len(traje_Lable)):
        if traje_Lable[i] != 0:  # is not circle
            start_flag = 0
            end_flag = len(Trajectory_Matrix[i])
            for j in range(len(Trajectory_Matrix[i])):
                dis = get_distance(Trajectory_Matrix[i, j], init_position[i])
                if dis > threashold:
                    start_flag = j
                    break
            for j in range(len(Trajectory_Matrix[i]) - 1, -1, -1):
                dis = get_distance(Trajectory_Matrix[i, j], end_position[i])
                if dis > threashold:
                    end_flag = j
                    break
            print([i, start_flag, end_flag])
            start_end_flag.append([i, start_flag, end_flag])

    scope = []
    for i in range(len(start_end_flag)):
        Trajectory_Matrix_new = Trajectory_Matrix[start_end_flag[i][0]][start_end_flag[i][1]:start_end_flag[i][2]]
        win_len = 10
        step = int(win_len * 0.5)
        scope_temp = []
        for j in range(0, len(Trajectory_Matrix_new) - win_len, step):
            w, b = line_fit(Trajectory_Matrix_new[j:j + win_len, 0], Trajectory_Matrix_new[j:j + win_len, 1])
            scope_temp.append([round(w, 2), int(j + 0.5 * win_len)])
        print(scope_temp)
        scope.append(scope_temp)
        # pylab.scatter(Trajectory_Matrix_new[0, 0], Trajectory_Matrix_new[0, 1])
        # pylab.plot(Trajectory_Matrix_new[:, 0], Trajectory_Matrix_new[:, 1], '-.')

    for i in range(len(scope)):
        count_1, count_2 = 0, 0
        for j in range(len(scope[i])):
            if abs(scope[i][j][0]) <= 0.2:#0.1
                count_1 += 1
            if abs(scope[i][j][0]) > 10:
                count_2 += 1
        if count_1 / len(scope[i]) >= 0.8:
            print('trajectory', start_end_flag[i][0], ' is horizontal line')
            trajectories_information[start_end_flag[i][0]] = [start_end_flag[i][0], 'horizontal_line',
                                                              init_position[start_end_flag[i][0]],
                                                              end_position[start_end_flag[i][0]]]

            pylab.scatter(init_position[start_end_flag[i][0]][0], init_position[start_end_flag[i][0]][1])
            pylab.scatter(end_position[start_end_flag[i][0]][0], end_position[start_end_flag[i][0]][1])
            pylab.plot(Trajectory_Matrix[start_end_flag[i][0], :, 0], Trajectory_Matrix[start_end_flag[i][0], :, 1],
                       '-.')
        elif count_2 / len(scope[i]) > 0.8:
            print('trajectory', start_end_flag[i][0], ' is vertical line')
            trajectories_information[start_end_flag[i][0]] = [start_end_flag[i][0], 'vertical_line',
                                                              init_position[start_end_flag[i][0]],
                                                              end_position[start_end_flag[i][0]]]
            pylab.scatter(init_position[start_end_flag[i][0]][0], init_position[start_end_flag[i][0]][1])
            pylab.scatter(end_position[start_end_flag[i][0]][0], end_position[start_end_flag[i][0]][1])
            pylab.plot(Trajectory_Matrix[start_end_flag[i][0], :, 0], Trajectory_Matrix[start_end_flag[i][0], :, 1],
                       '-.')
        else:
            broken_flag = False
            sign_opp = []
            for j in range(len(scope[i]) - 1):
                if scope[i][j][0] * scope[i][j + 1][0] <= 0:
                    if abs(scope[i][j][0]) < abs(scope[i][j + 1][0]):
                        sign_opp.append(j)
                    else:
                        sign_opp.append(j + 1)
                    broken_flag = True
            if broken_flag == False:
                print('trajectory', start_end_flag[i][0], ' is bias line')
                trajectories_information[start_end_flag[i][0]] = [start_end_flag[i][0], 'bias_line',
                                                                  init_position[start_end_flag[i][0]],
                                                                  end_position[start_end_flag[i][0]]]
                pylab.scatter(init_position[start_end_flag[i][0]][0], init_position[start_end_flag[i][0]][1])
                pylab.scatter(end_position[start_end_flag[i][0]][0], end_position[start_end_flag[i][0]][1])
                pylab.plot(Trajectory_Matrix[start_end_flag[i][0], :, 0], Trajectory_Matrix[start_end_flag[i][0], :, 1],
                           '-.')
            else:
                print('trajectory', start_end_flag[i][0], ' is broken line')
                broken_cor = []
                for k in range(len(sign_opp)):
                    broken_point = scope[i][sign_opp[k]][1] + start_end_flag[i][1]
                    cor_temp = Trajectory_Matrix[start_end_flag[i][0]][broken_point]
                    print('broken_cor', cor_temp)
                    pylab.scatter(cor_temp[0], cor_temp[1])
                    broken_cor.append(cor_temp)
                trajectories_information[start_end_flag[i][0]] = [start_end_flag[i][0], 'broken_line',
                                                                  init_position[start_end_flag[i][0]],
                                                                  end_position[start_end_flag[i][0]]] + broken_cor
                pylab.scatter(init_position[start_end_flag[i][0]][0], init_position[start_end_flag[i][0]][1])
                pylab.scatter(end_position[start_end_flag[i][0]][0], end_position[start_end_flag[i][0]][1])
                for n in range(len(broken_cor)):
                    pylab.scatter(broken_cor[n][0], broken_cor[n][1])
                pylab.plot(Trajectory_Matrix[start_end_flag[i][0], :, 0], Trajectory_Matrix[start_end_flag[i][0], :, 1],
                           '-.')

    print(temp_paths[tra])
    for i in range(len(trajectories_information)):
        print(trajectories_information[i])

    trajectories_infor_save=[Trajectory_Matrix,trajectories_information]

    with open(path+'trajectories_infor_save'+temp_paths[tra][17:-4]+'.pkl', 'wb') as handle:
        pickle.dump(trajectories_infor_save, handle, -1)





















