import numpy as np
import pickle
import pylab
from scipy import signal
import os

coordinate_rx_1 = [0, 0]
coordinate_rx_2 = [4, 0]
coordinate_tx = [2, 0]

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def get_coordinate(angle):
    if angle[0] == -90:
        x = coordinate_rx_1[0]
        y = coordinate_rx_2[0] * np.tan(angle[1] / 180 * np.pi)
    elif angle[1] == -90:
        x = coordinate_rx_2[0]
        y = coordinate_rx_2[0] * np.tan(angle[0] / 180 * np.pi)
    else:
        x = coordinate_rx_2[0] * np.tan(angle[1] / 180 * np.pi) / \
            (np.tan(angle[0] / 180 * np.pi) + np.tan(angle[1] / 180 * np.pi))
        y = coordinate_rx_2[0] * np.tan(angle[0] / 180 * np.pi) * np.tan(angle[1] / 180 * np.pi) / \
            (np.tan(angle[0] / 180 * np.pi) + np.tan(angle[1] / 180 * np.pi))
    return np.array([round(x, 2), round(y, 2)])

def get_rToF(coordinate_target):
    length_1 = np.sqrt(
        (coordinate_target[0] - coordinate_tx[0]) ** 2 + (coordinate_target[1] - coordinate_tx[1]) ** 2) + \
               np.sqrt((coordinate_target[0] - coordinate_rx_1[0]) ** 2 + (
                       coordinate_target[1] - coordinate_rx_1[1]) ** 2)
    rToF_1 = round(length_1 / 0.3, 2)
    length_2 = np.sqrt(
        (coordinate_target[0] - coordinate_tx[0]) ** 2 + (coordinate_target[1] - coordinate_tx[1]) ** 2) + \
               np.sqrt((coordinate_target[0] - coordinate_rx_2[0]) ** 2 + (
                       coordinate_target[1] - coordinate_rx_2[1]) ** 2)
    rToF_2 = round(length_2 / 0.3, 2)
    return np.array([rToF_1, rToF_2])


def get_position_combine(AoA_1, ToF_1, AoA_2, ToF_2):
    AoA_ToF_zip_1 = zip(AoA_1, ToF_1)
    AoA_ToF_zip_2 = zip(AoA_2, ToF_2)
    # 从大到小排列
    Zipped_1 = sorted(AoA_ToF_zip_1, key=lambda x: x[0], reverse=True)
    Zipped_2 = sorted(AoA_ToF_zip_2, key=lambda x: x[0], reverse=True)
    # print(Zipped_1,Zipped_2)

    AoA_1_Descend, ToF_1_Descend = zip(*Zipped_1)
    AoA_1_Descend = list(AoA_1_Descend)
    ToF_1_Descend = list(ToF_1_Descend)
    '''calculate estimated rToF'''
    for i in range(1, len(ToF_1_Descend)):
        ToF_1_Descend[i] = ToF_1_Descend[i] - ToF_1_Descend[0]

    AoA_2_Descend, ToF_2_Descend = zip(*Zipped_2)
    AoA_2_Descend = list(AoA_2_Descend)
    ToF_2_Descend = list(ToF_2_Descend)
    '''calculate estimated rToF'''
    for i in range(1, len(ToF_2_Descend)):
        ToF_2_Descend[i] = ToF_2_Descend[i] - ToF_2_Descend[0]
    print('rx1 AoA ToF Descend', AoA_1_Descend, ToF_1_Descend)
    print('rx2 AoA ToF Descend', AoA_2_Descend, ToF_2_Descend)
    if len(AoA_1_Descend) == 4 or len(AoA_2_Descend) == 4:
        print('3 targets estimated')
        if len(AoA_1_Descend) == 4 and len(AoA_2_Descend) == 4:
            combine_1 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[2], ToF_1_Descend[2], ToF_2_Descend[2]],
                         [AoA_1_Descend[3], AoA_2_Descend[3], ToF_1_Descend[3], ToF_2_Descend[3]]]

            combine_2 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[3], ToF_1_Descend[2], ToF_2_Descend[3]],
                         [AoA_1_Descend[3], AoA_2_Descend[2], ToF_1_Descend[3], ToF_2_Descend[2]]]

            combine_3 = [[AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]],
                         [AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]],
                         [AoA_1_Descend[3], AoA_2_Descend[3], ToF_1_Descend[3], ToF_2_Descend[3]]]

            combine_4 = [[AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]],
                         [AoA_1_Descend[2], AoA_2_Descend[3], ToF_1_Descend[2], ToF_2_Descend[3]],
                         [AoA_1_Descend[3], AoA_2_Descend[1], ToF_1_Descend[3], ToF_2_Descend[1]]]

            combine_5 = [[AoA_1_Descend[1], AoA_2_Descend[3], ToF_1_Descend[1], ToF_2_Descend[3]],
                         [AoA_1_Descend[2], AoA_2_Descend[2], ToF_1_Descend[2], ToF_2_Descend[2]],
                         [AoA_1_Descend[3], AoA_2_Descend[1], ToF_1_Descend[3], ToF_2_Descend[1]]]

            combine_6 = [[AoA_1_Descend[1], AoA_2_Descend[3], ToF_1_Descend[1], ToF_2_Descend[3]],
                         [AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]],
                         [AoA_1_Descend[3], AoA_2_Descend[2], ToF_1_Descend[3], ToF_2_Descend[2]]]
            combines = [combine_1, combine_2, combine_3, combine_4, combine_5, combine_6]
            coordinates = np.zeros((len(combines), len(combines[0]), 2))  # (x,y)
            '''calculate candidate rToF'''
            rToFs = np.zeros((len(combines), len(combines[0]), 2))  # (rToF_1,rToF_2)
            for i in range(len(combines)):
                for j in range(len(combines[i])):
                    combine_cor = get_coordinate([combines[i][j][0], combines[i][j][1]])
                    coordinates[i, j] = combine_cor
                    rToFs[i, j] = get_rToF(combine_cor)

            deta_ToFs = []
            for i in range(len(rToFs)):
                sum = 0
                for j in range(len(rToFs[0])):
                    for num in range(2):
                        deta = (rToFs[i][j][num] - combines[i][j][num + 2]) ** 2
                        sum += deta
                deta_ToFs.append(sum)
            min_value = min(deta_ToFs)
            index = deta_ToFs.index(min_value)
            return [x for x in coordinates[index]]

        elif len(AoA_1_Descend) == 4 and len(AoA_2_Descend) == 3:
            combine_1 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[2], ToF_1_Descend[2], ToF_2_Descend[2]],
                         [AoA_1_Descend[3], AoA_2_Descend[2], ToF_1_Descend[3], ToF_2_Descend[2]]]

            combine_2 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[2], ToF_1_Descend[2], ToF_2_Descend[2]],
                         [AoA_1_Descend[3], AoA_2_Descend[1], ToF_1_Descend[3], ToF_2_Descend[1]]]

            combine_3 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]],
                         [AoA_1_Descend[3], AoA_2_Descend[2], ToF_1_Descend[3], ToF_2_Descend[2]]]

            combine_4 = [[AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]],
                         [AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]],
                         [AoA_1_Descend[3], AoA_2_Descend[1], ToF_1_Descend[3], ToF_2_Descend[1]]]

            combine_5 = [[AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]],
                         [AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]],
                         [AoA_1_Descend[3], AoA_2_Descend[2], ToF_1_Descend[3], ToF_2_Descend[2]]]

            combine_6 = [[AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]],
                         [AoA_1_Descend[2], AoA_2_Descend[2], ToF_1_Descend[2], ToF_2_Descend[2]],
                         [AoA_1_Descend[3], AoA_2_Descend[1], ToF_1_Descend[3], ToF_2_Descend[1]]]
            combines = [combine_1, combine_2, combine_3, combine_4, combine_5, combine_6]
            coordinates = np.zeros((len(combines), len(combines[0]), 2))  # (x,y)
            rToFs = np.zeros((len(combines), len(combines[0]), 2))  # (rToF_1,rToF_2)
            for i in range(len(combines)):
                for j in range(len(combines[i])):
                    combine_cor = get_coordinate([combines[i][j][0], combines[i][j][1]])
                    coordinates[i, j] = combine_cor
                    rToFs[i, j] = get_rToF(combine_cor)

            deta_ToFs = []
            for i in range(len(rToFs)):
                sum = 0
                for j in range(len(rToFs[0])):
                    for num in range(2):
                        deta = (rToFs[i][j][num] - combines[i][j][num + 2]) ** 2
                        sum += deta
                deta_ToFs.append(sum)
            min_value = min(deta_ToFs)
            index = deta_ToFs.index(min_value)
            return [x for x in coordinates[index]]

        elif len(AoA_1_Descend) == 3 and len(AoA_2_Descend) == 4:
            combine_1 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[2], ToF_1_Descend[2], ToF_2_Descend[2]],
                         [AoA_1_Descend[2], AoA_2_Descend[3], ToF_1_Descend[2], ToF_2_Descend[3]]]

            combine_2 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[2], ToF_1_Descend[2], ToF_2_Descend[2]],
                         [AoA_1_Descend[1], AoA_2_Descend[3], ToF_1_Descend[1], ToF_2_Descend[3]]]

            combine_3 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]],
                         [AoA_1_Descend[2], AoA_2_Descend[3], ToF_1_Descend[2], ToF_2_Descend[3]]]

            combine_4 = [[AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]],
                         [AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]],
                         [AoA_1_Descend[1], AoA_2_Descend[3], ToF_1_Descend[1], ToF_2_Descend[3]]]

            combine_5 = [[AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]],
                         [AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]],
                         [AoA_1_Descend[2], AoA_2_Descend[3], ToF_1_Descend[2], ToF_2_Descend[3]]]

            combine_6 = [[AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[2], ToF_1_Descend[2], ToF_2_Descend[2]],
                         [AoA_1_Descend[1], AoA_2_Descend[3], ToF_1_Descend[1], ToF_2_Descend[3]]]
            combines = [combine_1, combine_2, combine_3, combine_4, combine_5, combine_6]
            coordinates = np.zeros((len(combines), len(combines[0]), 2))  # (x,y)
            rToFs = np.zeros((len(combines), len(combines[0]), 2))  # (rToF_1,rToF_2)
            for i in range(len(combines)):
                for j in range(len(combines[i])):
                    combine_cor = get_coordinate([combines[i][j][0], combines[i][j][1]])
                    coordinates[i, j] = combine_cor
                    rToFs[i, j] = get_rToF(combine_cor)

            deta_ToFs = []
            for i in range(len(rToFs)):
                sum = 0
                for j in range(len(rToFs[0])):
                    for num in range(2):
                        deta = (rToFs[i][j][num] - combines[i][j][num + 2]) ** 2
                        sum += deta
                deta_ToFs.append(sum)
            min_value = min(deta_ToFs)
            index = deta_ToFs.index(min_value)
            return [x for x in coordinates[index]]

        elif len(AoA_1_Descend) == 4 and len(AoA_2_Descend) == 2:
            combine_1 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]],
                         [AoA_1_Descend[3], AoA_2_Descend[1], ToF_1_Descend[3], ToF_2_Descend[1]]]

            combines = [combine_1]
            coordinates = np.zeros((len(combines), len(combines[0]), 2))  # (x,y)
            rToFs = np.zeros((len(combines), len(combines[0]), 2))  # (rToF_1,rToF_2)
            for i in range(len(combines)):
                for j in range(len(combines[i])):
                    combine_cor = get_coordinate([combines[i][j][0], combines[i][j][1]])
                    coordinates[i, j] = combine_cor
                    rToFs[i, j] = get_rToF(combine_cor)
            return [x for x in coordinates[0]]

        elif len(AoA_1_Descend) == 2 and len(AoA_2_Descend) == 4:
            combine_1 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]],
                         [AoA_1_Descend[1], AoA_2_Descend[3], ToF_1_Descend[1], ToF_2_Descend[3]]]

            combines = [combine_1]
            coordinates = np.zeros((len(combines), len(combines[0]), 2))  # (x,y)
            rToFs = np.zeros((len(combines), len(combines[0]), 2))  # (rToF_1,rToF_2)
            for i in range(len(combines)):
                for j in range(len(combines[i])):
                    combine_cor = get_coordinate([combines[i][j][0], combines[i][j][1]])
                    coordinates[i, j] = combine_cor
                    rToFs[i, j] = get_rToF(combine_cor)
            return [x for x in coordinates[0]]

        else:
            return []
    else:
        print('2 targets estimated')
        if len(AoA_1_Descend) == 3 and len(AoA_2_Descend) == 3:
            combine_1 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[2], ToF_1_Descend[2], ToF_2_Descend[2]]]

            combine_2 = [[AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]],
                         [AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]]]

            combine_1_cor1 = get_coordinate([combine_1[0][0], combine_1[0][1]])
            combine_1_cor2 = get_coordinate([combine_1[1][0], combine_1[1][1]])
            combine_2_cor1 = get_coordinate([combine_2[0][0], combine_2[0][1]])
            combine_2_cor2 = get_coordinate([combine_2[1][0], combine_2[1][1]])
            r11 = get_rToF(combine_1_cor1)
            r12 = get_rToF(combine_1_cor2)
            r21 = get_rToF(combine_2_cor1)
            r22 = get_rToF(combine_2_cor2)

            deta_1 = (r11[0] - combine_1[0][2]) ** 2 + (r11[1] - combine_1[0][3]) ** 2 + \
                     (r12[0] - combine_1[1][2]) ** 2 + (r12[1] - combine_1[1][3]) ** 2
            deta_2 = (r21[0] - combine_2[0][2]) ** 2 + (r21[1] - combine_2[0][3]) ** 2 + \
                     (r22[0] - combine_2[1][2]) ** 2 + (r22[1] - combine_2[1][3]) ** 2
            # print('deta_1 deta_2', deta_1, deta_2)
            if deta_1 < deta_2:
                return [combine_1_cor1, combine_1_cor2]

            else:
                return [combine_2_cor1, combine_2_cor2]

        elif len(AoA_1_Descend) == 2 and len(AoA_2_Descend) == 3:
            combine_1 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[1], AoA_2_Descend[2], ToF_1_Descend[1], ToF_2_Descend[2]]]
            # print(combine_1)
            combine_1_cor1 = get_coordinate([combine_1[0][0], combine_1[0][1]])
            combine_1_cor2 = get_coordinate([combine_1[1][0], combine_1[1][1]])
            return [combine_1_cor1, combine_1_cor2]


        elif len(AoA_1_Descend) == 3 and len(AoA_2_Descend) == 2:
            combine_2 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]],
                         [AoA_1_Descend[2], AoA_2_Descend[1], ToF_1_Descend[2], ToF_2_Descend[1]]]
            # print(combine_2)
            print('-------rx_2 lack one AoA-------')
            combine_2_cor1 = get_coordinate([combine_2[0][0], combine_2[0][1]])
            combine_2_cor2 = get_coordinate([combine_2[1][0], combine_2[1][1]])
            return combine_2_cor1, combine_2_cor2
        elif len(AoA_1_Descend) == 2 and len(AoA_2_Descend) == 2:
            print('-------rx_1 and rx_2 both lack one AoA-------')
            combine_2 = [[AoA_1_Descend[1], AoA_2_Descend[1], ToF_1_Descend[1], ToF_2_Descend[1]]]
            combine_2_cor1 = get_coordinate([combine_2[0][0], combine_2[0][1]])
            return [combine_2_cor1]

        elif len(AoA_1_Descend) == 1 and len(AoA_2_Descend) == 1:
            print('-------rx_1 and rx_2 no sense-------')
            return []


if __name__ == '__main__':
    for envir in range(0, 3, 1):
        for index in range(0, 5, 1):
            savePath='data/envir' +str(envir) + '/index'+str(envir)+'/'
            AoA_Target_paths = savePath+'AoA_ToF_target/'
            temp_paths = os.listdir(AoA_Target_paths)
            print('data lens are ', len(temp_paths))
            temp_paths.sort(key=lambda x: int(x[26:-4]))
            estimate_positions_seq=savePath+'estimate_positions_seq/'
            mkdir(estimate_positions_seq)

            for filename in temp_paths:
                print('----------filename is ', filename)
                # print(AoA_Target_paths+filename)
                with open(AoA_Target_paths + filename, 'rb') as f:
                    AoA_Target = pickle.load(f)
                AoA_list_1 = AoA_Target[0]
                AoA_list_2 = AoA_Target[1]
                ToF_list_1 = AoA_Target[2]
                ToF_list_2 = AoA_Target[3]
                coordinate_seq = []
                for i in range(len(AoA_list_1)):
                    combine_cors = get_position_combine(AoA_list_1[i], ToF_list_1[i], AoA_list_2[i], ToF_list_2[i])
                    coordinate_seq.append(combine_cors)
                x_, y_ = [], []
                for i in range(len(coordinate_seq)):
                    for j in range(len(coordinate_seq[i])):
                        x_.append(coordinate_seq[i][j][0])
                        y_.append(coordinate_seq[i][j][1])
                pylab.figure()
                pylab.scatter(x_, y_)
                pylab.xlim(-0.5, 4.5)
                pylab.ylim(4, 0)
                pylab.savefig(estimate_positions_seq + filename[0:-4] + '.jpg')
                pylab.show()
                pylab.close()




