import pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sc
import copy
import pickle
import os




packet_frequency = 1000

center_frequency = 5.5e9

speed_of_light = 299792458

antDistance = 2.727e-2

rx_antenna_num = 3
music_rx_antenna = 7
subcarrier_num = 30
subCarrierIndex40 = np.array([-58, -54, -50, -46, -42, -38, -34, -30, -26, -22, -18, -14, -10, -6, -2,
                              2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58])

subCarrierIndex20 = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                              1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28])
subcarrier_inteval = 3.125e5
default_signal=6

phase_shift_ToF_init = -1j * 2 * np.pi * subcarrier_inteval * subCarrierIndex20  * 1e-9
phase_shift_AoA_init =-1j * 2 * np.pi * antDistance / speed_of_light * center_frequency

coordinate_rx_1=[0,0]
coordinate_rx_2=[4,0]
coordinate_tx=[2,0]

class Track(object):
    def __init__(self,
                 search_space_AoA=(-np.pi / 2, np.pi / 2),
                 search_space_ToF=(0, 100),
                 slide_window_proportion=0.4,
                 CSI_matrix=None,
                 use_mdl=1,
                 use_pca=1,
                 use_40mhz=0,
                 packet_start=0,
                 packet_end=100,
                 first_run=1,
                 rx1_or_not=1,
                 ):

        self.search_space_AoA = search_space_AoA
        self.search_space_ToF = search_space_ToF
        self.slide_window_proportion = slide_window_proportion
        self.CSI_matrix = CSI_matrix
        self.use_mdl = use_mdl
        self.use_pca = use_pca
        self.use_40mhz = use_40mhz
        self.packet_start = packet_start
        self.packet_end = packet_end
        self.first_run = first_run
        self.rx1_or_not = rx1_or_not

        self.search_space_interval_num_AoA = int(
            (self.search_space_AoA[1] - self.search_space_AoA[0]) / np.pi * 180)
        self.search_space_interval_AoA = (self.search_space_AoA[1] - self.search_space_AoA[
            0]) / self.search_space_interval_num_AoA
        self.search_space_angle_AoA = np.arange(self.search_space_AoA[0], self.search_space_AoA[1],
                                                self.search_space_interval_AoA)

        self.search_space_interval_num_ToF = int((self.search_space_ToF[1] - self.search_space_ToF[0]) * 1)
        self.search_space_interval_ToF = (self.search_space_ToF[1] - self.search_space_ToF[
            0]) / self.search_space_interval_num_ToF

        self.search_space_ToF = np.arange(self.search_space_ToF[0], self.search_space_ToF[1],
                                          self.search_space_interval_ToF)
        self.slide_window_len = int(self.slide_window_proportion * packet_frequency)
        self.step_len = self.slide_window_len
        self.overlap_len = self.slide_window_len - self.step_len


        '''read the PLL phase compensation file'''
        with open('initphase_diff_3rx_receiver_1.pkl', 'rb') as f:
            self.initPhase_diff_1 = pickle.load(f)

        with open('initphase_diff_3rx_receiver_2.pkl', 'rb') as f:
            self.initPhase_diff_2 = pickle.load(f)

        self.precompute_steering_matrix()

    def getSteeringVector(self, one_of_angle, one_of_ToF, center_frequency):
        phase_shift_ToF = np.exp(one_of_ToF * phase_shift_ToF_init).reshape(-1,1)
        phase_shift_AoA=np.exp(np.arange(music_rx_antenna).reshape(-1,1)* np.sin(one_of_angle)*phase_shift_AoA_init)
        steering_vector=np.kron(phase_shift_AoA,phase_shift_ToF)

        return steering_vector

    def mdl_algorithm(self, eigenvalues):

        mdl = np.zeros(music_rx_antenna)
        lambda_tot = eigenvalues[:music_rx_antenna]
        sub_arr_size = music_rx_antenna
        n_segments = self.slide_window_len
        max_multipath = len(lambda_tot)
        for k in range(0, max_multipath):
            mdl[k] = -n_segments * (sub_arr_size - k) * np.log(sc.gmean(lambda_tot[k:]) / np.mean(lambda_tot[k:])) \
                     + 0.5 * k * (2 * sub_arr_size - k) * np.log(n_segments)
        index = max(np.argmin(mdl), 1)
        index = min(index, 4)
        print('mdl', mdl, index)
        return index


    def get_noise_subspace(self, matrix):


        mat = np.asarray(matrix).transpose()  # mat  Nrx * packages
        R_x = np.dot(mat, mat.conjugate().transpose())

        eignvalue, eigenvector = np.linalg.eig(R_x)
        eignvalue = np.abs(eignvalue)

        un = np.argsort(-eignvalue)
        eigenvalue_sort = -np.sort(-eignvalue)
        eigenvector_sort = eigenvector[:, un[:]]

        if self.use_mdl:
            incident_num = self.mdl_algorithm(eigenvalue_sort)
        else:
            incident_num = default_signal


        E_N = eigenvector_sort[:, incident_num:]

        return E_N, incident_num


    def findPeak2D(self, spectrum, incident_num):
        center = spectrum[1:-1, 1:-1]
        peak_mask = (
                (center > spectrum[:-2, 1:-1]) &
                (center > spectrum[2:, 1:-1]) &
                (center > spectrum[1:-1, :-2]) &
                (center > spectrum[1:-1, 2:])
        )
        index_i, index_j = np.where(peak_mask)
        index_i += 1
        index_j += 1
        peakIndexes = np.column_stack((index_i, index_j))
        peakValues = spectrum[index_i, index_j]

        AoA_interval=10
        spectral_interval=AoA_interval/180*self.search_space_interval_num_AoA

        Z = zip(peakValues, peakIndexes)
        Zipped = sorted(Z, reverse=True)

        valueDescend, indexDescend = zip(*Zipped)
        indexDescend=list(indexDescend)
        valueDescend=list(valueDescend)
        max_value=valueDescend[0]
        new_indexDescend=[indexDescend[0]]
        print(valueDescend)
        ratio=0.4
        for i in range(1,len(indexDescend)):
            save_flag = 1
            for j in range(len(new_indexDescend)):
                if abs(indexDescend[i][0]-new_indexDescend[j][0])<=spectral_interval:
                    save_flag = 0
                    break
            if save_flag == 1 and valueDescend[i]>=max_value*ratio :
                new_indexDescend.append(indexDescend[i])


        if len(peakIndexes) < incident_num:
            peakIndexes = new_indexDescend
        else:
            peakIndexes = new_indexDescend[0:incident_num]

        return peakIndexes


    def initPhase_Calibration(self, CSIMatrix, receiver_index):
        'compensate PLL phase error'
        if receiver_index == 1:
            CSIMatrix_Cal = np.zeros(CSIMatrix.shape, dtype=complex)
            fi_31 = np.exp(1j * self.initPhase_diff_1[0])
            fi_32 = np.exp(1j * self.initPhase_diff_1[1])
            CSIMatrix_Cal[0] = np.dot(np.diag(fi_31), CSIMatrix[0])
            CSIMatrix_Cal[1] = np.dot(np.diag(fi_32), CSIMatrix[1])
            CSIMatrix_Cal[2] = CSIMatrix[2] * 3
            return CSIMatrix_Cal
        if receiver_index == 2:
            CSIMatrix_Cal = np.zeros(CSIMatrix.shape, dtype=complex)
            fi_12 = np.exp(1j * self.initPhase_diff_1[2])
            fi_13 = np.exp(1j * self.initPhase_diff_1[3])
            CSIMatrix_Cal[0] = CSIMatrix[0] * 3
            CSIMatrix_Cal[1] = np.dot(np.diag(fi_12), CSIMatrix[1])
            CSIMatrix_Cal[2] = np.dot(np.diag(fi_13), CSIMatrix[2])
            return CSIMatrix_Cal
        if receiver_index == 3:
            CSIMatrix_Cal = np.zeros(CSIMatrix.shape, dtype=complex)
            fi_12 = np.exp(1j * self.initPhase_diff_1[4])
            fi_13 = np.exp(1j * self.initPhase_diff_1[5])
            CSIMatrix_Cal[0] = CSIMatrix[0] * 3
            CSIMatrix_Cal[1] = np.dot(np.diag(fi_12), CSIMatrix[1])
            CSIMatrix_Cal[2] = np.dot(np.diag(fi_13), CSIMatrix[2])
            return CSIMatrix_Cal
        if receiver_index == 4:
            CSIMatrix_Cal = np.zeros(CSIMatrix.shape, dtype=complex)
            fi_31 = np.exp(1j * self.initPhase_diff_2[0])
            fi_32 = np.exp(1j * self.initPhase_diff_2[1])
            CSIMatrix_Cal[0] = np.dot(np.diag(fi_31), CSIMatrix[0])
            CSIMatrix_Cal[1] = np.dot(np.diag(fi_32), CSIMatrix[1])
            CSIMatrix_Cal[2] = CSIMatrix[2] * 3
            return CSIMatrix_Cal
        if receiver_index == 5:
            CSIMatrix_Cal = np.zeros(CSIMatrix.shape, dtype=complex)
            fi_12 = np.exp(1j * self.initPhase_diff_2[2])
            fi_13 = np.exp(1j * self.initPhase_diff_2[3])
            CSIMatrix_Cal[0] = CSIMatrix[0] * 3
            CSIMatrix_Cal[1] = np.dot(np.diag(fi_12), CSIMatrix[1])
            CSIMatrix_Cal[2] = np.dot(np.diag(fi_13), CSIMatrix[2])
            return CSIMatrix_Cal
        if receiver_index == 6:
            CSIMatrix_Cal = np.zeros(CSIMatrix.shape, dtype=complex)
            fi_12 = np.exp(1j * self.initPhase_diff_2[4])
            fi_13 = np.exp(1j * self.initPhase_diff_2[5])
            CSIMatrix_Cal[0] = CSIMatrix[0] * 3
            CSIMatrix_Cal[1] = np.dot(np.diag(fi_12), CSIMatrix[1])
            CSIMatrix_Cal[2] = np.dot(np.diag(fi_13), CSIMatrix[2])
            return CSIMatrix_Cal

    def phaseCalibration_improve(self,csi, subCarrierIndex, rxNum, subCarrierNum):
        'remove SFO and PDD'
        phaseRaw = np.angle(csi)
        phaseUnwrapped = np.unwrap(phaseRaw)

        for antIndexForPhase in range(1, rxNum):
            if phaseUnwrapped[antIndexForPhase, 0] - phaseUnwrapped[0, 0] > np.pi:
                phaseUnwrapped[antIndexForPhase, :] -= 2 * np.pi
            elif phaseUnwrapped[antIndexForPhase, 0] - phaseUnwrapped[0, 0] < -np.pi:
                phaseUnwrapped[antIndexForPhase, :] += 2 * np.pi

        phase = phaseUnwrapped.reshape(-1)
        a_mat = np.tile(subCarrierIndex, (1, rxNum))
        a_mat = np.append(a_mat, np.ones((1, subCarrierNum * rxNum)), axis=0)
        a_mat = a_mat.transpose((1, 0))
        a_mat_inv = np.linalg.pinv(a_mat)
        x = np.dot(a_mat_inv, phase)
        phaseSlope = x[0]
        phaseCons = x[1]
        calibration = np.exp(1j * (-phaseSlope * np.tile(subCarrierIndex, rxNum).reshape(rxNum, -1)))
        csi = csi * calibration
        return csi

    def fillCSIMatrix_improve(self, CSImatrix, receiver_index):

        subCarrierIndex = subCarrierIndex40 if self.use_40mhz else subCarrierIndex20
        CSImatrix_cal = self.initPhase_Calibration(CSImatrix, receiver_index)

        CSImatrix_cal = np.transpose(CSImatrix_cal, [2, 0, 1])
        for filecount in range(len(CSImatrix_cal)):
            CSImatrix_cal[filecount] = \
                self.phaseCalibration_improve(CSImatrix_cal[filecount], subCarrierIndex, rxNum=rx_antenna_num,
                                         subCarrierNum=subcarrier_num)

        return CSImatrix_cal


    def CFO_Calibration_new(self, CSIMatrix, index1, index2, index3):
        'ratio of shared antenna to align the phase offsets and AGC noise among different NICs'

        for pakage in CSIMatrix:
            complex_deta_1_2 = pakage[index1] / pakage[index2]
            complex_deta_1_3 = pakage[index1] / pakage[index3]

            pakage[index2] = pakage[index2] * complex_deta_1_2
            pakage[index2 + 1] = pakage[index2 + 1] * complex_deta_1_2
            pakage[index2 + 2] = pakage[index2 + 2] * complex_deta_1_2
            #
            pakage[index3] = pakage[index3] * complex_deta_1_3
            pakage[index3 + 1] = pakage[index3 + 1] * complex_deta_1_3
            pakage[index3 + 2] = pakage[index3 + 2] * complex_deta_1_3

        CSIMatrix = np.delete(CSIMatrix, [index2, index3], axis=1)
        print('CFO_Calibration_AmpPhase CSIMatrix shape is', CSIMatrix.shape)
        return CSIMatrix


    def precompute_steering_matrix(self):
        angles = np.asarray(self.search_space_angle_AoA)
        tofs = np.asarray(self.search_space_ToF)

        N_AoA = len(angles)
        N_ToF = len(tofs)

        # AoA
        antenna_index = np.arange(
            music_rx_antenna
        ).reshape(-1, 1)

        steering_AoA = np.exp(
            antenna_index
            * np.sin(angles).reshape(1, -1)
            * phase_shift_AoA_init
        )

        # ToF
        tof_init = np.asarray(
            phase_shift_ToF_init
        ).reshape(-1, 1)

        steering_ToF = np.exp(
            tof_init
            * tofs.reshape(1, -1)
        )

        N_subcarrier = steering_ToF.shape[0]

        # AoA × ToF
        S = (
                steering_AoA[:, :, None, None]
                * steering_ToF[None, None, :, :]
        )

        S = S.transpose(
            0, 2, 1, 3
        ).reshape(
            music_rx_antenna * N_subcarrier,
            N_AoA * N_ToF
        )

        self.music_steering_matrix = S
        self.music_N_AoA = N_AoA
        self.music_N_ToF = N_ToF

    def get_AoA_spectrum(self, CSIMatrix):
        E_N, incident_num = self.get_noise_subspace(CSIMatrix)
        S = self.music_steering_matrix
        projected = E_N.conj().T @ S

        # || E_N^H a ||^2
        denominator = np.sum(np.abs(projected) ** 2,axis=0)

        spectrum = 1.0 / np.maximum(denominator,1e-12)

        AoA_TOF_Spectrum=spectrum.reshape(
            self.music_N_AoA,
            self.music_N_ToF)


        return AoA_TOF_Spectrum, incident_num


    def convert(self, csi):
        """Remove outliers """
        index = []
        csi_copy = copy.deepcopy(csi)
        for i in range(len(csi)):
            for j in range(len(csi[0])):
                if sum(abs(csi[i][j])) > 1e100 or sum(abs(csi[i][j])) < 1e-10:
                    # print(abs(csi[i][j]))
                    index.append(i)
                    break
        csi_copy = np.delete(csi_copy, index, axis=0)
        return csi_copy

    def get_AoA_ToF(self, cal_angel_1,cal_angel_2):


        receivers = int(len(self.CSI_matrix) / 3)
        print('the number of receivers is ', receivers)
        fileLen = len(CSI_matrix[0][0])
        print('CSI package len is ', fileLen)

        window_start = 0
        csi_matrix_rx1 = np.zeros([self.slide_window_len, rx_antenna_num, subcarrier_num], dtype=complex)
        csi_matrix_rx2 = np.zeros([self.slide_window_len, rx_antenna_num, subcarrier_num], dtype=complex)
        csi_matrix_rx3 = np.zeros([self.slide_window_len, rx_antenna_num, subcarrier_num], dtype=complex)
        csi_matrix_rx4 = np.zeros([self.slide_window_len, rx_antenna_num, subcarrier_num], dtype=complex)
        csi_matrix_rx5 = np.zeros([self.slide_window_len, rx_antenna_num, subcarrier_num], dtype=complex)
        csi_matrix_rx6 = np.zeros([self.slide_window_len, rx_antenna_num, subcarrier_num], dtype=complex)

        window_num_count = 0
        AoA_list_1, AoA_list_2 = [], []
        ToF_list_1, ToF_list_2 = [], []
        while window_start + self.slide_window_len <= fileLen:  # two Receiver file

            for i in range(receivers):
                receiver_index = i + 1
                if receiver_index == 1:
                    if window_start == 0:
                        csi_matrix_rx1 = self.fillCSIMatrix_improve(self.CSI_matrix[0:3, :, 0: self.slide_window_len],
                                                                    receiver_index)
                    else:
                        if self.overlap_len == 0:
                            csi_matrix_rx1 = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[0:3, :, window_start: window_start + self.slide_window_len],
                                 receiver_index)
                        else:
                            csi_matrix_rx1[:self.overlap_len, :, :] = csi_matrix_rx1[-self.overlap_len:, :, :]
                            csi_matrix_rx1[-self.step_len:, :, :] = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[0:3, :,
                                 window_start + self.overlap_len: window_start + self.slide_window_len],
                                 receiver_index)
                if receiver_index == 2:
                    if window_start == 0:
                        csi_matrix_rx2 = self.fillCSIMatrix_improve(self.CSI_matrix[3:6, :, 0: self.slide_window_len],
                                                                    receiver_index)
                    else:
                        if self.overlap_len == 0:
                            csi_matrix_rx2 = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[3:6, :, window_start: window_start + self.slide_window_len],
                                 receiver_index)
                        else:
                            csi_matrix_rx2[:self.overlap_len, :, :] = csi_matrix_rx2[-self.overlap_len:, :, :]
                            csi_matrix_rx2[-self.step_len:, :, :] = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[3:6, :,
                                 window_start + self.overlap_len: window_start + self.slide_window_len],
                                 receiver_index)
                if receiver_index == 3:
                    if window_start == 0:
                        csi_matrix_rx3 = self.fillCSIMatrix_improve(self.CSI_matrix[6:9, :, 0: self.slide_window_len],
                                                                    receiver_index)
                    else:
                        if self.overlap_len == 0:
                            csi_matrix_rx3 = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[6:9, :, window_start: window_start + self.slide_window_len],
                                 receiver_index)
                        else:
                            csi_matrix_rx3[:self.overlap_len, :, :] = csi_matrix_rx3[-self.overlap_len:, :, :]
                            csi_matrix_rx3[-self.step_len:, :, :] = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[6:9, :,
                                 window_start + self.overlap_len: window_start + self.slide_window_len],
                                 receiver_index)
                if receiver_index == 4:
                    if window_start == 0:
                        csi_matrix_rx4 = self.fillCSIMatrix_improve(self.CSI_matrix[9:12, :, 0: self.slide_window_len],
                                                                    receiver_index)
                    else:
                        if self.overlap_len == 0:
                            csi_matrix_rx4 = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[9:12, :, window_start: window_start + self.slide_window_len],
                                 receiver_index)
                        else:
                            csi_matrix_rx4[:self.overlap_len, :, :] = csi_matrix_rx4[-self.overlap_len:, :, :]
                            csi_matrix_rx4[-self.step_len:, :, :] = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[9:12, :,
                                 window_start + self.overlap_len: window_start + self.slide_window_len],
                                 receiver_index)
                if receiver_index == 5:
                    if window_start == 0:
                        csi_matrix_rx5 = self.fillCSIMatrix_improve(self.CSI_matrix[12:15, :, 0: self.slide_window_len],
                                                                    receiver_index)
                    else:
                        if self.overlap_len == 0:
                            csi_matrix_rx5 = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[12:15, :, window_start: window_start + self.slide_window_len],
                                 receiver_index)
                        else:
                            csi_matrix_rx5[:self.overlap_len, :, :] = csi_matrix_rx5[-self.overlap_len:, :, :]
                            csi_matrix_rx5[-self.step_len:, :, :] = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[12:15:,
                                 window_start + self.overlap_len: window_start + self.slide_window_len],
                                 receiver_index)
                if receiver_index == 6:
                    if window_start == 0:
                        csi_matrix_rx6 = self.fillCSIMatrix_improve(self.CSI_matrix[15:18, :, 0: self.slide_window_len],
                                                                    receiver_index)
                    else:
                        if self.overlap_len == 0:
                            csi_matrix_rx6 = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[15:18, :, window_start: window_start + self.slide_window_len],
                                 receiver_index)
                        else:
                            csi_matrix_rx6[:self.overlap_len, :, :] = csi_matrix_rx6[-self.overlap_len:, :, :]
                            csi_matrix_rx6[-self.step_len:, :, :] = self.fillCSIMatrix_improve \
                                (self.CSI_matrix[15:18, :,
                                 window_start + self.overlap_len: window_start + self.slide_window_len],
                                 receiver_index)

            csi_matrix_all_1 = np.concatenate((csi_matrix_rx1, csi_matrix_rx2, csi_matrix_rx3), axis=1)
            csi_matrix_all_2 = np.concatenate((csi_matrix_rx4, csi_matrix_rx5, csi_matrix_rx6), axis=1)



            "不同网卡上CFO一致化"
            csi_matrix_all_1 = self.CFO_Calibration_new(csi_matrix_all_1, 2, 3, 6)
            csi_matrix_all_2 = self.CFO_Calibration_new(csi_matrix_all_2, 2, 3, 6)

            "去除离群点操作"
            csi_matrix_all_1 = np.nan_to_num(csi_matrix_all_1)
            csi_matrix_all_1 = self.convert(csi_matrix_all_1)

            csi_matrix_all_2 = np.nan_to_num(csi_matrix_all_2)
            csi_matrix_all_2 = self.convert(csi_matrix_all_2)
            print('csi_matrix_all 1 2 shape is ', csi_matrix_all_1.shape, csi_matrix_all_2.shape)


            CSIMatrixx_1 = np.zeros([csi_matrix_all_1.shape[0], 30 * music_rx_antenna], dtype=complex)
            CSIMatrixx_2 = np.zeros([csi_matrix_all_2.shape[0], 30 * music_rx_antenna], dtype=complex)

            for csi_index in range(len(csi_matrix_all_1)):
                antenna_reshape = np.zeros([30 * music_rx_antenna], dtype=complex)
                for antenna in range(music_rx_antenna):
                    ant = csi_matrix_all_1[csi_index, antenna, :]
                    antenna_reshape[antenna * 30:(antenna + 1) * 30] = ant
                CSIMatrixx_1[csi_index] = antenna_reshape


            for csi_index in range(len(csi_matrix_all_2)):
                antenna_reshape = np.zeros([30 * music_rx_antenna], dtype=complex)
                for antenna in range(music_rx_antenna):
                    ant = csi_matrix_all_2[csi_index, antenna, :]
                    antenna_reshape[antenna * 30:(antenna + 1) * 30] = ant
                CSIMatrixx_2[csi_index] = antenna_reshape


            AoASpectrum_1, incident_num_1 = self.get_AoA_spectrum(CSIMatrixx_1)
            AoASpectrum_2, incident_num_2 = self.get_AoA_spectrum(CSIMatrixx_2)



            peak_angle_index_1 = self.findPeak2D(AoASpectrum_1, incident_num_1)
            peak_angle_index_2 = self.findPeak2D(AoASpectrum_2, incident_num_1)

            AoA_1,AoA_2=[],[]
            ToF_1,ToF_2=[],[]
            for i in range(len(peak_angle_index_1)):
                AoA = self.search_space_angle_AoA[peak_angle_index_1[i][0]] / np.pi * 180 - cal_angel_1
                ToF = self.search_space_ToF[peak_angle_index_1[i][1]]
                AoA_1.append(round(AoA, 2))
                ToF_1.append(round(ToF, 2))
                print('rx1 AoA ToF : ', i, round(AoA, 2), round(ToF, 2))
            # print('------------------')
            for i in range(len(peak_angle_index_2)):
                AoA = self.search_space_angle_AoA[peak_angle_index_2[i][0]] / np.pi * 180 - cal_angel_2
                ToF = self.search_space_ToF[peak_angle_index_2[i][1]]
                AoA_2.append(round(AoA, 2))
                ToF_2.append(round(ToF, 2))
                print('rx2 AoA ToF: ', i, round(AoA, 2), round(ToF, 2))


            window_start += self.step_len
            window_num_count += 1

        return AoA_list_1,AoA_list_2,ToF_list_1,ToF_list_2

if __name__ == '__main__':

    for envir in range(0, 3, 1):
        for index in range(0, 5, 1):
            CSI_SavePath ='data/envir' +str(envir) + '/index'+str(envir)+'/'
            temp_paths = os.listdir(CSI_SavePath)
            print('data lens are ', len(temp_paths))
            temp_paths.sort(key=lambda x: int(x[15:-4]))
            for filename in temp_paths:
                print('filename is ', filename)
                with open(CSI_SavePath + filename, 'rb') as f:
                    CSI_matrix = pickle.load(f)
                print('CSI_matrix shape is ', CSI_matrix.shape)
                'array rotation angle'
                cal_angel_1 = 45
                cal_angel_2 = 45
                print('CSI_matrix shape is ', CSI_matrix.shape)
                rx = Track(search_space_AoA=(-np.pi / 2, np.pi / 2), search_space_ToF=(-100, 100),
                            slide_window_proportion=0.2,
                            CSI_matrix=CSI_matrix, use_pca=1, use_mdl=1,
                            use_40mhz=0, packet_start=0, packet_end=CSI_matrix.shape[-1],
                            first_run=1, rx1_or_not=0)

                AoA_list_1, AoA_list_2, ToF_list_1, ToF_list_2 = rx.get_AoA_ToF(cal_angel_1,cal_angel_2)

                AoA_ToF = [AoA_list_1, AoA_list_2, ToF_list_1, ToF_list_2]
                with open(CSI_SavePath+'AoA_ToF.pkl', 'wb') as handle:
                    pickle.dump(AoA_ToF, handle, -1)













