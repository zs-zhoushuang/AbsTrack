import numpy as np
import tensorflow as tf
from transformer_revise import Transformer
import random
import os
import pickle
import time
import pylab
from scipy.signal import savgol_filter

class DETR():
    def __init__(self,train_data=None,test_data=None, test_NoTrainset=None,verbose=True,pe_input=256,
                 max_target=3, n_classes=4, fc_activation=None,
                 hidden_dim=128,nheads=8,num_encoder_layers=6,num_decoder_layers=6,
                 windows=20,n_query_pos=10,batch_size=256):
        self.train_data=train_data
        self.test_data=test_data
        self.test_NoTrainset=test_NoTrainset
        self.verbose = verbose
        self.pe_input=pe_input
        self.max_target=max_target
        self.n_classes = n_classes+1
        self.fc_activation = fc_activation

        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.windows=windows
        self.n_query_pos = n_query_pos
        self.batch_size=batch_size
        self.training_epochs = 300
        self.lr=0.00001



        self.is_train = False
        self.param_file = False

        self.build()
        print("Neural networks build!")
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.is_train is True:
            if self.param_file is True:
                self.saver.restore(self.sess, "./params_300/train.ckpt")
                print("loading neural-network params...")
                self.learn()
            else:
                print("learning initialization!")
                self.learn()
        else:

            self.saver.restore(self.sess, "params_300/train.ckpt")
            print('load params sucess')

            self.test_ac(test_dataset=self.test_data,epoches=300,Draw_picture=True)



    def build(self):

        self.input = tf.placeholder(tf.float32, shape=[None, self.windows*self.max_target, 2], name='meaurement_input')
        self.input_embiding = tf.placeholder(tf.float32, shape=[None, self.windows * self.max_target, self.hidden_dim],name='input_embiding')
        self.tag_cla = tf.placeholder(tf.float32, shape=[None, self.n_query_pos, self.n_classes], name='target_classification')
        self.position_pre = tf.placeholder(tf.float32, shape=[None,self.n_query_pos, 2], name='position_prediction')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.keep_prob_1 = tf.placeholder(tf.float32, name='keep_prob_1')
        self.class_output, self.position_output = self.__make_transformer_top()
        print("Neural networks build!")

        with tf.variable_scope('loss'):
            """
            classification and prediction losses
            """
            self.class_loss=tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.tag_cla, logits=self.class_output))

            self.pre_loss=tf.reduce_sum(tf.square(self.position_pre-self.position_output))

            self.loss = self.class_loss + self.pre_loss

        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def __make_transformer_top(self):

        h = tf.layers.conv1d(self.input, self.hidden_dim, kernel_size=1, strides=1,
                     padding='same',kernel_initializer='he_normal',use_bias=True,data_format='channels_last')
        print(h.shape,11111111111111)
        query_pos = self.get_trainable_parameter(shape=(self.n_query_pos, self.hidden_dim))
        temp_input = self.input_embiding+h
        print('input shape', temp_input.shape)



        h_tag = tf.transpose(h,perm=[0, 2, 1])
        if self.verbose: print('h_tag transpose1',h_tag.shape)
        h_tag=tf.layers.conv1d(h_tag,query_pos.shape[0],kernel_size=1,strides=1,
                    padding='same',kernel_initializer='he_normal',
                    use_bias=True,data_format='channels_last')

        if self.verbose: print('h_tag conv',h_tag.shape)
        h_tag = tf.transpose(h_tag,perm=[0, 2, 1])
        if self.verbose: print('h_tag transpose2',h_tag.shape)

        query_pos = tf.expand_dims(query_pos,0)
        if self.verbose: print('query_pos',query_pos.shape)
        query_pos+=h_tag
        query_pos-=h_tag
        if self.verbose: print('query_pos+-', query_pos.shape)


        transformer = Transformer(pe_input = 256, pe_target = 100, num_layers = self.num_encoder_layers,
                                  d_model=self.hidden_dim,num_heads = self.nheads,dff = 2048,
                                  keep_prob = self.keep_prob_1,input_vocab_size = 256, target_vocab_size = 100,
                                  num_classes=self.n_classes,inp=temp_input, tar=query_pos,training=self.is_train)

        self.out, attention_weights = transformer.build()

        print('transform output shape ',self.out.shape)#[-1,10,128]

        class_output = self.linear_class(self.out, name='class')
        bbox_output = self.linear_bbox(self.out, name='box')
        bbox_output = tf.nn.sigmoid(bbox_output)
        return class_output,bbox_output


    def get_trainable_parameter(self, shape=(100, 128)):
        w_init = tf.random_normal_initializer()
        parameter = tf.Variable(
            initial_value=w_init(shape=shape, dtype='float32'), trainable=True)
        return parameter


    def linear_class(self, out, name=None):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(out, 128, activation=tf.nn.relu,
                                     kernel_initializer=tf.glorot_uniform_initializer,
                                     name='fc1')
            ffn_output = tf.layers.dense(layer1, self.n_classes, kernel_initializer=tf.glorot_uniform_initializer,
                                         name='fc2')
            return ffn_output


    def linear_bbox(self, out, name=None):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(out, self.hidden_dim, activation=tf.nn.relu,
                                     kernel_initializer=tf.glorot_uniform_initializer,
                                     name='fc1')
            ffn_output = tf.layers.dense(layer1, 2, kernel_initializer=tf.glorot_uniform_initializer,
                                         name='fc2')
            return ffn_output

    def get_angles(self,pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self,position, d_model,Seq_embiding):
        all_pos_encoding=None
        for i in range(len(Seq_embiding)):
            angle_rads = self.get_angles(Seq_embiding[i, :, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)


            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

            pos_encoding = angle_rads[np.newaxis, ...]
            all_pos_encoding = np.array(pos_encoding) if all_pos_encoding is None \
                else np.append(all_pos_encoding, pos_encoding, axis=0)

        return all_pos_encoding

    def tracking_error(self, ground_truth, trajectories):
        ground_truth_1 = ground_truth * 4
        trajectories_1 = trajectories * 4
        error_matrix = trajectories_1 - ground_truth_1
        error_sum = 0
        for i in range(len(error_matrix)):
            error_track = 0
            for j in range(len(error_matrix[i])):
                temp = np.sqrt(error_matrix[i][j][0] ** 2 + error_matrix[i][j][1] ** 2) * 100
                error_track += temp
            error_sum += error_track

        return error_sum / (len(error_matrix) * len(error_matrix[0]))
    def test_ac(self, test_dataset=None):
        all_input_test, all_class_test, all_position_test, all_ground_truth, all_Seq_embiding = [], [], [], [], []
        count = 0
        for one_data in test_dataset:
            split_data, tar_class, tar_position, Seq_embiding, _ = self.split(one_data, count)
            all_input_test.append(split_data)
            all_class_test.append(tar_class)
            all_position_test.append(tar_position)
            all_ground_truth.append(one_data[1][:, self.windows:, :])
            all_Seq_embiding.append(Seq_embiding)

        correct_count = 0
        test_sum = 0

        track_error_sum = 0
        track_num_all = 0
        transformer_error_data = []
        for num in range(len(all_input_test)):
            class_output, position_output = self.sess.run([self.class_output, self.position_output],
                                                          feed_dict={self.input: all_input_test[num],
                                                                     self.input_embiding: all_Seq_embiding[num],
                                                                     self.keep_prob_1: 1.0, })
            ''''calculate class accuracy'''
            class_pre = np.argmax(class_output, axis=-1)
            class_target = np.argmax(all_class_test[num], axis=-1)
            test_sum += class_pre.shape[0] * class_pre.shape[1]
            correct_count += np.sum(class_pre == class_target)

            '''calculate tracking error'''
            track_id = []
            track_time = []
            track_positions = []
            for i in range(len(class_pre)):
                for j in range(self.n_query_pos):
                    if class_pre[i, j] != self.n_classes - 1:
                        track_id.append(class_pre[i, j])
                        track_time.append(i)
                        track_positions.append(position_output[i, j])

            track_num = []
            [track_num.append(x) for x in track_id if x not in track_num]
            track_num = sorted(track_num)
            track_id_count = []
            for i in range(len(track_num)):
                track_id_count.append(track_id.count(track_num[i]))

            exceeding_num = len(track_num) - len(all_ground_truth[num])
            if exceeding_num > 0:
                for i in range(exceeding_num):
                    index = track_id_count.index(min(track_id_count))
                    track_id_count.pop(index)
                    track_num.pop(index)

            if len(track_num) != 0:
                Trajectory_Matrix = np.zeros((len(track_num), len(all_ground_truth[num][0]), 2))

                for i in range(len(track_id)):
                    if track_num.count(track_id[i]) > 0:
                        save_index = track_num.index(track_id[i])
                        Trajectory_Matrix[save_index][track_time[i]] = track_positions[i]
                for i in range(len(Trajectory_Matrix)):
                    for j in range(1, len(Trajectory_Matrix[0])):
                        if Trajectory_Matrix[i][j][0] == 0 and Trajectory_Matrix[i][j][1] == 0:
                            Trajectory_Matrix[i][j] = Trajectory_Matrix[i][j - 1]
                track_error = self.tracking_error(all_ground_truth[num], Trajectory_Matrix)
                transformer_error_data.append(track_error)
                track_error_sum += track_error
                track_num_all += 1



        print('class prediction accuracy ', correct_count / test_sum)
        print('tracking error ', track_error_sum / track_num_all)

        return correct_count / test_sum, track_error_sum / track_num_all

    def split(self, Seq_and_Label, count):
        measure_Seq = Seq_and_Label[0]
        ground_truth = Seq_and_Label[1]
        track_id = Seq_and_Label[2]

        one_length = len(measure_Seq) - self.windows
        count += one_length
        split_data = np.zeros((one_length, self.max_target * self.windows, 2))
        in_embiding = np.zeros((one_length, self.max_target * self.windows))
        Seq_mask = np.zeros((one_length, self.max_target * self.windows))
        for i in range(0, one_length):
            index_count = 0
            for j in range(self.windows):
                for measure_index in range(len(measure_Seq[i + j])):
                    split_data[i, index_count] = measure_Seq[i + j][measure_index]
                    in_embiding[i, index_count] = i + j + 1
                    Seq_mask[i, index_count] = 1
                    index_count += 1

        tar_class = np.zeros((one_length, self.n_query_pos, self.n_classes))
        tar_position = np.zeros((one_length, self.n_query_pos, 2))
        for i in range(0, one_length):
            for j in range(self.n_query_pos):
                onehot_vector = np.zeros([self.n_classes])
                if j <= np.max(track_id[i + self.windows]):
                    onehot_vector[j] = 1
                    tar_class[i, j] = onehot_vector
                    tar_position[i, j] = ground_truth[j, i + self.windows]
                else:
                    onehot_vector[-1] = 1
                    tar_class[i, j] = onehot_vector
                    tar_position[i, j] = np.zeros(2)

        Seq_embiding = self.positional_encoding(self.max_target * self.windows, self.hidden_dim, in_embiding)

        return split_data, tar_class, tar_position, Seq_embiding, count

    def learn(self):

        all_seq_input = np.empty((200000, self.max_target * self.windows, 2))
        all_tar_class = np.empty((200000, self.n_query_pos, self.n_classes))
        all_tar_postions = np.empty((200000, self.n_query_pos, 2))
        all_Seq_embiding = np.empty((200000, self.max_target * self.windows, self.hidden_dim))
        count = 0
        for one_data in self.train_data:
            split_data, tar_class, tar_position, Seq_embiding, count_after = self.split(one_data, count)
            all_seq_input[count:count_after] = split_data
            all_tar_class[count:count_after] = tar_class
            all_tar_postions[count:count_after] = tar_position
            all_Seq_embiding[count:count_after] = Seq_embiding
            count = count_after

        all_seq_input = all_seq_input[0:count]
        all_tar_class = all_tar_class[0:count]
        all_tar_postions = all_tar_postions[0:count]
        all_Seq_embiding = all_Seq_embiding[0:count]
        print('input and target shape ', all_seq_input.shape, all_tar_class.shape, all_tar_postions.shape)
        np.random.seed(10)
        np.random.shuffle(all_seq_input)
        np.random.seed(10)
        np.random.shuffle(all_tar_class)
        np.random.seed(10)
        np.random.shuffle(all_tar_postions)
        np.random.seed(10)
        np.random.shuffle(all_Seq_embiding)

        train_data_batch, target_class_batch, target_postions_batch,train_data_embiding =[],[],[],[]

        for i in range(len(all_seq_input)//self.batch_size):
            train_data_batch.append(all_seq_input[i*self.batch_size:(i+1)*self.batch_size])
            target_class_batch.append(all_tar_class[i * self.batch_size:(i + 1) * self.batch_size])
            target_postions_batch.append(all_tar_postions[i * self.batch_size:(i + 1) * self.batch_size])
            train_data_embiding.append(all_Seq_embiding[i * self.batch_size:(i + 1) * self.batch_size])
        if len(all_seq_input) % self.batch_size != 0:
            train_data_batch.append(all_seq_input[len(all_seq_input)//self.batch_size * self.batch_size:])
            target_class_batch.append(all_tar_class[len(all_seq_input)//self.batch_size * self.batch_size:])
            target_postions_batch.append(all_tar_postions[len(all_seq_input)//self.batch_size * self.batch_size:])
            train_data_embiding.append(all_Seq_embiding[len(all_seq_input) // self.batch_size * self.batch_size:])
        print('batch number is ',len(train_data_batch), train_data_batch[0].shape)

        all_cost = []
        for j in range(self.training_epochs):
            init_time = time.time()
            sum_loss = 0

            for num in range(len(train_data_batch)):
                _, class_loss, postion_loss, loss= self.sess.run([self.optimizer, self.class_loss, self.pre_loss,self.loss],
                                                 feed_dict={self.input: train_data_batch[num],
                                                            self.input_embiding:train_data_embiding[num],
                                                            self.tag_cla: target_class_batch[num],
                                                            self.position_pre:target_postions_batch[num],
                                                            self.keep_prob_1: 0.8,
                                                            self.learning_rate: self.lr,
                                                            })
                sum_loss += loss

            sum_loss=sum_loss
            print("Total Epoch:", '%d' % (j), "total cost=", "{:.9f}".format(sum_loss),
                  'time: ', "{:.5f}".format(time.time() - init_time), 'lr:', self.lr)
            if j >20:
                all_cost.append(sum_loss)

            if (j + 1) % 10 == 0:

                print('test data of persons in trainingset')
                class_acc,tracking_err = self.test_ac(test_dataset=self.test_data)
            if (j + 1)>150 and (j + 1) % 50 == 0:
                self.saver.save(self.sess, './params_' + str(j + 1) + '/train.ckpt')



def mkdir(path):

    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

if __name__ == '__main__':

    np.set_printoptions(threshold=np.inf)
    train_data = []
    test_data = []

    for envir in range(0, 1, 1):
        path = '../training_data/virtual_training_data/envir' + str(envir)
        temp_paths = os.listdir(path, )
        temp_paths.sort(key=lambda x: int(x[12:-4]))
        for i in range(len(temp_paths)):
            with open(path + temp_paths[i], 'rb') as handle:
                data_temp = pickle.load(handle)
            train_data.append(data_temp)


    for envir in range(0, 1, 1):
            path = '../test_data/envir'+str(envir)
            temp_paths = os.listdir(path, )
            temp_paths.sort(key=lambda x: int(x[12:-4]))
            for i in range(len(temp_paths)):
                with open(path + temp_paths[i], 'rb') as handle:
                    data_temp = pickle.load(handle)
                test_data.append(data_temp)

    print('trainData_len is:', len(train_data))
    print('testData_len is:', len(test_data))

    random.shuffle(train_data)

    DETR(train_data=train_data, test_data=test_data)




