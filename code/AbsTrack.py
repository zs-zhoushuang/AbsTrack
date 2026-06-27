import numpy as np
import tensorflow as tf
from transformer_revise import Transformer
import random
import os
import pickle
import time
import pylab

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



        return all_pos_encoding #tf.cast(pos_encoding, dtype=tf.float32)

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


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        # print path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print path + ' 目录已存在'
        return False

if __name__ == '__main__':

    np.set_printoptions(threshold=np.inf)
    train_data = []
    test_data = []
    test_NoTrainset=[]

    print('trainData_len is:', len(train_data))
    print('testData_len is:', len(test_data))
    print('test_NoTrainset_len is', len(test_NoTrainset))
    random.shuffle(train_data)

    DETR(train_data=train_data, test_data=test_data,test_NoTrainset=test_NoTrainset)




