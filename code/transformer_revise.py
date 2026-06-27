'''
Adapted from https://www.tensorflow.org/tutorials/text/transformer
'''
import tensorflow as tf
import numpy as np

# Positional Encoding

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

# Masking

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  print(enc_padding_mask.shape)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask

# Scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask):


  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  # print('matmul_qk',matmul_qk.shape)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  # print('scaled_attention_logits',scaled_attention_logits.shape)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  # print('attention_weights',attention_weights.shape)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  # print('output',output.shape)
  return output, attention_weights


class Transformer():
  def __init__(self, pe_input=256, pe_target=100, num_layers=6, d_model=512, num_heads=8, dff=2048, keep_prob=0.8,
               input_vocab_size=256, target_vocab_size=100, num_classes=81,inp=None, tar=None, training=True,
               enc_padding_mask=None, dec_padding_mask=None,look_ahead_mask=None):
    self.pe_input = pe_input
    self.pe_target = pe_target
    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.keep_prob = keep_prob
    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size
    self.num_classes=num_classes+1
    self.inp = inp
    self.tar = tar
    self.training = training
    self.enc_padding_mask = enc_padding_mask
    self.look_ahead_mask = look_ahead_mask
    self.dec_padding_mask = dec_padding_mask

  def build(self):
    self.enc_output = self.Encoder(self.pe_input,self.inp, self.training, self.enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    self.dec_output, self.attention_weights = self.Decoder(self.pe_target,self.tar, self.enc_output, self.training,
                                                           self.look_ahead_mask, self.dec_padding_mask)





    return self.dec_output, self.attention_weights

  def Encoder(self, maximum_position_encoding, x, training, mask):
    seq_len = tf.shape(x)[1]

    for i in range(self.num_layers):
      x = self.EncoderLayer(x, mask,name='encoder'+str(i))
    return x

  def Decoder(self, maximum_position_encoding, x, enc_output, training, look_ahead_mask, padding_mask):
    seq_len = tf.shape(x)[1]
    attention_weights = {}



    for i in range(self.num_layers):
      x, block1, block2 = self.DecoderLayer(x, enc_output, training, look_ahead_mask, padding_mask,name='decoder'+str(i))

      attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

  def EncoderLayer(self, x, mask,name=None):
    self.mha = MultiHeadAttention(self.d_model, self.num_heads, x, x, x, mask)
    attn_output, _ = self.mha.call(name)  # (batch_size, input_seq_len, d_model)
    attn_output = tf.nn.dropout(attn_output, keep_prob=self.keep_prob, name=name+'dropout1')
    out1 = tf.contrib.layers.layer_norm(x + attn_output)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.point_wise_feed_forward_network(out1, name=name)  # (batch_size, input_seq_len, d_model)
    ffn_output = tf.nn.dropout(ffn_output, keep_prob=self.keep_prob, name=name+'dropout2')
    out2 = tf.contrib.layers.layer_norm(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    return out2

  def DecoderLayer(self, x, enc_output, training,look_ahead_mask, padding_mask,name=None):
    self.mha1 = MultiHeadAttention(self.d_model, self.num_heads, x, x, x, look_ahead_mask)
    attn1, attn_weights_block1 = self.mha1.call(name+'mha1')  # (batch_size, target_seq_len, d_model)
    attn1 = tf.nn.dropout(attn1, keep_prob=self.keep_prob, name=name+'dropout1')
    out1 = tf.contrib.layers.layer_norm(attn1 + x)
    self.mha2 = MultiHeadAttention(self.d_model, self.num_heads, enc_output, enc_output, out1, padding_mask)
    attn2, attn_weights_block2 = self.mha2.call(name+'mha2')  # (batch_size, target_seq_len, d_model)
    attn2 = tf.nn.dropout(attn2, keep_prob=self.keep_prob, name=name+'dropout2')
    out2 = tf.contrib.layers.layer_norm(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.point_wise_feed_forward_network(out2,name=name)  # (batch_size, target_seq_len, d_model)
    ffn_output = tf.nn.dropout(ffn_output, keep_prob=self.keep_prob, name='dropout2')
    out3 = tf.contrib.layers.layer_norm(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2

  def point_wise_feed_forward_network(self,out1, name=None):
    with tf.variable_scope(name):
      layer1 = tf.layers.dense(out1, self.dff, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer,
                               name='fc1')
      ffn_output = tf.layers.dense(layer1, self.d_model, kernel_initializer=tf.glorot_uniform_initializer, name='fc2')
      return ffn_output

  def linear_class(self, out, name=None):
    with tf.variable_scope(name):
      layer1 = tf.layers.dense(out, 128, activation=tf.nn.relu,
                               kernel_initializer=tf.glorot_uniform_initializer,
                               name='fc1')
      ffn_output = tf.layers.dense(layer1, self.num_classes, kernel_initializer=tf.glorot_uniform_initializer,
                                   name='fc2')
      return ffn_output

  def linear_bbox(self, out, name=None):
    with tf.variable_scope(name):
      layer1 = tf.layers.dense(out, self.d_model, activation=tf.nn.relu,
                               kernel_initializer=tf.glorot_uniform_initializer,
                               name='fc1')
      ffn_output = tf.layers.dense(layer1, 2, kernel_initializer=tf.glorot_uniform_initializer,
                                   name='fc2')
      return ffn_output

#Multi-head attention

class MultiHeadAttention():
  def __init__(self, d_model, num_heads,v, k, q, mask):
    self.num_heads = num_heads
    self.d_model = d_model
    self.v=v
    self.k=k
    self.q=q
    self.mask=mask


    assert d_model % self.num_heads == 0

    self.depth =  self.d_model  // self.num_heads

  def call(self,name=None):
    batch_size = tf.shape(self.q)[0]
    with tf.variable_scope(name):
      self.wq = tf.layers.dense(self.q, self.d_model, kernel_initializer=tf.glorot_uniform_initializer,
                                name='wq')  # (batch_size, seq_len, d_model)
      self.wk = tf.layers.dense(self.k, self.d_model, kernel_initializer=tf.glorot_uniform_initializer,
                                name='wk')  # (batch_size, seq_len, d_model)
      self.wv = tf.layers.dense(self.v, self.d_model, kernel_initializer=tf.glorot_uniform_initializer,
                                name='wv')  # (batch_size, seq_len, d_model)
      self.wq = self.split_heads(self.wq, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
      self.wk = self.split_heads(self.wk, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
      self.wv = self.split_heads(self.wv, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

      # print('wq,wk,wv',self.wq.shape, self.wk.shape,self.wv.shape)
      # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
      # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
      scaled_attention, attention_weights = scaled_dot_product_attention(
        self.wq, self.wk, self.wv, self.mask)
      # print('scaled_attention', scaled_attention.shape)

      scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

      concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

      output = tf.layers.dense(concat_attention, self.d_model, kernel_initializer=tf.glorot_uniform_initializer,
                               name='dense')  # (batch_size, seq_len_q, d_model)

      return output, attention_weights

  def split_heads(self, x, batch_size):

    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])


