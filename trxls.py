# TRXL tf v2 including GTrXL and TrXL-I
# based on https://github.com/kimiyoung/transformer-xl/tf
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.keras.mixed_precision import experimental as prec

import tools


# Gate for GTrXL
class Gate(tools.Module):

  def __init__(self, d_model, kernel_initializer, gate='plus'):

    self._gate = gate

    if gate == 'plus':
      self._gate = gate

    elif gate == 'input':
      # \sig(W_g^l.*x)*x + y
      self._gate = tfkl.Dense(d_model,
                              use_bias=False,
                              activation=tf.nn.sigmoid,
                              kernel_initializer=kernel_initializer,
                              name='gate')

    elif gate == 'output':
      # x + \sig(W_g^l.*x -b_g^l)*y
      self._gate = tfkl.Dense(d_model,
                              activation=tf.nn.sigmoid,
                              kernel_initializer=kernel_initializer,
                              name='gate')

    elif gate == 'highway':
      #\sig(W_g^l.*x+b_g^l)*x + (1-\sig(W_g^l.*x+b_g^l))*y
      self._gate = tfkl.Dense(d_model,
                              activation=tf.nn.sigmoid,
                              kernel_initializer=kernel_initializer,
                              name='gate')

    elif gate == 'sigmoid-tanh':
      # x + \sig(W_g^l.*x-b_g^l)*\tanh(U_g^l.*y)
      self._gate_W = tfkl.Dense(d_model,
                                activation=tf.nn.sigmoid,
                                kernel_initializer=kernel_initializer,
                                name='gate_W')
      self._gate_U = tfkl.Dense(d_model,
                                use_bias=False,
                                activation=tf.nn.tanh,
                                kernel_initializer=kernel_initializer,
                                name='gate_U')

    elif gate == 'gru':
      self._gate_r = tfkl.Dense(d_model,
                                use_bias=False,
                                activation=tf.nn.sigmoid,
                                kernel_initializer=kernel_initializer,
                                name='gate_r')
      self._gate_z = tfkl.Dense(d_model,
                                activation=tf.nn.sigmoid,
                                kernel_initializer=kernel_initializer,
                                name='gate_z')
      self._gate_h = tfkl.Dense(d_model,
                                use_bias=False,
                                activation=tf.nn.tanh,
                                kernel_initializer=kernel_initializer,
                                name='gate_h')

    else:
      raise ValueError(
        "gate must be one of plus, input, highway, sig-tanh and gru")


  def __call__(self, output, inp):

    if self._gate == 'plus':
      return output + inp

    elif self._gate == 'input':
      # \sig(W_g^l.*x)*x + y
      _inp = self._gate(inp)
      return tf.multiply(_inp, inp) + output

    elif self._gate == 'output':
      # x + \sig(W_g^l.*x -b_g^l)*y
      _inp = self._gate(inp)
      return inp + tf.multiply(_inp, output)

    elif self._gate == 'highway':
      #\sig(W_g^l.*x+b_g^l)*x + (1-\sig(W_g^l.*x+b_g^l))*y
      _inp = self._gate(inp)
      return tf.multiply(_inp, inp) + tf.multiply(1-_inp, output)

    elif self._gate == 'sigmoid-tanh':
      # x + \sig(W_g^l.*x-b_g^l)*\tanh(U_g^l.*y)
      _inp = self._gate_W(inp)
      _output = self._gate_U(output)
      return inp + tf.multiply(_inp, _output)

    elif self._gate == 'gru':
      r = self._gate_r(tf.concat([output, inp], axis=-1))
      z = self._gate_z(tf.concat([output, inp], axis=-1))
      h = self._gate_h(tf.concat([output, tf.multiply(r, inp)], axis=-1))
      return tf.multiply(1-z, inp) + tf.multiply(z, h)


class PositionwiseFF(tools.Module):
  

  def __init__(self,
               d_model, d_inner, dropout, kernel_initializer,
               pre_lnorm=False, gate='plus'):

    self._pre_lnorm = pre_lnorm

    self._layer_norm = tfkl.LayerNormalization(axis=-1)

    self._layer1 = tfkl.Dense(d_inner, activation='relu',
                              kernel_initializer=kernel_initializer,
                              name='layer_1')
    #self._drop1 = tfkl.Dropout(dropout, name='drop_1')
    self._layer2 = tfkl.Dense(d_model,
                              kernel_initializer=kernel_initializer,
                              name='layer_2')
    #self._drop2 = tfkl.Dropout(dropout, name='drop_2')

    self._gate = Gate(d_model, kernel_initializer, gate)

    self._relu = tfkl.ReLU()

  def __call__(self, inp, is_training=True):

    if self._pre_lnorm:
      output = self._layer_norm(inp)
    else:
      output = inp

    output = self._layer1(output)
    #output = self._drop1(output)
    output = self._layer2(output)
    #output = self._drop2(output)

    if not self._pre_lnorm:
      output = self._layer_norm(output + inp)
    else:
      output = self._gate(self._relu(output), inp)

    return output


class RelMultiheadAttn(tools.Module):


  def __init__(self, d_model, n_head, d_head,
               dropout, dropatt,
               kernel_initializer, pre_lnorm=False, gate='plus'):

    self._n_head = n_head
    self._d_head = d_head
    self._pre_lnorm = pre_lnorm
    self._gate = gate

    self._scale = 1 / (d_head ** 0.5)

    self._layer_norm = tfkl.LayerNormalization(axis=-1)

    self._w_heads = tfkl.Dense(3 * n_head * d_head, use_bias=False,
                               kernel_initializer=kernel_initializer,name='qkv')
    self._r_head_k = tfkl.Dense(n_head * d_head, use_bias=False,
                                kernel_initializer=kernel_initializer, name='r')

    #self._dropatt = tfkl.Dropout(dropatt)

    self._attn_out = tfkl.Dense(d_model, use_bias=False,
                               kernel_initializer=kernel_initializer,name='o')

    #self._dropout = tfkl.Dropout(dropout)

    self._gate = Gate(d_model, kernel_initializer, gate)

    self._relu = tfkl.ReLU()


  def _rel_shift(self, x):
    x_size = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)

    return x


  def __call__(self, inp,
               r, r_w_bias, r_r_bias,
               attn_mask, mems, is_training=True):

    if self._pre_lnorm:
      w = self._layer_norm(inp)
    else:
      w = inp

    qlen = w.get_shape().as_list()[0]
    rlen = r.shape[0]
    bsz = w.get_shape().as_list()[1]

    cat = tf.concat([mems, w],
                    0) if mems is not None and mems.shape.ndims > 1 else w
    w_heads = self._w_heads(cat)
    r_head_k = self._r_head_k(r)

    w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
    w_head_q = w_head_q[-qlen:]

    klen = w_head_k.shape[0]

    w_head_q = tf.reshape(w_head_q, [qlen, -1, self._n_head, self._d_head])
    w_head_k = tf.reshape(w_head_k, [klen, -1, self._n_head, self._d_head])
    w_head_v = tf.reshape(w_head_v, [klen, -1, self._n_head, self._d_head])

    r_head_k = tf.reshape(r_head_k, [rlen, self._n_head, self._d_head])

    dtype = prec.global_policy().compute_dtype
    rw_head_q = w_head_q + tf.cast(r_w_bias, dtype=dtype)
    rr_head_q = w_head_q + tf.cast(r_r_bias, dtype=dtype)

    AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
    BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
    BD = self._rel_shift(BD)

    attn_score = (AC + BD) * self._scale
    attn_mask_t = attn_mask[:, :, None, None]
    attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

    attn_prob = tf.nn.softmax(attn_score, 1)
    #attn_prob = self._dropatt(attn_prob, training=is_training)

    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
    size_t = attn_vec.shape
    attn_vec = tf.reshape(attn_vec, [size_t[0], -1, self._n_head * self._d_head])

    attn_out = self._attn_out(attn_vec)
    #attn_out = self._dropout(attn_out, training=is_training)

    if not self._pre_lnorm:
      output = self._layer_norm(attn_out + inp)
    else:
      output = self._gate(self._relu(attn_out), inp)

    return output


class TrXL(tools.Module):

  def __init__(self,
         pre_lnorm=False,
         gate='plus',
         n_layer=6,
         d_model=256,
         n_head=8,
         d_head=64,
         d_inner=256,
         mem_len=512,
         dropout=0.0,
         dropatt=0.0,
         seed=1,
         init='normal', # {normal|uniform}
         init_range=0.1,
         init_std=0.02,
         same_length=False,
         clamp_len=-1,
         untie_r=False):

    self._n_layer = n_layer
    self._mem_len = mem_len
    self._untie_r = untie_r

    if init == 'uniform':
      initializer = tf.keras.initializers.RandomUniform(
            minval=-init_range,
            maxval=init_range)
            #seed=seed) # without seed
    elif init == "normal":
      initializer = tf.keras.initializers.RandomNormal(
            stddev=init_std)
            #seed=seed)  # without seed

    new_mems = []
    dtype = prec.global_policy().variable_dtype
    if untie_r:
      self._r_w_bias = tf.Variable(name='r_w_bias',
              initial_value=initializer(shape=(n_layer, n_head, d_head), dtype=dtype),
              dtype=dtype)
      self._r_r_bias = tf.Variable(name='r_r_bias',
              initial_value=initializer(shape=(n_layer, n_head, d_head), dtype=dtype),
              dtype=dtype)
    else:
      self._r_w_bias = tf.Variable(name='r_w_bias',
              initial_value=initializer(shape=(n_head, d_head), dtype=dtype),
              dtype=dtype)
      self._r_r_bias = tf.Variable(name='r_r_bias',
              initial_value=initializer(shape=(n_head, d_head), dtype=dtype),
              dtype=dtype)

    qlen = 1
    mlen = mem_len
    klen = qlen + mlen
    self._attn_mask = self._create_mask(qlen, mlen, same_length)

    pos_seq = tf.range(klen - 1, -1, -1.0)
    if clamp_len > 0:
      pos_seq = tf.minimum(pos_seq, clamp_len)
    inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
    self._pos_emb = self._positional_embedding(pos_seq, inv_freq)

    #self._dropout = tfkl.Dropout(dropout)

    self._attn_layers = []
    self._ff_layers = []
    for i in range(n_layer):
      self._attn_layers.append(RelMultiheadAttn(
            d_model=d_model,
            n_head=n_head,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropatt,
            kernel_initializer=initializer,
            pre_lnorm=pre_lnorm,
            gate=gate))
      self._ff_layers.append(PositionwiseFF(
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            kernel_initializer=initializer,
            pre_lnorm=pre_lnorm,
            gate=gate))

  def _create_mask(self, qlen, mlen, same_length=False):

    dtype = prec.global_policy().compute_dtype
    attn_mask = tf.ones([qlen, qlen], dtype=dtype)
    mask_u = tf.compat.v1.matrix_band_part(attn_mask, 0, -1)
    mask_dia = tf.compat.v1.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
      mask_l = tf.compat.v1.matrix_band_part(attn_mask, -1, 0)
      ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


  def _positional_embedding(self, pos_seq, inv_freq, bsz=None):

    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
      return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
      return pos_emb[:, None, :]


  def _cache_mem(self, curr_out, prev_mem, mem_len=None):

    if mem_len is None or prev_mem is None:
      new_mem = curr_out
    elif mem_len == 0:
      return prev_mem
    else:
      new_mem = tf.concat([prev_mem, curr_out], 0)[- mem_len:]

    return tf.stop_gradient(new_mem)


  def __call__(self, dec_inp, mems, is_training=True):

    new_mems = []

    output = dec_inp
    pos_emb = self._pos_emb
    #output = self._dropout(dec_inp, training=is_training)
    #pos_emb = self._dropout(self._pos_emb, training=is_training)

    for i in range(self._n_layer):
      # cache new mems
      new_mems.append(self._cache_mem(output, mems[i], self._mem_len))

      output = self._attn_layers[i](inp=output,
          r=pos_emb,
          r_w_bias=self._r_w_bias if not self._untie_r else self._r_w_bias[i],
          r_r_bias=self._r_r_bias if not self._untie_r else self._r_r_bias[i],
          attn_mask=self._attn_mask,
          mems=mems[i],
          is_training=is_training)

      output = self._ff_layers[i](inp=output,
                      is_training=is_training)


    #output = self._dropout(output, training=is_training)

    return tf.reshape(output, [-1, output.shape[-1]]), \
           tf.stack(new_mems, axis=0)

