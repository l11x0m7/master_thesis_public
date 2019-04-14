import theano.tensor as T
import lasagne
import lasagne.layers as L
from lasagne import nonlinearities
from lasagne import init

epsilon = 1e-8

def softmax(x, axis=-1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    sm = e_x / (e_x.sum(axis=axis, keepdims=True) + epsilon)
    return sm


# my addition
class LambdaLayer(lasagne.layers.MergeLayer):
    def __init__(self, func, incomings, **kwargs):
        super(LambdaLayer, self).__init__(incomings, **kwargs)
        self.func = func
    def get_output_for(self, inputs, **kwargs):
        return self.func(*inputs)

    def get_output_shape_for(self, input_shape):
        return input_shape[0]


class FreeSqueezeLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(FreeSqueezeLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        A, B, C, D = input.shape
        return T.reshape(input, (A, B, C))

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]


class SelfAttention(L.MergeLayer):
    # incomings[0]: B * P * 2D
    # B * P
    def __init__(self, incomings, num_units,
                 nonlinearity=lasagne.nonlinearities.tanh,
                 mask_input=None,
                 name='',
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(SelfAttention, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        self.num_units = num_units
        self.W0 = self.add_param(init, (self.num_units, self.num_units), name='W0_sa_{}'.format(name))
        self.Wb = self.add_param(init, (self.num_units, ), name='Wb_sa_{}'.format(name))
    # inputs[0]: B * P * 2D
    # inputs[1]: B * P
    def get_output_for(self, inputs, **kwargs):
        B, P, D = inputs[0].shape
        # B * P
        alphas = T.dot(self.nonlinearity(T.dot(inputs[0], self.W0)), self.Wb)
        alphas = T.nnet.softmax(alphas) * inputs[1]
        alphas = alphas / (alphas.sum(axis=1, keepdims=True) + epsilon) * inputs[1]

        att = T.sum(inputs[0] * alphas.dimshuffle(0, 1, 'x'), axis=1)

        return att

    def get_output_shape_for(self, input_shapes):
        # outputs: B * 2D
        return (input_shapes[0][0], input_shapes[0][2])


class SqueezeLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return T.cast(input.sum(axis=-1) > 0, 'int32')

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]


class CombineLayer(L.MergeLayer):
    # inputs[0]: B * N
    # inputs[1]: B * N

    def get_output_for(self, inputs, **kwargs):
        return (inputs[0] + inputs[1]) / 2.

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]


class CharEncoding(L.Layer):
    # incomings[0]: B * P * 2D
    # B * P
    def __init__(self, incoming, 
                 num_filters, filter_size, 
                 stride=1, pad=0, 
                 W=None, b=None,
                 name='',
                 flip_filters=True,
                 W_init=init.GlorotUniform(), b_init=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(CharEncoding, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
        self.num_filters = num_filters
        self.filter_size = (filter_size, )
        self.flip_filters = (flip_filters, )
        self.stride = (stride, )
        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = (0, )
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = (pad, )
        if W is not None:
            self.W = W
        else:
            self.W = self.add_param(W_init, self.get_W_shape(), name="W_conv_{}".format(name))
        if b is not None:
            self.b = b
        else:
            if b_init is None:
                self.b = None
            else:
                biases_shape = (num_filters,)
                self.b = self.add_param(b_init, biases_shape, name="b_conv_{}".format(name),
                                        regularizable=False)
        self.convolution = lasagne.theano_extensions.conv.conv1d_mc0

    def get_W_shape(self):
        num_input_channels = self.input_shape[3]
        return (self.num_filters, num_input_channels) + self.filter_size

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        extra_kwargs = {}
        conved = self.convolution(input, self.W,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters,
                                  **extra_kwargs)
        return conved

    # inputs[0]: B * W * max_word_len * D
    def get_output_for(self, inputs, **kwargs):
        B, W, C, D = inputs.shape
        new_batch = T.reshape(inputs, (B * W, C, D))
        new_batch = new_batch.dimshuffle(0, 2, 1)
        conved = self.convolve(new_batch)
        if self.b is None:
            activation = conved
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x')
        activated_out = self.nonlinearity(activation)
        poolinged = T.max(activated_out, axis=-1)
        out = T.reshape(poolinged, (B, W, self.num_filters))
        return out

    def get_output_shape_for(self, input_shapes):
        # outputs: B * 2D
        return (input_shapes[0], input_shapes[1], self.num_filters)

class HighwayLayer(L.MergeLayer):
    # inputs[0]: B * N * D
    # inputs[1]: B * N * D
    # inputs[2]: B * N * D

    def get_output_for(self, inputs, **kwargs):
        gate = T.nnet.sigmoid(inputs[1])
        return inputs[0] * gate + (1 - gate) * inputs[2]

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]


def ConvLayer(l_emb, l_mask, dim, args, depth=[]):
    l_emb_exp = L.DimshuffleLayer(l_emb, (0, 2, 1))
    if not depth:
        o_in = L.Conv1DLayer(l_emb_exp, dim, filter_size=1)
        o_out = L.DimshuffleLayer(o_in, (0, 2, 1))
    else:
        o_in = l_emb_exp
        o_1 = L.Conv1DLayer(o_in, dim, filter_size=1)
        all_os = [o_1]
        o_1 = lasagne.layers.DropoutLayer(o_1, p=args.dropout_rate)
        for i in depth:
            # o_cur = L.DilatedConv2DLayer(o_pad, dim, filter_size=(3, 1), dilation=(expand, 1))
            # gate_o = L.DilatedConv2DLayer(o_pad, dim, filter_size=(3, 1), dilation=(expand, 1))
            o_cur = L.Conv1DLayer(o_1, dim, filter_size=i, pad='same', stride=1, nonlinearity=nonlinearities.linear)
            # gate_o = L.Conv1DLayer(o_in, dim, filter_size=i, pad='same', stride=1, nonlinearity=nonlinearities.sigmoid)
            # o = HighwayLayer([o_in, gate_o, o_cur])
            all_os.append(o_cur)
        o = L.ConcatLayer(all_os, axis=1)
        o = L.DropoutLayer(o, p=args.dropout_rate)
        o = L.Conv1DLayer(o, dim, filter_size=1, pad='same', stride=1, nonlinearity=nonlinearities.rectify)
        o = L.DropoutLayer(o, p=args.dropout_rate)
        g = L.Conv1DLayer(o, dim, filter_size=1, pad='same', stride=1, nonlinearity=nonlinearities.linear)
        z = L.Conv1DLayer(o_1, dim, filter_size=1, nonlinearity=nonlinearities.tanh)
        o_out = HighwayLayer([o_1, g, z])
        o_out = L.DimshuffleLayer(o_out, (0, 2, 1))

    l_mask_out = L.DimshuffleLayer(l_mask, (0, 1, 'x'))

    res = LambdaLayer(lambda a, b: a * b, [o_out, l_mask_out])
    return res

def MRU(l_emb, l_mask, dim, args, depth=[]):
    l_emb_exp = L.DimshuffleLayer(l_emb, (0, 2, 1))
    if not depth:
        o_in = L.Conv1DLayer(l_emb_exp, dim, filter_size=1)
        o_out = L.DimshuffleLayer(o_in, (0, 2, 1))
    else:
        o_in = L.Conv1DLayer(l_emb_exp, dim, filter_size=1)
        o_out = L.DimshuffleLayer(o_in, (0, 2, 1))
        all_os = [o_out]
        o_in = L.DimshuffleLayer(l_emb, (0, 'x', 1, 2))
        for i in depth:
            o_pad = RightPadLayer(o_in, i)
            o_cur = L.Conv2DLayer(o_pad, 1, filter_size=(i, 1), stride=(i, 1), 
                                  nonlinearity=nonlinearities.linear, 
                                  W=lasagne.init.Constant(1.), b=lasagne.init.Constant(0.))
            o_cur.params[o_cur.W].remove('trainable')
            o_cur.params[o_cur.b].remove('trainable')
            o_cur = ExpandLayer(o_cur)
            all_os.append(o_cur)






class BiDirectionAttentionLayer(L.MergeLayer):
    def __init__(self, incomings, num_units,
                 nonlinearity=lasagne.nonlinearities.tanh,
                 mask_input=None,
                 name='',
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 4:
            raise NotImplementedError
        super(BiDirectionAttentionLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        self.num_units = num_units * 3
        self.W_att = self.add_param(init, (self.num_units, ), name='W_att_bi_{}'.format(name))
    # inputs[0]: B * N * 2D
    # inputs[1]: B * Q * 2D
    # inputs[2]: B * N, mask for inputs[0]
    # inputs[3]: B * Q, mask for inputs[1]
    # outputs: B * N * 8D
    def get_output_for(self, inputs, **kwargs):
        B, N, H = inputs[0].shape
        B, Q, H = inputs[1].shape
        doc_expand_emb = T.tile(inputs[0].dimshuffle(0, 1, 'x', 2), (1, 1, Q, 1))
        qry_expand_emb = T.tile(inputs[1].dimshuffle(0, 'x', 1, 2), (1, N, 1, 1))
        doc_qry_dot_emb = doc_expand_emb * qry_expand_emb
        # B * N * Q
        doc_qry_mask = T.batched_dot(inputs[2].dimshuffle(0, 1, 'x'), inputs[3].dimshuffle(0, 'x', 1))
        # B * N * Q * 6D
        doc_qry_emb = T.concatenate([doc_expand_emb, qry_expand_emb, doc_qry_dot_emb], axis=3)
        # B * N * Q
        S = T.dot(doc_qry_emb, self.W_att)
        c2q_att = T.nnet.softmax(S.reshape((B * N, Q))).reshape((B, N, Q)) * doc_qry_mask
        c2q_att = c2q_att / (c2q_att.sum(axis=2, keepdims=True) + epsilon) * doc_qry_mask
        # B * N * 2D
        c2q = T.batched_dot(c2q_att, inputs[1])
        # B * N
        q2c_att = T.nnet.softmax(S.max(axis=2, keepdims=False)) * inputs[2]
        q2c_att = q2c_att / (q2c_att.sum(axis=1,keepdims=True) + epsilon) * inputs[2]
        # B * N * 2D
        q2c = T.tile(T.batched_dot(q2c_att.dimshuffle(0, 'x', 1), inputs[0]), (1, N, 1))
        G = T.concatenate([inputs[0], c2q, inputs[0] * c2q, inputs[0] * q2c], axis=-1)
        # B * N * 8D
        return G

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2] * 4)


class HybridInteractionLayer(L.MergeLayer):
    def __init__(self, incomings, num_units,
                 nonlinearity=lasagne.nonlinearities.tanh,
                 mask_input=None,
                 name='',
                 init=lasagne.init.Uniform(), use_relu=False, **kwargs):
        if len(incomings) != 4:
            raise NotImplementedError
        super(HybridInteractionLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        self.num_units = num_units * 3
        self.W_att = self.add_param(init, (self.num_units, ), name='W_att_bi_{}'.format(name))
        self.use_relu = use_relu
        if self.use_relu:
            self.W_dense = self.add_param(lasagne.init.GlorotUniform(), (2 * num_units, num_units), name='dense_W_{}'.format(name))
            self.b_dense = self.add_param(lasagne.init.Constant(0.), (num_units,), name="dense_b_{}".format(name),
                                    regularizable=False)
            self.dense_nonlinearity = T.nnet.relu
    # inputs[0]: B * N * 2D
    # inputs[1]: B * Q * 2D
    # inputs[2]: B * N mask for inputs[0]
    # inputs[3]: B * Q, mask for inputs[1]
    # outputs: B * N * ?
    def get_output_for(self, inputs, **kwargs):
        B, N, H = inputs[0].shape
        B, Q, H = inputs[1].shape
        doc_expand_emb = T.tile(inputs[0].dimshuffle(0, 1, 'x', 2), (1, 1, Q, 1))
        qry_expand_emb = T.tile(inputs[1].dimshuffle(0, 'x', 1, 2), (1, N, 1, 1))
        doc_qry_dot_emb = doc_expand_emb * qry_expand_emb
        # B * N * Q
        doc_qry_mask = T.batched_dot(inputs[2].dimshuffle(0, 1, 'x'), inputs[3].dimshuffle(0, 'x', 1))
        # B * N * Q * 6D
        doc_qry_emb = T.concatenate([doc_expand_emb, qry_expand_emb, doc_qry_dot_emb], axis=3)
        # B * N * Q
        S = T.dot(doc_qry_emb, self.W_att)
        c2q_att = softmax(S, axis=2) * doc_qry_mask
        c2q_att = c2q_att / (c2q_att.sum(axis=2, keepdims=True) + epsilon) * doc_qry_mask
        # B * N * 2D
        c2q = T.batched_dot(c2q_att, inputs[1]) * inputs[2].dimshuffle(0, 1, 'x')
        c2q_gated = c2q * inputs[0]

        # AoA
        # B * N * Q
        aoa_doc_qry_att = T.batched_dot(inputs[0], inputs[1].dimshuffle(0, 2, 1))
        # # B * N * Q
        aoa_alpha = softmax(aoa_doc_qry_att, axis=1)
        # aoa_alpha = aoa_alpha * doc_qry_mask
        # aoa_alpha = aoa_alpha / (aoa_alpha.sum(axis=1, keepdims=True) + 1e-8) * doc_qry_mask
        # # B * N * Q
        # aoa_beta = softmax(aoa_doc_qry_att, axis=2)
        # aoa_beta = aoa_beta * doc_qry_mask
        # aoa_beta = aoa_beta / (aoa_beta.sum(axis=2, keepdims=True) + 1e-8) * doc_qry_mask
        # d_length = inputs[2].sum(axis=-1)
        # d_length = T.tile(d_length.dimshuffle(0, 'x'), (1, Q))
        # # B * Q
        # aoa_beta_sum = aoa_beta.sum(axis=1) / d_length
        # # B * N
        # doc_att = T.batched_dot(aoa_alpha, aoa_beta_sum) * inputs[2]
        # # B * N * 2D
        # doc_att = T.tile(doc_att.dimshuffle(0, 1, 'x'), (1, 1, H))

        # Gated Attention
        alphas = aoa_alpha
        alphas_r = alphas * inputs[3].dimshuffle(0, 'x', 1) # B x N x Q
        alphas_r = alphas_r / alphas_r.sum(axis=2, keepdims=True)               # B x N x Q
        q_rep = T.batched_dot(alphas_r, inputs[1])                              # B x N x 2D
        d_gated = inputs[0] * q_rep

        G = T.concatenate([c2q_gated, d_gated], axis=-1)
        if self.use_relu:
            activation = T.dot(G, self.W_dense)
            activation = activation + self.b_dense
            activation = self.dense_nonlinearity(activation)
            G = activation
        # B * N * 2D or B * N * 4D
        return G

    def get_output_shape_for(self, input_shapes):
        if self.use_relu:
            return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2])
        else:
            return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2] * 2)


class HybridInteractionWithDense(L.MergeLayer):
    def __init__(self, incomings, num_units,
                 nonlinearity=lasagne.nonlinearities.tanh,
                 mask_input=None,
                 name='',
                 init=lasagne.init.Uniform(), W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), dense_nonlinearity=T.nnet.relu, **kwargs):
        if len(incomings) != 4:
            raise NotImplementedError
        super(HybridInteractionWithDense, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        self.dense_nonlinearity = dense_nonlinearity
        self.num_units = num_units * 3
        # self.W_att = self.add_param(init, (self.num_units, ), name='W_att_bi_{}'.format(name))
        num_inputs = num_units
        self.W_dense = self.add_param(W, (num_inputs, num_units), name='dense_W_{}'.format(name))
        if b is None:
            self.b_dense = None
        else:
            self.b_dense = self.add_param(b, (num_units,), name="dense_b_{}".format(name),
                                    regularizable=False)
    # inputs[0]: (B*4, O, 2D)
    # inputs[1]: (B, N, 2D)
    # inputs[2]: (B*4, O), mask for inputs[0]
    # inputs[3]: (B, N), mask for inputs[1]
    # outputs: (B*4, 2D)
    def get_output_for(self, inputs, **kwargs):
        B_4, O, H = inputs[0].shape
        B, N, H = inputs[1].shape
        O_4 = O * 4
        reshape_o = T.reshape(inputs[0], (B, O_4, H))
        reshape_o_mask = T.reshape(inputs[2], (B, O_4))
        # doc_expand_emb = T.tile(reshape_o.dimshuffle(0, 1, 'x', 2), (1, 1, N, 1))
        # qry_expand_emb = T.tile(inputs[1].dimshuffle(0, 'x', 1, 2), (1, O_4, 1, 1))
        # doc_qry_dot_emb = doc_expand_emb * qry_expand_emb
        # # B * O_4 * N
        # doc_qry_mask = T.batched_dot(reshape_o_mask.dimshuffle(0, 1, 'x'), inputs[3].dimshuffle(0, 'x', 1))
        # # B * O_4 * N * 6D
        # doc_qry_emb = T.concatenate([doc_expand_emb, qry_expand_emb, doc_qry_dot_emb], axis=3)
        # # B * O_4 * N
        # S = T.dot(doc_qry_emb, self.W_att)
        # c2q_att = softmax(S, axis=2) * doc_qry_mask
        # c2q_att = c2q_att / (c2q_att.sum(axis=2, keepdims=True) + epsilon) * doc_qry_mask
        # # B * N * 2D
        # c2q = T.batched_dot(c2q_att, inputs[1]) * reshape_o_mask.dimshuffle(0, 1, 'x')
        # c2q_gated = c2q * reshape_o

        # AoA
        # B * O_4 * N
        aoa_doc_qry_att = T.batched_dot(reshape_o, inputs[1].dimshuffle(0, 2, 1))
        # B * O_4 * N
        aoa_alpha = softmax(aoa_doc_qry_att, axis=1)
        # aoa_alpha = aoa_alpha * doc_qry_mask
        # aoa_alpha = aoa_alpha / (aoa_alpha.sum(axis=1, keepdims=True) + 1e-8) * doc_qry_mask
        # # B * N * Q
        # aoa_beta = softmax(aoa_doc_qry_att, axis=2)
        # aoa_beta = aoa_beta * doc_qry_mask
        # aoa_beta = aoa_beta / (aoa_beta.sum(axis=2, keepdims=True) + 1e-8) * doc_qry_mask
        # d_length = inputs[2].sum(axis=-1)
        # d_length = T.tile(d_length.dimshuffle(0, 'x'), (1, Q))
        # # B * Q
        # aoa_beta_sum = aoa_beta.sum(axis=1) / d_length
        # # B * N
        # doc_att = T.batched_dot(aoa_alpha, aoa_beta_sum) * inputs[2]
        # # B * N * 2D
        # doc_att = T.tile(doc_att.dimshuffle(0, 1, 'x'), (1, 1, H))

        # Gated Attention
        alphas = aoa_alpha
        alphas_r = alphas * inputs[3].dimshuffle(0, 'x', 1) # B x O_4 x N
        alphas_r = alphas_r / alphas_r.sum(axis=2, keepdims=True)               # B x O_4 x N
        q_rep = T.batched_dot(alphas_r, inputs[1])                              # B x O_4 x 2D
        d_gated = reshape_o * q_rep

        # B x O_4 x 4D
        # G = T.concatenate([c2q_gated, d_gated], axis=-1)
        G = d_gated
        G = T.reshape(G, (B*O_4, G.shape[2]))
        activation = T.dot(G, self.W_dense)
        if self.b_dense is not None:
            activation = activation + self.b_dense
        # B x O_4 x 2D
        activation = self.dense_nonlinearity(activation)
        # B_4 x O x 2D
        activation = T.reshape(activation, (B_4, O, H))
        return activation

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2])




# -----------------------------origin------------------------------

class GatedAttentionLayerWithQueryAttention(L.MergeLayer):
    # inputs[0]: B * N * 2D
    # inputs[1]: B * Q * 2D
    # inputs[2]: B * Q          (l_m_q)

    def get_output_for(self, inputs, **kwargs):
        M = T.batched_dot(inputs[0], inputs[1].dimshuffle((0,2,1)))             # B x N x Q
        B, N, Q = M.shape
        alphas = T.nnet.softmax(T.reshape(M, (B*N, Q)))
        alphas_r = T.reshape(alphas, (B,N,Q)) * inputs[2].dimshuffle(0, 'x', 1) # B x N x Q
        alphas_r = alphas_r / alphas_r.sum(axis=2, keepdims=True)               # B x N x Q
        q_rep = T.batched_dot(alphas_r, inputs[1])                              # B x N x 2D
        d_gated = inputs[0] * q_rep
        return d_gated

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

class QuerySliceLayer(L.MergeLayer):
    # inputs[0]: B * Q * 2D (q)
    # inputs[1]: B          (q_var)

    def get_output_for(self, inputs, **kwargs):
        q_slice = inputs[0][T.arange(inputs[0].shape[0]), inputs[1]-1, :]     # B x 2D
        return q_slice

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][2])

class GatedAttentionLayer(L.MergeLayer):
    # inputs[0]: B * N * 2D
    # inputs[1]: N * 2D

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1].dimshuffle(0, 'x', 1)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]


class AttentionSumLayer(L.MergeLayer):
    # inputs[0]: batch * len * h            (d)
    # inputs[1]: batch * h                  (q_slice)
    # inputs[2]: batch * len * num_cand     (c_var)
    # inputs[3]: batch * len                (m_c_var)

    def get_output_for(self, inputs, **kwargs):
        dq = T.batched_dot(inputs[0], inputs[1])    # B x len
        attention = T.nnet.softmax(dq) * inputs[3]  # B x len
        attention = attention / attention.sum(axis=1, keepdims=True)
        probs = T.batched_dot(attention, inputs[2]) # B x num_cand
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[2][0], input_shapes[2][2])

def stack_rnn(l_emb, l_mask, num_layers, num_units,
              grad_clipping=10, dropout_rate=0.,
              bidir=True,
              only_return_final=False,
              name='',
              rnn_layer=lasagne.layers.LSTMLayer):
    """
        Stack multiple RNN layers.
    """

    def _rnn(backwards=True, name=''):
        network = l_emb
        for layer in range(num_layers):
            if dropout_rate > 0:
                network = lasagne.layers.DropoutLayer(network, p=dropout_rate)
            c_only_return_final = only_return_final and (layer == num_layers - 1)
            network = rnn_layer(network, num_units,
                                grad_clipping=grad_clipping,
                                mask_input=l_mask,
                                only_return_final=c_only_return_final,
                                backwards=backwards,
                                name=name + '_layer' + str(layer + 1))
        return network

    network = _rnn(True, name)
    if bidir:
        network = lasagne.layers.ConcatLayer([network, _rnn(False, name + '_back')], axis=-1)
    return network


class AveragePoolingLayer(lasagne.layers.MergeLayer):
    """
        Average pooling.
        incoming: batch x len x h
    """
    def __init__(self, incoming, mask_input=None, **kwargs):
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)
        super(AveragePoolingLayer, self).__init__(incomings, **kwargs)
        if len(self.input_shapes[0]) != 3:
            raise ValueError('the shape of incoming must be a 3-element tuple')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:-2] + input_shapes[0][-1:]

    def get_output_for(self, inputs, **kwargs):
        if len(inputs) == 1:
            # mask_input is None
            return T.mean(inputs[0], axis=1)
        else:
            # inputs[0]: batch x len x h
            # inputs[1] = mask_input: batch x len
            return (T.sum(inputs[0] * inputs[1].dimshuffle(0, 1, 'x'), axis=1) /
                    T.sum(inputs[1], axis=1).dimshuffle(0, 'x'))


class MLPAttentionLayer(lasagne.layers.MergeLayer):
    """
        An MLP attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
        Reference: http://arxiv.org/abs/1506.03340
    """
    def __init__(self, incomings, num_units,
                 nonlinearity=lasagne.nonlinearities.tanh,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(MLPAttentionLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        self.num_units = num_units
        self.W0 = self.add_param(init, (self.num_units, self.num_units), name='W0_mlp')
        self.W1 = self.add_param(init, (self.num_units, self.num_units), name='W1_mlp')
        self.Wb = self.add_param(init, (self.num_units, ), name='Wb_mlp')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        M = T.dot(inputs[0], self.W0) + T.dot(inputs[1], self.W1).dimshuffle(0, 'x', 1)
        M = self.nonlinearity(M)
        alpha = T.nnet.softmax(T.dot(M, self.Wb))
        if len(inputs) == 3:
            alpha = alpha * inputs[2]
            alpha = alpha / alpha.sum(axis=1).reshape((alpha.shape[0], 1))
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)


class LengthLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return T.cast(input.sum(axis=-1), 'int32')

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]

class QuerySliceLayer(L.MergeLayer):
    # inputs[0]: B * Q * 2D (q)
    # inputs[1]: B          (q_var)

    def get_output_for(self, inputs, **kwargs):
        q_slice = inputs[0][T.arange(inputs[0].shape[0]), inputs[1] - 1, :]     # B x 2D
        return q_slice

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][2])


class BilinearAttentionPQLayer(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer between P and Q.
    """
    def __init__(self, incomings, num_units,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 4:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(BilinearAttentionPQLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        if 'name' not in kwargs:
            self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear')
        else:
            self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear_{}'.format(kwargs['name']))


    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        # inputs[0]: batch * len * h, P
        # inputs[1]: batch * len2 * h, Q
        # inputs[2]: batch * len, P mask
        # inputs[3]: batch * len2, Q mask
        # W: h * h
        B, PL, D = inputs[0].shape
        B, QL, D = inputs[1].shape
        input0 = T.reshape(inputs[0], (B * PL, D))
        input1 = inputs[1]
        input2 = inputs[2]
        input3 = inputs[3]
        doc_qry_mask = T.batched_dot(input2.dimshuffle((0, 1, 'x')), input3.dimshuffle(0, 'x', 1))
        M = T.dot(input0, self.W).reshape((B, PL, D))
        # batch * len * len2
        doc_qry_mat = T.batched_dot(M, input1.dimshuffle((0, 2, 1)))
        alpha = softmax(doc_qry_mat) * doc_qry_mask
        alpha = alpha / (alpha.sum(axis=2, keepdims=True) + 1e-8) * doc_qry_mask
        q_aware_P = T.batched_dot(alpha, input1)
        return q_aware_P



class BilinearAttentionPALayer(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer between P and A.
    """
    def __init__(self, incomings, num_units,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 4:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(BilinearAttentionPALayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        if 'name' not in kwargs:
            self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear')
            self.fc1_W = self.add_param(init, (2 * self.num_units, self.num_units), name='fc1_W')
            self.fc2_W = self.add_param(init, (self.num_units, ), name='fc2_W')
        else:
            self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear_{}'.format(kwargs['name']))
            self.fc1_W = self.add_param(init, (2 * self.num_units, self.num_units), name='fc1_W_{}'.format(kwargs['name']))
            self.fc2_W = self.add_param(init, (self.num_units, ), name='fc2_W_{}'.format(kwargs['name']))

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], 4)

    def get_output_for(self, inputs, **kwargs):
        # inputs[0]: (batch * 4) * len * h, A
        # inputs[1]: batch * len2 * h, P
        # inputs[2]: (batch * 4) * len, A mask
        # inputs[3]: batch * len2, P mask
        # W: h * h
        B4, AL, D = inputs[0].shape
        B, PL, D = inputs[1].shape
        input0 = T.reshape(inputs[0], (B4*AL, D))
        input1 = inputs[1]
        input2 = inputs[2]
        input2 = T.reshape(input2, (B, 4*AL))
        input3 = inputs[3]
        doc_qry_mask = T.batched_dot(input2.dimshuffle((0, 1, 'x')), input3.dimshuffle(0, 'x', 1))
        M = T.dot(input0, self.W).reshape((B, 4*AL, D))
        # batch * 4*len * len2
        doc_qry_mat = T.batched_dot(M, input1.dimshuffle((0, 2, 1)))
        alpha = softmax(doc_qry_mat) * doc_qry_mask
        alpha = alpha / (alpha.sum(axis=2, keepdims=True) + 1e-8) * doc_qry_mask
        doc_qry_mat = doc_qry_mat.reshape((B, 4, AL, PL))
        doc_qry_mask = doc_qry_mask.reshape((B, 4, AL, PL))
        # B * 4 * AL * D
        p_aware_A = T.batched_dot(alpha, input1).reshape((B, 4, AL, D))
        input0 = T.reshape(input0, (B, 4, AL, D))
        all_aware_P = []
        for i in xrange(4):
            beta = softmax(doc_qry_mat[:,i,:,:], axis=1) * doc_qry_mask[:,i,:,:]
            beta = beta / (beta.sum(axis=1, keepdims=True) + 1e-8) * doc_qry_mask[:,i,:,:]
            # B * PL * D
            a_aware_P = T.batched_dot(beta.dimshuffle((0, 2, 1)), input0[:,i,:,:])
            all_aware_P.append(a_aware_P)
        # B * 4 * PL * D
        a_aware_P = T.stack(all_aware_P, axis=1)
        # B * 4 * D
        p_aware_A = p_aware_A.sum(axis=2)
        # B * 4 * D
        a_aware_P = a_aware_P.sum(axis=2)
        # B * 4 * 2D
        out = T.concatenate([p_aware_A, a_aware_P], axis=-1)
        out = out.reshape((B * 4, 2 * D))
        fc1 = T.nnet.relu(T.dot(out, self.fc1_W))
        fc2 = T.dot(fc1, self.fc2_W)
        # B * 4
        fc2 = fc2.reshape((B, 4))
        out = softmax(fc2)
        return out

class BilinearAttentionLayer(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
    """
    def __init__(self, incomings, num_units,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(BilinearAttentionLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        if 'name' not in kwargs:
            self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear')
        else:
            self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear_{}'.format(kwargs['name']))


    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):

        # inputs[0]: batch * len * h
        # inputs[1]: batch * h
        # W: h * h
        M = T.dot(inputs[1], self.W).dimshuffle(0, 'x', 1)
        alpha = T.nnet.softmax(T.sum(inputs[0] * M, axis=2))
        if len(inputs) == 3:
            alpha = alpha * inputs[2]
            alpha = alpha / (alpha.sum(axis=1).reshape((alpha.shape[0], 1)) + 1e-8) * inputs[2]
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)

class BilinearDotLayer(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
    """
    def __init__(self, incomings, num_units,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(BilinearDotLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        if 'name' not in kwargs:
            self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear')
        else:
            self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear_{}'.format(kwargs['name']))

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:2]

    def get_output_for(self, inputs, **kwargs):

        # inputs[0]: batch * len * h
        # inputs[1]: batch * h
        # inputs[2]: batch * len
        # W: h * h
        M = T.dot(inputs[1], self.W).dimshuffle(0, 'x', 1) #batch * 1 * h
        alpha = T.nnet.softmax(T.sum(inputs[0] * M, axis=2)) #batch * len
        if len(inputs) == 3:
            alpha = alpha * inputs[2]
            alpha = alpha / (alpha.sum(axis=1).reshape((alpha.shape[0], 1)) + 1e-8) * inputs[2]
        return alpha

class BilinearDotLayerTensor(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x len x h

    """
    def __init__(self, incomings, num_units,
                 mask_input=None,
                 init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(BilinearDotLayerTensor, self).__init__(incomings, **kwargs)
        self.num_units = num_units

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:2]

    def get_output_for(self, inputs, **kwargs):
        alpha = T.nnet.softmax(T.sum(inputs[0] * inputs[1], axis=2))
        return alpha

class DotProductAttentionLayer(lasagne.layers.MergeLayer):
    """
        A bilinear attention layer.
        incomings[0]: batch x len x h
        incomings[1]: batch x h
    """
    def __init__(self, incomings, mask_input=None, **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        if mask_input is not None:
            incomings.append(mask_input)
        super(DotProductAttentionLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):

        # inputs[0]: batch * len * h
        # inputs[1]: batch * h
        # mask_input (if any): batch * len

        alpha = T.nnet.softmax(T.sum(inputs[0] * inputs[1].dimshuffle(0, 'x', 1), axis=2))
        if len(inputs) == 3:
            alpha = alpha * inputs[2]
            alpha = alpha / alpha.sum(axis=1).reshape((alpha.shape[0], 1))
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)
