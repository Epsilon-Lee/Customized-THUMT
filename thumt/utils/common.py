# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def infer_shape(x):
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.shape.dims is None:
        return tf.shape(x)

    static_shape = x.shape.as_list()
    dynamic_shape = tf.shape(x)

    ret = []
    for i in range(len(static_shape)):
        dim = static_shape[i]
        if dim is None:
            dim = dynamic_shape[i]
        ret.append(dim)

    return ret


def infer_shape_invariants(tensor):
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


def merge_first_two_dims(tensor):
    shape = infer_shape(tensor)
    shape[0] *= shape[1]
    shape.pop(1)
    return tf.reshape(tensor, shape)


def split_first_two_dims(tensor, dim_0, dim_1):
    shape = infer_shape(tensor)
    new_shape = [dim_0] + [dim_1] + shape[1:]
    return tf.reshape(tensor, new_shape)


def tile_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size. """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size

    return tf.tile(tensor, tile_dims)


def tile_batch(tensor, batch_size):
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[0] = batch_size

    return tf.tile(tensor, tile_dims)


def gather_2d(params, indices, name=None):
    """ Gather the 2nd dimension given indices
    :param params: A tensor with shape [batch_size, M, ...]
    :param indices: A tensor with shape [batch_size, N]
    :param name: An optional string
    :return: A tensor with shape [batch_size, N, ...]
    """
    batch_size = tf.shape(params)[0]
    range_size = tf.shape(indices)[1]
    batch_pos = tf.range(batch_size * range_size) // range_size
    batch_pos = tf.reshape(batch_pos, [batch_size, range_size])
    indices = tf.stack([batch_pos, indices], axis=-1)
    output = tf.gather_nd(params, indices, name=name)

    return output

def assembly_gather_reduce_sum(gather_src, gather_index, index_mask):
    """ Used for gather_index is so large that leads to OOM on GPU, 
    for example [5003, 458, 4080]
    """
    size_axis_0 = gather_index.shape[0].value
    gather_index_list = tf.split(gather_index, size_axis_0, axis=0)
    index_mask_list = tf.split(index_mask, size_axis_0, axis=0)
    index_length_list = tf.split(tf.reduce_sum(tf.squeeze(index_mask, 2), axis=1), size_axis_0, axis=0)  # [V_i, ]
    #res = tf.zeros([0, gather_src.shape[-1].value], dtype=tf.float32)
    gather_list = []
    #import ipdb; ipdb.set_trace()
    for gather_index_i, index_mask_i, length_i in zip(gather_index_list, index_mask_list, index_length_list):
        #res = tf.concat([res, tf.reduce_sum(tf.gather(gather_src, gather_index_i) * index_mask_i, axis=0)], axis=0)
        len_i = tf.to_int32(length_i[0])
        #gather_list.append(tf.reduce_sum(tf.gather(gather_src, gather_index_i[:, :len_i]) * index_mask_i, axis=1))
        gather_list.append(tf.reduce_sum(tf.gather(gather_src, gather_index_i[:, :len_i]), axis=1))
        #tmp = tf.gather(gather_src, gather_index_i)  # [1, max_cluster_size, ...]
        #tmp = tmp * index_mask_i  # [1, max_cluster_size, ...]
        #tf.reduce(sum)
    res = tf.stack(gather_list, axis=0)
    #ipdb.set_trace()
    return tf.squeeze(res, axis=1)
