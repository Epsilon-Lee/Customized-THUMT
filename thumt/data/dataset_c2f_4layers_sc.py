# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import operator

import numpy as np
import tensorflow as tf


def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True):
    """ Batch examples

    :param example: A dictionary of <feature name, Tensor>.
    :param batch_size: The number of tokens or sentences in a batch
    :param max_length: The maximum length of a example to keep
    :param mantissa_bits: An integer
    :param shard_multiplier: an integer increasing the batch_size to suit
        splitting across data shards.
    :param length_multiplier: an integer multiplier that is used to
        increase the batch sizes and sequence length tolerance.
    :param constant: Whether to use constant batch size
    :param num_threads: Number of threads
    :param drop_long_sequences: Whether to drop long sequences

    :returns: A dictionary of batched examples
    """

    with tf.name_scope("batch_examples"):
        max_length = max_length or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        # Compute boundaries
        x = min_length
        boundaries = []

        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

        # Whether the batch size is constant
        if not constant:
            batch_sizes = [max(1, batch_size // length)
                           for length in boundaries + [max_length]]
            batch_sizes = [b * shard_multiplier for b in batch_sizes]
            bucket_capacities = [2 * b for b in batch_sizes]
        else:
            batch_sizes = batch_size * shard_multiplier
            bucket_capacities = [2 * n for n in boundaries + [max_length]]

        max_length *= length_multiplier
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length = max_length if drop_long_sequences else 10 ** 9

        # The queue to bucket on will be chosen based on maximum length
        max_example_length = 0
        for v in example.values():
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,
            example,
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=bucket_capacities,
            dynamic_pad=True,
            keep_input=(max_example_length <= max_length)
        )

    return outputs


def read_index_map_from_file(index_map_filename):
    index_maps = []
    index_lengths = []
    max_len = 0
    with open(index_map_filename, 'r') as f:
        for line in f:
            indices = [int(idx) for idx in line.strip().split()]
            indices_len = len(indices)
            index_lengths.append(indices_len)
            if indices_len > max_len:
                max_len = len(indices)
            index_maps.append(indices)
    tensor = np.zeros([len(index_maps), max_len], dtype=int)
    tensor_tf = tf.constant(tensor, dtype=tf.int32)
    for idx, indices in enumerate(index_maps):
        tensor[idx][:index_lengths[idx]] = indices
    mask = tf.sequence_mask(index_lengths, maxlen=max_len, dtype=tf.float32)
    return tensor_tf, mask


def get_training_input_and_c2f_label(filenames, c2f_filenames, index_map_filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])
        tgt_l1_dataset = tf.data.TextLineDataset(c2f_filenames[0])
        tgt_l2_dataset = tf.data.TextLineDataset(c2f_filenames[1])
        tgt_l3_dataset = tf.data.TextLineDataset(c2f_filenames[2])
        tgt_l4_dataset = tf.data.TextLineDataset(c2f_filenames[3])

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, tgt_l1_dataset, tgt_l2_dataset, tgt_l3_dataset, tgt_l4_dataset))
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        # Split string
        dataset = dataset.map(
            lambda src, tgt, tgt_l1, tgt_l2, tgt_l3, tgt_l4: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values,
                tf.string_split([tgt_l1]).values,
                tf.string_split([tgt_l2]).values,
                tf.string_split([tgt_l3]).values,
                tf.string_split([tgt_l4]).values,
            ),
            num_parallel_calls=params.num_threads
        )

        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt, tgt_l1, tgt_l2, tgt_l3, tgt_l4: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt_l1, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt_l2, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt_l3, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt_l4, [tf.constant(params.eos)]], axis=0),
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt, tgt_l1, tgt_l2, tgt_l3, tgt_l4: {
                "source": src,
                "target": tgt,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt),
                "target_l1": tgt_l1,
                "target_l2": tgt_l2,
                "target_l3": tgt_l3,
                "target_l4": tgt_l4,
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )
        tgt_l1_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target_l1"]),
            default_value=params.mapping["target_l1"][params.unk]
        )
        tgt_l2_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target_l2"]),
            default_value=params.mapping["target_l2"][params.unk]
        )
        tgt_l3_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target_l3"]),
            default_value=params.mapping["target_l3"][params.unk]
        )
        tgt_l4_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target_l4"]),
            default_value=params.mapping["target_l4"][params.unk]
        )

        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])
        features["target_l1"] = tgt_l1_table.lookup(features["target_l1"])
        features["target_l2"] = tgt_l2_table.lookup(features["target_l2"])
        features["target_l3"] = tgt_l3_table.lookup(features["target_l3"])
        features["target_l4"] = tgt_l4_table.lookup(features["target_l4"])

        # Batching
        shard_multiplier = len(params.device_list) * params.update_cycle
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=shard_multiplier,
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Get index_map as constant tensors, where 'l1' means map between l1 and l2
        #import ipdb; ipdb.set_trace()
        index_map_l1, index_map_l1_mask = read_index_map_from_file(index_map_filenames[0])
        index_map_l2, index_map_l2_mask = read_index_map_from_file(index_map_filenames[1])
        index_map_l3, index_map_l3_mask = read_index_map_from_file(index_map_filenames[2])
        index_map_l4, index_map_l4_mask = read_index_map_from_file(index_map_filenames[3])

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["target_l1"] = tf.to_int32(features["target_l1"])
        features["target_l2"] = tf.to_int32(features["target_l2"])
        features["target_l3"] = tf.to_int32(features["target_l3"])
        features["target_l4"] = tf.to_int32(features["target_l4"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        constant_features = {}
        constant_features["index_map_l1"] = index_map_l1
        constant_features["index_map_l2"] = index_map_l2
        constant_features["index_map_l3"] = index_map_l3
        constant_features["index_map_l4"] = index_map_l4
        constant_features["index_map_l1_mask"] = index_map_l1_mask
        constant_features["index_map_l2_mask"] = index_map_l2_mask
        constant_features["index_map_l3_mask"] = index_map_l3_mask
        constant_features["index_map_l4_mask"] = index_map_l4_mask
        
        #sess = tf.train.MonitoredTrainingSession()
        #feats = sess.run(features)
        #import ipdb; ipdb.set_trace()

        return features, constant_features


def sort_input_file(filename, reverse=True):
    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_keys, sorted_inputs


def sort_and_zip_files(names):
    inputs = []
    input_lens = []
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=True)
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])

    return [list(x) for x in zip(*sorted_inputs)]


def get_evaluation_input(inputs, params):
    with tf.device("/cpu:0"):
        # Create datasets
        datasets = []

        for data in inputs:
            dataset = tf.data.Dataset.from_tensor_slices(data)
            # Split string
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)
            # Append <eos>
            dataset = dataset.map(
                lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                num_parallel_calls=params.num_threads
            )
            datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(datasets))

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda *x: {
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                "references": x[1:]
            },
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.eval_batch_size,
            {
                "source": [tf.Dimension(None)],
                "source_length": [],
                "references": (tf.Dimension(None),) * (len(inputs) - 1)
            },
            {
                "source": params.pad,
                "source_length": 0,
                "references": (params.pad,) * (len(inputs) - 1)
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Covert source symbols to ids
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )

        features["source"] = src_table.lookup(features["source"])

    return features


def get_inference_input(inputs, params):
    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(inputs)
        )

        # Split string
        dataset = dataset.map(lambda x: tf.string_split([x]).values,
                              num_parallel_calls=params.num_threads)

        # Append <eos>
        dataset = dataset.map(
            lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
            num_parallel_calls=params.num_threads
        )

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda x: {"source": x, "source_length": tf.shape(x)[0]},
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.decode_batch_size * len(params.device_list),
            {"source": [tf.Dimension(None)], "source_length": []},
            {"source": params.pad, "source_length": 0}
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        features["source"] = src_table.lookup(features["source"])

        return features
