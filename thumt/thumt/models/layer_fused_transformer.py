# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None,
                        dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        layer_wise_state = {}
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)
            layer_wise_state[layer_name] = x  # x, y can both be tried

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, layer_wise_state, next_state

        return outputs, layer_wise_state


def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

    bias = tf.get_variable("bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    encoder_input = layers.attention.add_timing_signal(encoder_input)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params)

    return encoder_output


def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    tgt_seq = features["target"]
    if mode == "train":
        tgt_l0_seq = features["target_l0"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    if mode == "train":
        t_l0_vocab = params.vocabulary["target_l0"]
        tgt_l0_vocab_size = len(t_l0_vocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)
        if mode == "train":
            weights_l0 = tf.get_variable("softmax_l0", [tgt_l0_vocab_size, hidden_size],
                                    initializer=initializer)

    targets = tf.gather(tgt_embedding, tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]

    if mode != "infer":
        decoder_output, layer_wise_state = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params)
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, _, decoder_state = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

    # regularize layer_4
    #decoder_output_l0 = layer_wise_state["layer_4"]
    # regularize layer_3
    #decoder_output_l0 = layer_wise_state["layer_3"]
    # regularize layer_2
    #decoder_output_l0 = layer_wise_state["layer_2"]
    # regularize layer_1
    decoder_output_l0 = layer_wise_state["layer_1"]
    # regularize layer_0
    #decoder_output_l0 = layer_wise_state["layer_0"]

    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    decoder_output_l0 = tf.reshape(decoder_output_l0, [-1, hidden_size])

    logits = tf.matmul(decoder_output, weights, False, True)
    logits_l0 = tf.matmul(decoder_output_l0, weights_l0, False, True)

    labels = features["target"]
    labels_l0 = features["target_l0"]

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    ce_l0 = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits_l0,
        labels=labels_l0,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))
    ce_l0 = tf.reshape(ce_l0, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    loss_l0 = tf.reduce_sum(ce_l0 * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss, loss_l0


def decoding_graph_l4_expect(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    t_l4_vocab = params.vocabulary["target_l0"]
    tgt_l4_vocab_size = len(t_l4_vocab)

    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)
        
        weights_l4 = tf.get_variable("softmax_l4", [tgt_l4_vocab_size, hidden_size],
                                    initializer=initializer)

    targets = tf.gather(tgt_embedding, tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]

    if mode != "infer":
        decoder_output, layer_wise_state = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params)
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, layer_wise_state, decoder_state = decoder_outputs
        decoder_output_l4 = layer_wise_state["layer_4"]
        decoder_output = decoder_output[:, -1, :]
        decoder_output_l4 = decoder_output_l4[:, -1, :]
        
        logits_l4 = tf.matmul(decoder_output_l4, weights_l4, False, True)
        softmax_l4 = tf.nn.softmax(logits_l4)
        expected_l4_emb = tf.matmul(softmax_l4, weights_l4)

        merged_decoder_output = expected_l4_emb + decoder_output

        logits = tf.matmul(merged_decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

    decoder_output_l4 = layer_wise_state["layer_4"]

    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    decoder_output_l4 = tf.reshape(decoder_output_l4, [-1, hidden_size])

    logits_l4 = tf.matmul(decoder_output_l4, weights_l4, False, True)
    softmax_l4 = tf.nn.softmax(logits_l4)  # [N, V_L0]
    expected_l4_emb = tf.matmul(softmax_l4, weights_l4)  # [N, D]
    merged_decoder_output = expected_l4_emb + decoder_output
    
    logits = tf.matmul(merged_decoder_output, weights, False, True)

    labels = features["target"]
    labels_l4 = features["target_l0"]

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    ce_l4 = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits_l4,
        labels=labels_l4,
        smoothing=params.label_smoothing,
        normalize=True
    )

    ce = tf.reshape(ce, tf.shape(tgt_seq))
    ce_l4 = tf.reshape(ce_l4, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    loss_l4 = tf.reduce_sum(ce_l4 * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss, loss_l4


def uncertainty_weighed_layer_fusion_decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    targets = tf.gather(tgt_embedding, tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]

    if mode != "infer":
        decoder_output, layer_wise_state = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params)
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, layer_wise_state, decoder_state = decoder_outputs
        # layer_5
        decoder_output_l5 = layer_wise_state["layer_5"]
        # layer_4
        decoder_output_l4 = layer_wise_state["layer_4"]
        # layer_3
        decoder_output_l3 = layer_wise_state["layer_3"]
        # layer_2
        decoder_output_l2 = layer_wise_state["layer_2"]
        # layer_1
        decoder_output_l1 = layer_wise_state["layer_1"]
        # layer_0
        decoder_output_l0 = layer_wise_state["layer_0"]

        #decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
        #import ipdb; ipdb.set_trace()
        decoder_output_l5 = tf.reshape(decoder_output_l5[:, -1, :], [-1, hidden_size])  # [N, D]
        decoder_output_l4 = tf.reshape(decoder_output_l4[:, -1, :], [-1, hidden_size])
        decoder_output_l3 = tf.reshape(decoder_output_l3[:, -1, :], [-1, hidden_size])
        decoder_output_l2 = tf.reshape(decoder_output_l2[:, -1, :], [-1, hidden_size])
        decoder_output_l1 = tf.reshape(decoder_output_l1[:, -1, :], [-1, hidden_size])
        decoder_output_l0 = tf.reshape(decoder_output_l0[:, -1, :], [-1, hidden_size])

        logits_l5 = tf.matmul(decoder_output_l5, weights, False, True)
        logits_l4 = tf.matmul(decoder_output_l4, weights, False, True)
        logits_l3 = tf.matmul(decoder_output_l3, weights, False, True)
        logits_l2 = tf.matmul(decoder_output_l2, weights, False, True)
        logits_l1 = tf.matmul(decoder_output_l1, weights, False, True)
        logits_l0 = tf.matmul(decoder_output_l0, weights, False, True)  # [N x T, V]

        plogp_l5 = tf.nn.softmax(logits_l5, -1) * tf.nn.log_softmax(logits_l5, -1)  # [N, V]
        plogp_l4 = tf.nn.softmax(logits_l4, -1) * tf.nn.log_softmax(logits_l4, -1)  # [N x T, V]
        plogp_l3 = tf.nn.softmax(logits_l3, -1) * tf.nn.log_softmax(logits_l3, -1)  # [N x T, V]
        plogp_l2 = tf.nn.softmax(logits_l2, -1) * tf.nn.log_softmax(logits_l2, -1)  # [N x T, V]
        plogp_l1 = tf.nn.softmax(logits_l1, -1) * tf.nn.log_softmax(logits_l1, -1)  # [N x T, V]
        plogp_l0 = tf.nn.softmax(logits_l0, -1) * tf.nn.log_softmax(logits_l0, -1)  # [N x T, V]

        ent_l5 = tf.reduce_sum(plogp_l5, -1, keep_dims=False)
        ent_l4 = tf.reduce_sum(plogp_l4, -1, keep_dims=False)
        ent_l3 = tf.reduce_sum(plogp_l3, -1, keep_dims=False)
        ent_l2 = tf.reduce_sum(plogp_l2, -1, keep_dims=False)
        ent_l1 = tf.reduce_sum(plogp_l1, -1, keep_dims=False)
        ent_l0 = tf.reduce_sum(plogp_l0, -1, keep_dims=False)  # [N x T, 1]

        ent = tf.stack([1.0/ent_l5, 1.0/ent_l4, 1.0/ent_l3, 1.0/ent_l2, 1.0/ent_l1, 1.0/ent_l0], axis=1)  # [N x T, 6]
        T = 1.0
        ent = ent / T
        ent_norm = tf.nn.softmax(ent, -1)  # [N x T, 6]

        coef_l5, coef_l4, coef_l3, coef_l2, coef_l1, coef_l0 = tf.split(ent_norm, 6, axis=1)  # [N x T, 1]
        decoder_output_l5_weighed = coef_l5 * decoder_output_l5
        decoder_output_l4_weighed = coef_l4 * decoder_output_l4
        decoder_output_l3_weighed = coef_l3 * decoder_output_l3
        decoder_output_l2_weighed = coef_l2 * decoder_output_l2
        decoder_output_l1_weighed = coef_l1 * decoder_output_l1
        decoder_output_l0_weighed = coef_l0 * decoder_output_l0

        fused_decoder_output = decoder_output_l5_weighed + decoder_output_l4_weighed + decoder_output_l3_weighed + decoder_output_l2_weighed + decoder_output_l1_weighed + decoder_output_l0_weighed
        fused_logits = tf.matmul(fused_decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(fused_logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state}
    
    # layer_5
    decoder_output_l5 = layer_wise_state["layer_5"]
    # layer_4
    decoder_output_l4 = layer_wise_state["layer_4"]
    # layer_3
    decoder_output_l3 = layer_wise_state["layer_3"]
    # layer_2
    decoder_output_l2 = layer_wise_state["layer_2"]
    # layer_1
    decoder_output_l1 = layer_wise_state["layer_1"]
    # layer_0
    decoder_output_l0 = layer_wise_state["layer_0"]

    #decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    decoder_output_l5 = tf.reshape(decoder_output_l5, [-1, hidden_size])
    decoder_output_l4 = tf.reshape(decoder_output_l4, [-1, hidden_size])
    decoder_output_l3 = tf.reshape(decoder_output_l3, [-1, hidden_size])
    decoder_output_l2 = tf.reshape(decoder_output_l2, [-1, hidden_size])
    decoder_output_l1 = tf.reshape(decoder_output_l1, [-1, hidden_size])
    decoder_output_l0 = tf.reshape(decoder_output_l0, [-1, hidden_size])

    logits_l5 = tf.matmul(decoder_output_l5, weights, False, True)
    logits_l4 = tf.matmul(decoder_output_l4, weights, False, True)
    logits_l3 = tf.matmul(decoder_output_l3, weights, False, True)
    logits_l2 = tf.matmul(decoder_output_l2, weights, False, True)
    logits_l1 = tf.matmul(decoder_output_l1, weights, False, True)
    logits_l0 = tf.matmul(decoder_output_l0, weights, False, True)  # [N x T, V]

    plogp_l5 = tf.nn.softmax(logits_l5, -1) * tf.nn.log_softmax(logits_l5, -1)  # [N x T, V]
    plogp_l4 = tf.nn.softmax(logits_l4, -1) * tf.nn.log_softmax(logits_l4, -1)  # [N x T, V]
    plogp_l3 = tf.nn.softmax(logits_l3, -1) * tf.nn.log_softmax(logits_l3, -1)  # [N x T, V]
    plogp_l2 = tf.nn.softmax(logits_l2, -1) * tf.nn.log_softmax(logits_l2, -1)  # [N x T, V]
    plogp_l1 = tf.nn.softmax(logits_l1, -1) * tf.nn.log_softmax(logits_l1, -1)  # [N x T, V]
    plogp_l0 = tf.nn.softmax(logits_l0, -1) * tf.nn.log_softmax(logits_l0, -1)  # [N x T, V]

    ent_l5 = tf.reduce_sum(plogp_l5, -1, keep_dims=False)
    ent_l4 = tf.reduce_sum(plogp_l4, -1, keep_dims=False)
    ent_l3 = tf.reduce_sum(plogp_l3, -1, keep_dims=False)
    ent_l2 = tf.reduce_sum(plogp_l2, -1, keep_dims=False)
    ent_l1 = tf.reduce_sum(plogp_l1, -1, keep_dims=False)
    ent_l0 = tf.reduce_sum(plogp_l0, -1, keep_dims=False)  # [N x T, 1]

    ent = tf.stack([1.0/ent_l5, 1.0/ent_l4, 1.0/ent_l3, 1.0/ent_l2, 1.0/ent_l1, 1.0/ent_l0], axis=1)  # [N x T, 6]
    T = 1.0
    ent = ent / T
    ent_norm = tf.nn.softmax(ent, -1)  # [N x T, 6]
    
    coef_l5, coef_l4, coef_l3, coef_l2, coef_l1, coef_l0 = tf.split(ent_norm, 6, axis=1)  # [N x T, 1]
    decoder_output_l5_weighed = coef_l5 * decoder_output_l5
    decoder_output_l4_weighed = coef_l4 * decoder_output_l4
    decoder_output_l3_weighed = coef_l3 * decoder_output_l3
    decoder_output_l2_weighed = coef_l2 * decoder_output_l2
    decoder_output_l1_weighed = coef_l1 * decoder_output_l1
    decoder_output_l0_weighed = coef_l0 * decoder_output_l0

    fused_decoder_output = decoder_output_l5_weighed + decoder_output_l4_weighed + decoder_output_l3_weighed + decoder_output_l2_weighed + decoder_output_l1_weighed + decoder_output_l0_weighed

    fused_logits = tf.matmul(fused_decoder_output, weights, False, True)
    labels = features["target"]
    #labels_l0 = features["target_l0"]

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=fused_logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    # Uncertainty loss
    ce_l5 = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits_l5,
        labels=labels,
        smoothing=None,
        normalize=True
    )
    ce_l4 = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits_l4,
        labels=labels,
        smoothing=None,
        normalize=True
    )
    ce_l3 = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits_l3,
        labels=labels,
        smoothing=None,
        normalize=True
    )
    ce_l2 = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits_l2,
        labels=labels,
        smoothing=None,
        normalize=True
    )
    ce_l1 = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits_l1,
        labels=labels,
        smoothing=None,
        normalize=True
    )
    ce_l0 = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits_l0,
        labels=labels,
        smoothing=None,
        normalize=True
    )
    #ce_l0 = layers.nn.smoothed_softmax_cross_entropy_with_logits(
    #    logits=logits_l0,
    #    labels=labels_l0,
    #    smoothing=params.label_smoothing,
    #    normalize=True
    #)

    ce = tf.reshape(ce, tf.shape(tgt_seq))
    ce_l5 = tf.reshape(ce_l5, tf.shape(tgt_seq))
    ce_l4 = tf.reshape(ce_l4, tf.shape(tgt_seq))
    ce_l3 = tf.reshape(ce_l3, tf.shape(tgt_seq))
    ce_l2 = tf.reshape(ce_l2, tf.shape(tgt_seq))
    ce_l1 = tf.reshape(ce_l1, tf.shape(tgt_seq))
    ce_l0 = tf.reshape(ce_l0, tf.shape(tgt_seq))
    #ce_l0 = tf.reshape(ce_l0, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    loss_l5 = tf.reduce_sum(ce_l5 * tgt_mask) / tf.reduce_sum(tgt_mask)
    loss_l4 = tf.reduce_sum(ce_l4 * tgt_mask) / tf.reduce_sum(tgt_mask)
    loss_l3 = tf.reduce_sum(ce_l3 * tgt_mask) / tf.reduce_sum(tgt_mask)
    loss_l2 = tf.reduce_sum(ce_l2 * tgt_mask) / tf.reduce_sum(tgt_mask)
    loss_l1 = tf.reduce_sum(ce_l1 * tgt_mask) / tf.reduce_sum(tgt_mask)
    loss_l0 = tf.reduce_sum(ce_l0 * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss, loss_l5, loss_l4, loss_l3, loss_l2, loss_l1, loss_l0


def model_graph(features, mode, params):
    encoder_output = encoding_graph(features, mode, params)
    state = {
        "encoder": encoder_output
    }
    output = uncertainty_weighed_layer_fusion_decoding_graph(features, state, mode, params)

    return output


class Transformer(interface.NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = uncertainty_weighed_layer_fusion_decoding_graph(features, state, "infer",
                                                                                        params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            submodel="c2f-l4",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0
        )

        return params
