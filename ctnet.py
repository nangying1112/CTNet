# -*- coding: utf-8 -*-

import tensorflow as tf
import time
from sklearn import metrics
from TDNN import TDNN
import numpy as np
from pdtb_data import pddata
from tensorflow.python.client import device_lib

path = r''

# print(tf.__version__)
# print(sklearn.__version__)
# print(allennlp.__version__)
# print(np.__version__)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('classes', 4, 'num-class classification:[2,4,11]')
flags.DEFINE_string('pos_class', 'Temporal', 'positive class in 2-class classification:')
flags.DEFINE_integer('embedding_size', 300, 'glove embedding size.')
flags.DEFINE_integer('vocab_size', 38927, 'vocab size.')
flags.DEFINE_integer('char_vocab_size', 84, 'vocab size.')
flags.DEFINE_integer('rnn_size', 128, 'hidden_units_size of lstm')
flags.DEFINE_integer('para_sen_num', 6, 'the num of sentences in a paragraph')
flags.DEFINE_integer('slstm_size', 600, 'hidden_units_size of slstm')
flags.DEFINE_integer('char_embed_dim', 300, 'char embedding size')
flags.DEFINE_integer('slstm_layer', 3, 'num layers of slstm')
flags.DEFINE_integer('slstm_steps', 2, 'steps')
flags.DEFINE_integer('slstm_gcn_layer', 3, 'num layers of slstm_gcn')
flags.DEFINE_integer('slstm_gcn_steps', 2, '')
flags.DEFINE_integer('batch_size', 2, 'batch_size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 40, 'Number of epochs to train.')
flags.DEFINE_integer('conn_nums', 132, 'Number of epochs to train.')
flags.DEFINE_string('arg_encoder', 'bilstm', 'encoder of arguments')
flags.DEFINE_float('dropout', 0.5, ' keep probability')
flags.DEFINE_float('embedding_drop', 0.9, ' keep probability')
flags.DEFINE_float('cell_drop', 0.9, ' keep probability')
flags.DEFINE_integer('elmo_cuda', 0, 'embedding size.')
flags.DEFINE_float('weight_decay', 0.9, 'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_boolean('use_para_info', True, '')
flags.DEFINE_boolean('use_hrnn', False, '')
flags.DEFINE_boolean('use_char', True, '')
flags.DEFINE_boolean('use_exp', True, '')
flags.DEFINE_boolean('use_elmo', True, '')
flags.DEFINE_boolean('use_mt', True, '')
flags.DEFINE_boolean('relation_balance', False, '')
flags.DEFINE_boolean('type_balance', True, '')
flags.DEFINE_integer('word_length', 27, '')
flags.DEFINE_string('gt', 'pag', 'graph type')

rate = 0.9
num_step = 2
num_layers = 3


class CTNET():

    def __init__(self, FLAGS, embedding):
        # inputs: features, mask, keep_prob, labels

        self.lr = tf.Variable(FLAGS.learning_rate, trainable=False)
        self.trainable = tf.placeholder(tf.bool, None)
        self.lr_decay_factor = tf.placeholder(tf.float32, None)
        self.lr_decay_op = tf.assign(self.lr, self.lr * self.lr_decay_factor)
        # batch_size, steps
        self.arg1_ids = tf.placeholder(tf.int32, [FLAGS.batch_size, None], "arg1_ids")
        self.arg1_len = tf.placeholder(tf.int32, [FLAGS.batch_size], "arg1_len")
        self.arg2_ids = tf.placeholder(tf.int32, [FLAGS.batch_size, None], "arg2_ids")
        self.arg2_len = tf.placeholder(tf.int32, [FLAGS.batch_size], "arg2_len")
        self.para_len = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.para_sen_num], "para_len")
        para_len = tf.reshape(self.para_len, [FLAGS.batch_size * FLAGS.para_sen_num], 'para_len_reshape')
        self.supports = tf.placeholder(tf.float32, [FLAGS.batch_size, 3, 6, 6])

        self.char1 = tf.placeholder(tf.int32, [FLAGS.batch_size, None, 27], "char1")
        self.char2 = tf.placeholder(tf.int32, [FLAGS.batch_size, None, 27], "char2")

        if FLAGS.use_elmo:
            self.arg1_elmo = tf.placeholder(tf.float32, [FLAGS.batch_size, 3, None, 1024], "arg1_elmo")
            self.arg2_elmo = tf.placeholder(tf.float32, [FLAGS.batch_size, 3, None, 1024], "arg2_elmo")
            if FLAGS.use_para_info:
                self.para_elmo = tf.placeholder(tf.float32, [FLAGS.batch_size * FLAGS.para_sen_num, 3, None, 1024],
                                                "para_elmo")
            else:
                self.para_elmo = tf.placeholder(tf.float32, [None], "para_elmo")
        else:
            self.arg1_elmo = tf.placeholder(tf.float32, [None], "arg1_elmo")
            self.arg2_elmo = tf.placeholder(tf.float32, [None], "arg2_elmo")
            self.para_elmo = tf.placeholder(tf.float32, [None], "para_elmo")

        self.para_ids = tf.placeholder(tf.int32, [FLAGS.batch_size, 6, None], "para_ids")
        self.para_chars = tf.placeholder(tf.int32, [FLAGS.batch_size, 6, None, 27], "para_chars")

        self.char_vocab_size = 84
        self.char_embed_dim = FLAGS.char_embed_dim

        self.labels = tf.placeholder(tf.int32, [FLAGS.batch_size], 'labels')
        self.conn_labels = tf.placeholder(tf.int32, [FLAGS.batch_size], 'conn_labels')
        self.type_labels = tf.placeholder(tf.int32, [FLAGS.batch_size], 'type_labels')

        self.global_step = tf.Variable(0, False)

        # vocabulary: pad, unk, ...

        if type(embedding) == type(None):
            embedding = tf.get_variable('embedding', [FLAGS.vocab_size - 1, FLAGS.embedding_size], tf.float32,
                                        tf.truncated_normal_initializer)
            # pad 不更新
            pad_embedding = tf.zeros([1, FLAGS.embedding_size], tf.float32)
            embedding = tf.concat([pad_embedding, embedding], axis=0)
        else:
            embedding = tf.constant(embedding, tf.float32)

        print(embedding.shape)
        # print(char1_cnn.output)
        self.arg1_embedded = tf.nn.embedding_lookup(embedding, self.arg1_ids)
        self.arg2_embedded = tf.nn.embedding_lookup(embedding, self.arg2_ids)

        print('para_ids', self.para_ids.shape)
        para_ids = tf.reshape(self.para_ids, [self.para_ids.shape[0] * self.para_ids.shape[1], -1], 'para_ids_reshape')
        self.para_embedded = tf.nn.embedding_lookup(embedding, para_ids)

        if FLAGS.use_char:
            char_W = tf.get_variable("char_embed", [self.char_vocab_size, self.char_embed_dim])
            char1_index = tf.reshape(self.char1, [-1, FLAGS.word_length], "char1_index_reshape")
            char2_index = tf.reshape(self.char2, [-1, FLAGS.word_length], "char2_index_reshape")

            char1_embed = tf.nn.embedding_lookup(char_W, char1_index)
            char2_embed = tf.nn.embedding_lookup(char_W, char2_index)

            para_chars_index = tf.reshape(self.para_chars, [-1, FLAGS.word_length], 'para_chars_reshape')

            para_char_embed = tf.nn.embedding_lookup(char_W, para_chars_index)

            with tf.variable_scope('char') as arg_scope:
                char1_cnn = TDNN(char1_embed, self.char_embed_dim)
                arg_scope.reuse_variables()
                char2_cnn = TDNN(char2_embed, self.char_embed_dim)
                arg_scope.reuse_variables()
                para_cnn = TDNN(para_char_embed, self.char_embed_dim)
                char1_cnn_out = tf.reshape(char1_cnn.output, [FLAGS.batch_size, -1, self.char_embed_dim])
                char2_cnn_out = tf.reshape(char2_cnn.output, [FLAGS.batch_size, -1, self.char_embed_dim])
                para_char_out = tf.reshape(para_cnn.output,
                                           [FLAGS.batch_size * FLAGS.para_sen_num, -1, self.char_embed_dim])
                print(char1_cnn.output.shape)

                self.arg1_embedded = tf.concat([self.arg1_embedded, char1_cnn_out], 2)
                self.arg2_embedded = tf.concat([self.arg2_embedded, char2_cnn_out], 2)
                self.para_embedded = tf.concat([self.para_embedded, para_char_out], 2)

        print('arg1embed', self.arg1_embedded.shape)
        print('arg1embed', self.arg2_embedded.shape)
        print('arg1elmo', self.arg2_elmo.shape)
        print('arg1elmo', self.arg2_elmo.shape)
        if FLAGS.use_elmo:
            self.arg1_embedded = tf.concat([self.arg1_embedded, tf.squeeze(self.arg1_elmo[:, 1, :, :])], 2)
            self.arg2_embedded = tf.concat([self.arg2_embedded, tf.squeeze(self.arg2_elmo[:, 1, :, :])], 2)
            self.arg1_embedded.set_shape([FLAGS.batch_size, None, 1024 + FLAGS.char_embed_dim + FLAGS.embedding_size])
            self.arg2_embedded.set_shape([FLAGS.batch_size, None, 1024 + FLAGS.char_embed_dim + FLAGS.embedding_size])
            # self.para_elmo = tf.reshape(self.para_elmo, [FLAGS.batch_size * FLAGS.para_sen_num, 3,
            #                                              -1, 1024], 'para_elmo_reshape')
            if FLAGS.use_para_info:
                print('para_elmo_concat', self.para_embedded.shape)
                self.para_embedded = tf.concat([self.para_embedded, tf.squeeze(self.para_elmo[:, 1, :, :])], 2)
        else:
            self.arg1_embedded.set_shape([FLAGS.batch_size, None, FLAGS.char_embed_dim + FLAGS.embedding_size])
            self.arg2_embedded.set_shape([FLAGS.batch_size, None, FLAGS.char_embed_dim + FLAGS.embedding_size])

        # apply embedding
        print('arg1embed', self.arg1_embedded.shape)
        print('arg1embed', self.arg2_embedded.shape)
        if FLAGS.arg_encoder == "bilstm":
            with tf.variable_scope('argument') as arg_scope:
                arg_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.rnn_size)
                # arg_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_size)
                arg_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.rnn_size)
                # arg_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_size)

                init_fw_state = tf.nn.rnn_cell.LSTMStateTuple(
                    0.001 * tf.truncated_normal([tf.shape(self.labels)[0], FLAGS.rnn_size], dtype=tf.float32),
                    0.001 * tf.truncated_normal([tf.shape(self.labels)[0], FLAGS.rnn_size], dtype=tf.float32)
                )

                init_bw_state = tf.nn.rnn_cell.LSTMStateTuple(
                    0.001 * tf.truncated_normal([tf.shape(self.labels)[0], FLAGS.rnn_size], dtype=tf.float32),
                    0.001 * tf.truncated_normal([tf.shape(self.labels)[0], FLAGS.rnn_size], dtype=tf.float32)
                )
                print('self.arg1', self.arg1_embedded.shape)

                arg1_outputs, arg1_final_states = tf.nn.bidirectional_dynamic_rnn(arg_fw_cell, arg_bw_cell,
                                                                                  self.arg1_embedded,
                                                                                  sequence_length=self.arg1_len,
                                                                                  dtype=tf.float32,
                                                                                  )

                arg_scope.reuse_variables()

                arg2_outputs, arg2_final_states = tf.nn.bidirectional_dynamic_rnn(arg_fw_cell, arg_bw_cell,
                                                                                  self.arg2_embedded,
                                                                                  sequence_length=self.arg2_len,
                                                                                  dtype=tf.float32,
                                                                                  # initial_state_fw=init_fw_state,
                                                                                  # initial_state_bw=init_bw_state,
                                                                                  )

            #
            bi_arg1_final_states_c = tf.concat([arg1_final_states[0][0], arg1_final_states[1][0]], axis=1)
            bi_arg2_final_states_c = tf.concat([arg2_final_states[0][0], arg2_final_states[1][0]], axis=1)
            # [batch_size,2*hidden_size] => [batch_size,4*hidden_size]
            rnn_out = tf.concat([bi_arg1_final_states_c, bi_arg2_final_states_c], axis=1)

            representation = tf.layers.dropout(rnn_out, rate=0, training=self.trainable)
        else:

            initial_arg1 = tf.nn.dropout(self.arg1_embedded, FLAGS.embedding_drop)
            initial_arg2 = tf.nn.dropout(self.arg2_embedded, FLAGS.embedding_drop)
            #    initial_para = tf.nn.dropout(self.para_embedded, FLAGS.embedding_drop)

            initial_arg1_cell = tf.identity(self.arg1_embedded)
            initial_arg2_cell = tf.identity(self.arg2_embedded)
            #  initial_para_cell = tf.identity(self.para_embedded)

            initial_arg1_cell = tf.nn.dropout(initial_arg1_cell, FLAGS.cell_drop)
            initial_arg2_cell = tf.nn.dropout(initial_arg2_cell, FLAGS.cell_drop)
            #  initial_para_cell = tf.nn.dropout(initial_para_cell, FLAGS.cell_drop)

            # create layers
            # for argument1
            new_hidden_states, new_cell_state, dummynode_hidden_states = self.slstm_cell("word_slstm",
                                                                                         FLAGS.slstm_size,
                                                                                         self.arg1_len,
                                                                                         initial_arg1,
                                                                                         initial_arg1_cell,
                                                                                         FLAGS.slstm_layer)

            # #representation=dummynode_hidden_states
            representation = tf.reduce_mean(
                tf.concat([new_hidden_states, tf.expand_dims(dummynode_hidden_states, axis=1)], axis=1), axis=1)
            # representation = tf.reduce_mean(dummynode_hidden_states, axis=1)

            softmax_w1 = tf.Variable(
                tf.random_normal([FLAGS.slstm_size, 2 * FLAGS.slstm_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                dtype=tf.float32, name="softmax_w1")
            softmax_b1 = tf.Variable(tf.random_normal([2 * FLAGS.slstm_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                     dtype=tf.float32, name="softmax_b1")
            arg1_representation = tf.nn.tanh(tf.matmul(representation, softmax_w1) + softmax_b1)

            # for argument 2
            arg2_new_hidden_states, arg2_new_cell_state, arg2_dummynode_hidden_states = self.slstm_cell("word_slstm",
                                                                                                        FLAGS.slstm_size,
                                                                                                        self.arg2_len,
                                                                                                        initial_arg2,
                                                                                                        initial_arg2_cell,
                                                                                                        FLAGS.slstm_layer
                                                                                                        )
            # representation=dummynode_hidden_states
            arg2_representation = tf.reduce_mean(
                tf.concat([arg2_new_hidden_states, tf.expand_dims(arg2_dummynode_hidden_states, axis=1)], axis=1),
                axis=1)

            softmax_w2 = tf.Variable(
                tf.random_normal([FLAGS.slstm_size, 2 * FLAGS.slstm_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                dtype=tf.float32, name="softmax_w2")
            softmax_b2 = tf.Variable(tf.random_normal([2 * FLAGS.slstm_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                     dtype=tf.float32, name="softmax_b2")
            arg2_representation = tf.nn.tanh(tf.matmul(arg2_representation, softmax_w2) + softmax_b2)
            representation = 0.5 * arg1_representation + 0.5 * arg2_representation

        if FLAGS.use_para_info:
            if FLAGS.use_hrnn:
                print('use hrnn')
                initial_para = tf.nn.dropout(self.para_embedded, FLAGS.embedding_drop)
                word_embedding = self.word_embedding(initial_para, para_len)
                word_c = tf.concat([word_embedding[0][0], word_embedding[1][0]], axis=1)
                # print('word_embedding', word_embedding.shape)
                word_embedding = tf.reshape(word_c, [FLAGS.batch_size, 6, -1])
                sen_embedding = self.sen_embedding(word_embedding)
                # print(sen_embedding.shape)
                print(sen_embedding[0][0].shape)
                # sen_embedding=tf.layers.dense(sen_embedding,2)
                print(representation.shape)
                sen_c = tf.concat([sen_embedding[0][0], sen_embedding[1][0]], axis=1)

                # [batch_size,2*hidden_size] => [batch_size,4*hidden_size]

                representation = tf.concat([representation, sen_c], 1)

                print('para', initial_para.shape)
            # for para_info
            else:
                if FLAGS.use_elmo:
                    self.para_embedded = tf.reshape(self.para_embedded,
                                                    [-1, 1024 + FLAGS.embedding_size + FLAGS.char_embed_dim])
                    softmax_wre = tf.Variable(
                        tf.random_normal([1024 + FLAGS.embedding_size + FLAGS.char_embed_dim, FLAGS.slstm_size],
                                         mean=0.0, stddev=0.1, dtype=tf.float32),
                        dtype=tf.float32, name="softmax_wre")
                    softmax_bre = tf.Variable(
                        tf.random_normal([FLAGS.slstm_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                        dtype=tf.float32, name="softmax_bre")
                else:
                    self.para_embedded = tf.reshape(self.para_embedded,
                                                    [-1, FLAGS.embedding_size + FLAGS.char_embed_dim])
                    softmax_wre = tf.Variable(
                        tf.random_normal([FLAGS.embedding_size + FLAGS.char_embed_dim, FLAGS.slstm_size], mean=0.0,
                                         stddev=0.1, dtype=tf.float32),
                        dtype=tf.float32, name="softmax_wre")
                    softmax_bre = tf.Variable(
                        tf.random_normal([FLAGS.slstm_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                        dtype=tf.float32, name="softmax_bre")
                self.para_embedded = tf.nn.tanh(tf.matmul(self.para_embedded, softmax_wre) + softmax_bre)
                self.para_embedded = tf.reshape(self.para_embedded,
                                                [6 * FLAGS.batch_size, -1, self.para_embedded.shape[-1]])
                initial_para = tf.nn.dropout(self.para_embedded, FLAGS.embedding_drop)
                initial_para_cell = tf.identity(self.para_embedded)
                initial_para_cell = tf.nn.dropout(initial_para_cell, FLAGS.cell_drop)
                para_new_hidden_states, para_new_cell_state, para_dummynode_hidden_states = self.slstm_gcn_cell(
                    "word_slstm",
                    FLAGS.slstm_size,
                    para_len,
                    initial_para,
                    initial_para_cell,
                    FLAGS.slstm_gcn_layer,
                    self.supports)
                # representation=dummynode_hidden_states
                para_representation = tf.reduce_mean(
                    tf.concat([para_new_hidden_states, tf.expand_dims(para_dummynode_hidden_states, axis=1)], axis=1),
                    axis=1)

                softmax_w3 = tf.Variable(
                    tf.random_normal([FLAGS.slstm_size, FLAGS.rnn_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="softmax_w3")
                softmax_b3 = tf.Variable(tf.random_normal([FLAGS.rnn_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                         dtype=tf.float32, name="softmax_b3")
                para_representation = tf.nn.tanh(tf.matmul(para_representation, softmax_w3) + softmax_b3)
                para_representation = tf.reduce_mean(
                    tf.reshape(para_representation, [FLAGS.batch_size, FLAGS.para_sen_num, -1]), axis=1)
                representation = tf.concat([representation, para_representation], 1)

        if FLAGS.use_exp == False:

            if FLAGS.arg_encoder == 'bilstm':
                dense1_out = tf.layers.dense(representation, 64, name='dense1', reuse=False)
                # dense1_out_bn = tf.layers.batch_normalization(dense1_out, trainable=trainable)
                dense1_out_ac = tf.nn.relu(dense1_out)
                dense1_out_drop = tf.layers.dropout(dense1_out_ac, rate=0., training=self.trainable)
                self.dense2_out = tf.layers.dense(dense1_out_drop, FLAGS.classes, name='dense2', reuse=False)
                # self.dense2_out_bn = tf.layers.batch_normalization(dense1_out, trainable=trainable)
                self.out = tf.nn.softmax(self.dense2_out)
                self.predict = tf.argmax(self.dense2_out, axis=1)
                self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.dense2_out)
            # relation classifier
            else:
                softmax_w = tf.Variable(
                    tf.random_normal([2 * FLAGS.slstm_size, FLAGS.classes], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="softmax_w")
                softmax_b = tf.Variable(tf.random_normal([FLAGS.classes], mean=0.0, stddev=0.1, dtype=tf.float32),
                                        dtype=tf.float32, name="softmax_b")

                self.logits = logits = tf.matmul(representation, softmax_w) + softmax_b
                self.out = tf.nn.softmax(logits)
                # operators for prediction
                self.predict = tf.argmax(logits, 1)
                # cross entropy loss
                self.rel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
                self.dense2_out = self.logits
                self.loss = loss = tf.reduce_mean(self.rel_loss)

        else:
            self.imp_mask = self.type_labels
            # explicit_classifier
            self.exp_mask = tf.to_int32(tf.ones(shape=self.imp_mask.shape)) - self.imp_mask
            self.imp_mask = self.imp_mask > 0
            self.exp_mask = self.exp_mask > 0
            # self.imp_mask = tf.boolean_mask(self.imp_mask, self.mask)
            # self.imp_mask = tf.boolean_mask(self.exp_imp_mask, tf.to_int32(tf.ones(shape=self.mask.shape))-self.mask)

            # exp/imp classification
            exp_imp_dense1 = tf.layers.dense(representation, 128, name='exp_imp_dense1', reuse=False)
            exp_imp_dense1 = tf.nn.relu(exp_imp_dense1)
            self.exp_imp_out = tf.layers.dense(exp_imp_dense1, 2, name='exp_imp', reuse=False)
            self.exp_imp_out1 = tf.nn.softmax(self.exp_imp_out, axis=1)
            self.exp_imp_predict = tf.argmax(self.exp_imp_out1, axis=1)
            # self.exp_imp_predict = tf.argmax(self.exp_imp_out1, axis=1)

            # self.new_exp_imp_labels = tf.boolean_mask(self.exp_imp_labels, self.mask, axis=0)
            self.new_exp_imp_labels = self.type_labels
            print('exp_imp_labels', self.type_labels.shape)
            print('exp_imp_out', self.exp_imp_out.shape)
            self.exp_imp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.new_exp_imp_labels,
                                                                               logits=self.exp_imp_out)

            # Explicit classifier
            self.exp_inputs = tf.boolean_mask(representation, self.exp_mask, axis=0)
            exp_dense1 = tf.layers.dense(self.exp_inputs, 128, name='exp_dense1', reuse=False)
            exp_dense1_drop = tf.layers.dropout(exp_dense1, rate=0., training=self.trainable)
            exp_dense1_ac = tf.nn.relu(exp_dense1_drop)

            self.dense2_out = self.exp_dense2 = tf.layers.dense(exp_dense1_ac, FLAGS.classes, name='exp_dense2',
                                                                reuse=False)
            self.out = self.exp_out = tf.nn.softmax(self.exp_dense2, axis=1)
            self.exp_predict = tf.argmax(self.exp_out, axis=1)

            print('exp_mask', self.exp_mask)
            self.exp_labels = tf.boolean_mask(self.labels, self.exp_mask, axis=0)

            print('exp_labels', self.exp_labels.shape)
            print('exp_out', self.exp_out.shape)

            self.exp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.exp_labels,
                                                                           logits=self.exp_dense2)

            # Implicit classifier
            self.imp_inputs = tf.boolean_mask(representation, self.imp_mask, axis=0)
            imp_dense1 = tf.layers.dense(self.imp_inputs, 128, name='imp_dense1', reuse=False)
            imp_dense1_drop = tf.layers.dropout(imp_dense1, rate=0., training=self.trainable)
            imp_dense1_ac = tf.nn.relu(imp_dense1_drop)

            self.imp_dense2 = tf.layers.dense(imp_dense1_ac, FLAGS.classes, name='imp_dense2', reuse=False)
            self.imp_out = tf.nn.softmax(self.imp_dense2, axis=1)
            self.predict = tf.argmax(self.imp_out, axis=1)

            print('real', self.type_labels.shape)
            print('imp_', self.imp_mask)
            self.imp_labels = tf.boolean_mask(self.labels, self.imp_mask, axis=0)

            print('imp_labels', self.imp_labels.shape)
            print('imp_out', self.imp_out.shape)
            self.imp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.imp_labels,
                                                                           logits=self.imp_dense2)

            # final loss
            self.loss = loss = tf.reduce_mean(self.imp_loss) + 0.5 * tf.reduce_mean(
                self.exp_loss) + 0.5 * tf.reduce_mean(self.exp_imp_loss)

        if FLAGS.use_mt:
            # connection classifier
            if FLAGS.arg_encoder == 'bilstm':
                if FLAGS.use_para_info:
                    softmax_w_c = tf.Variable(
                        tf.random_normal([5 * FLAGS.rnn_size, 132], mean=0.0, stddev=0.1, dtype=tf.float32),
                        dtype=tf.float32, name="softmax_w")
                    softmax_b_c = tf.Variable(tf.random_normal([132], mean=0.0, stddev=0.1, dtype=tf.float32),
                                              dtype=tf.float32, name="softmax_b")
                else:
                    softmax_w_c = tf.Variable(
                        tf.random_normal([4 * FLAGS.rnn_size, 132], mean=0.0, stddev=0.1, dtype=tf.float32),
                        dtype=tf.float32, name="softmax_w")
                    softmax_b_c = tf.Variable(tf.random_normal([132], mean=0.0, stddev=0.1, dtype=tf.float32),
                                              dtype=tf.float32, name="softmax_b")
            else:
                softmax_w_c = tf.Variable(
                    tf.random_normal([2 * FLAGS.slstm_size, 132], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="softmax_w")
                softmax_b_c = tf.Variable(tf.random_normal([132], mean=0.0, stddev=0.1, dtype=tf.float32),
                                          dtype=tf.float32, name="softmax_b")

            self.conn_logits = tf.matmul(representation, softmax_w_c) + softmax_b_c
            self.conn_out = tf.nn.softmax(self.conn_logits)
            # operators for prediction
            self.conn_predict = tf.argmax(self.conn_logits, 1)
            # cross entropy loss
            self.conn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.conn_labels,
                                                                            logits=self.conn_logits)
            self.dense2_out_conn = self.conn_logits
            self.conn_cost = cost = tf.reduce_mean(self.conn_loss)
            self.loss += self.conn_cost

        # designate training variables
        tvars = tf.trainable_variables()
        # self.lr = tf.Variable(0.0, trainable=False)
        grads = tf.gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, 5.)
        self.grads = grads
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        self.saver = tf.train.Saver()
        # count model parameters
        def count1():
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print('total_num', total_parameters)

        count1()

    def get_hidden_states_before(self, hidden_states, step, shape, hidden_size):
        # padding zeros
        padding = tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        # remove last steps
        displaced_hidden_states = hidden_states[:, :-step, :]
        # concat padding
        return tf.concat([padding, displaced_hidden_states], axis=1)
        # return tf.cond(step<=shape[1], lambda: tf.concat([padding, displaced_hidden_states], axis=1), lambda: tf.zeros((shape[0], shape[1], self.config.hidden_size_sum), dtype=tf.float32))

    def get_hidden_states_after(self, hidden_states, step, shape, hidden_size):
        # padding zeros
        padding = tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        # remove last steps
        displaced_hidden_states = hidden_states[:, step:, :]
        # concat padding
        return tf.concat([displaced_hidden_states, padding], axis=1)
        # return tf.cond(step<=shape[1], lambda: tf.concat([displaced_hidden_states, padding], axis=1), lambda: tf.zeros((shape[0], shape[1], self.config.hidden_size_sum), dtype=tf.float32))

    def sum_together(self, l):
        combined_state = None
        for tensor in l:
            if combined_state == None:
                combined_state = tensor
            else:
                combined_state = combined_state + tensor
        return combined_state

    def get_D_matrix(self, A):
        indices = []
        d_matrix = tf.reduce_sum(A, axis=2, name="degree_matrix")
        print(d_matrix)
        diag = tf.expand_dims(tf.matrix_diag(tf.pow(d_matrix[0], -0.5)), axis=0)
        for i in range(1, FLAGS.batch_size):
            one_diag = tf.expand_dims(tf.matrix_diag(tf.pow(d_matrix[i], -0.5)), axis=0)
            diag = tf.concat([diag, one_diag], axis=0, name="D_matrix")
        print('diag', diag.shape)
        return diag

    def slstm_gcn_cell(self, name_scope_name, hidden_size, lengths, initial_hidden_states, initial_cell_states,
                       num_layers, A):
        with tf.name_scope(name_scope_name):
            # Word parameters
            # forget gate for left
            with tf.name_scope("f1_gate"):
                # current
                Wxf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wxf")
                # left right
                Whf1 = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whf")
                # initial state
                Wif1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wif")
                # dummy node
                Wdf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wdf")
            # forget gate for right
            with tf.name_scope("f2_gate"):
                Wxf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wxf")
                Whf2 = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whf")
                Wif2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wif")
                Wdf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wdf")
            # forget gate for inital states
            with tf.name_scope("f3_gate"):
                Wxf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wxf")
                Whf3 = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whf")
                Wif3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wif")
                Wdf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wdf")
            # forget gate for dummy states
            with tf.name_scope("f4_gate"):
                Wxf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wxf")
                Whf4 = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whf")
                Wif4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wif")
                Wdf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                   dtype=tf.float32, name="Wdf")
            # input gate for current state
            with tf.name_scope("i_gate"):
                Wxi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wxi")
                Whi = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whi")
                Wii = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wii")
                Wdi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wdi")
            # input gate for output gate
            with tf.name_scope("o_gate"):
                Wxo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wxo")
                Who = tf.Variable(
                    tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Who")
                Wio = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wio")
                Wdo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="Wdo")
            # bias for the gates
            with tf.name_scope("biases"):
                bi = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                 dtype=tf.float32, name="bi")
                bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                 dtype=tf.float32, name="bo")
                bf1 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="bf1")
                bf2 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="bf2")
                bf3 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="bf3")
                bf4 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32, name="bf4")

            # dummy node gated attention parameters
            # input gate for dummy state
            with tf.name_scope("gated_d_gate"):
                gated_Wxd = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Wxf")
                gated_Whd = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Whf")
            # output gate
            with tf.name_scope("gated_o_gate"):
                gated_Wxo = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Wxo")
                gated_Who = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Who")
            # forget gate for states of word
            with tf.name_scope("gated_f_gate"):
                gated_Wxf = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Wxo")
                gated_Whf = tf.Variable(
                    tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name="Who")
            # biases
            with tf.name_scope("gated_biases"):
                gated_bd = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                       dtype=tf.float32, name="bi")
                gated_bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                       dtype=tf.float32, name="bo")
                gated_bf = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                       dtype=tf.float32, name="bo")

        # filters for attention
        mask_softmax_score = tf.cast(tf.sequence_mask(lengths), tf.float32) * 1e25 - 1e25
        mask_softmax_score_expanded = tf.expand_dims(mask_softmax_score, dim=2)
        # filter invalid steps
        sequence_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lengths), tf.float32), axis=2)
        # filter embedding states
        initial_hidden_states = initial_hidden_states * sequence_mask
        initial_cell_states = initial_cell_states * sequence_mask
        # record shape of the batch
        shape = tf.shape(initial_hidden_states)

        # initial embedding states
        embedding_hidden_state = tf.reshape(initial_hidden_states, [-1, hidden_size])
        embedding_cell_state = tf.reshape(initial_cell_states, [-1, hidden_size])

        # randomly initialize the states
        # if config.random_initialize:
        initial_hidden_states = tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None,
                                                  name=None)
        initial_cell_states = tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None,
                                                name=None)
        # filter it
        initial_hidden_states = initial_hidden_states * sequence_mask
        initial_cell_states = initial_cell_states * sequence_mask

        # inital dummy node states
        dummynode_hidden_states = tf.reduce_mean(initial_hidden_states, axis=1)
        dummynode_cell_states = tf.reduce_mean(initial_cell_states, axis=1)

        for i in range(num_layers):
            # update dummy node states
            # average states
            combined_word_hidden_state = tf.reduce_mean(initial_hidden_states, axis=1)
            reshaped_hidden_output = tf.reshape(initial_hidden_states, [-1, hidden_size])
            # copy dummy states for computing forget gate
            transformed_dummynode_hidden_states = tf.reshape(
                tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1], 1]), [-1, hidden_size])
            # input gate
            gated_d_t = tf.nn.sigmoid(
                tf.matmul(dummynode_hidden_states, gated_Wxd) + tf.matmul(combined_word_hidden_state,
                                                                          gated_Whd) + gated_bd
            )
            # output gate
            gated_o_t = tf.nn.sigmoid(
                tf.matmul(dummynode_hidden_states, gated_Wxo) + tf.matmul(combined_word_hidden_state,
                                                                          gated_Who) + gated_bo
            )
            # forget gate for hidden states
            gated_f_t = tf.nn.sigmoid(
                tf.matmul(transformed_dummynode_hidden_states, gated_Wxf) + tf.matmul(reshaped_hidden_output,
                                                                                      gated_Whf) + gated_bf
            )

            # softmax on each hidden dimension
            reshaped_gated_f_t = tf.reshape(gated_f_t, [shape[0], shape[1], hidden_size]) + mask_softmax_score_expanded
            gated_softmax_scores = tf.nn.softmax(
                tf.concat([reshaped_gated_f_t, tf.expand_dims(gated_d_t, dim=1)], axis=1), dim=1)
            # split the softmax scores
            new_reshaped_gated_f_t = gated_softmax_scores[:, :shape[1], :]
            new_gated_d_t = gated_softmax_scores[:, shape[1]:, :]
            # new dummy states
            dummy_c_t = tf.reduce_sum(new_reshaped_gated_f_t * initial_cell_states, axis=1) + tf.squeeze(new_gated_d_t,
                                                                                                         axis=1) * dummynode_cell_states
            dummy_h_t = gated_o_t * tf.nn.tanh(dummy_c_t)

            print('dummy_c_t', dummy_c_t.shape)
            print('dummy_h_t', dummy_h_t.shape)
            dummy_h_t = tf.reshape(dummy_h_t, [FLAGS.batch_size, FLAGS.para_sen_num, -1])
            if FLAGS.gt == 'pag':
                features = dummy_h_t
                self.W = tf.concat([tf.Variable(
                    tf.random_normal([FLAGS.slstm_size, FLAGS.slstm_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32, name='rgcn_w') for _ in range(3)], axis=0)
                self.b = tf.Variable(tf.random_normal([FLAGS.slstm_size], mean=0.0, stddev=0.1, dtype=tf.float32),
                                     dtype=tf.float32, name='rgcn_b')
                for i in range(FLAGS.batch_size):
                    supports = []
                    temp = features[i]
                    for j in range(3):
                        supports.append(tf.matmul(A[i][j], temp))

                    supports = tf.concat(supports, axis=1)
                    output = tf.matmul(supports, self.W)
                    output = tf.expand_dims(output, axis=0)
                    # bias
                    output += self.b
                    if i == 0:
                        outputs = output
                    else:
                        outputs = tf.concat([outputs, output], axis=0)
                self.outputs = tf.nn.relu(outputs)
                dummy_h_t = tf.reshape(self.outputs, [-1, FLAGS.slstm_size])

            elif FLAGS.gt == 'fcg':
                self.X_matrix = dummy_h_t
                # get adjacent matrix
                self.A_matrix = tf.expand_dims(tf.matmul(dummy_h_t[0], tf.transpose(dummy_h_t[0], [1, 0])), 0)
                for i in range(1, FLAGS.batch_size):
                    this_dim = tf.expand_dims(tf.matmul(dummy_h_t[i], tf.transpose(dummy_h_t[i], [1, 0])), 0)
                    self.A_matrix = tf.concat([self.A_matrix, this_dim], 0)
                print(self.A_matrix.shape)
                # degree matrix d, D = d^-1/2
                self.A_matrix = tf.nn.softmax(self.A_matrix)
                self.D_matrix = self.get_D_matrix(self.A_matrix)
                # Normalized matrix Norm_A_matrix = DAD
                self.Norm_A_matrix = tf.matmul(tf.matmul(self.D_matrix, self.A_matrix), self.D_matrix,
                                               name="Norm_A_matrix")
                # one-layer graph convolution
                W1 = tf.Variable(tf.truncated_normal([FLAGS.slstm_size, FLAGS.slstm_size], stddev=0.1),
                                 name='gcn_weights')
                b1 = tf.Variable(tf.zeros([FLAGS.slstm_size]))
                for i in range(FLAGS.batch_size):
                    temp = self.X_matrix[i]
                    pre_sup = tf.matmul(temp, W1)
                    output = tf.matmul(self.Norm_A_matrix[i], pre_sup)
                    output = tf.expand_dims(output, axis=0)
                    # bias
                    output += b1
                    if i == 0:
                        outputs = output
                    else:
                        outputs = tf.concat([outputs, output], axis=0)
                self.outputs = tf.nn.relu(outputs)
                dummy_h_t = tf.reshape(self.outputs, [-1, FLAGS.slstm_size])
            # get states before
            initial_hidden_states_before = [
                tf.reshape(self.get_hidden_states_before(initial_hidden_states, step + 1, shape, hidden_size),
                           [-1, hidden_size]) for step in range(num_step)]
            initial_hidden_states_before = self.sum_together(initial_hidden_states_before)
            initial_hidden_states_after = [
                tf.reshape(self.get_hidden_states_after(initial_hidden_states, step + 1, shape, hidden_size),
                           [-1, hidden_size]) for step in range(num_step)]
            initial_hidden_states_after = self.sum_together(initial_hidden_states_after)
            # get states after
            initial_cell_states_before = [
                tf.reshape(self.get_hidden_states_before(initial_cell_states, step + 1, shape, hidden_size),
                           [-1, hidden_size]) for step in range(num_step)]
            initial_cell_states_before = self.sum_together(initial_cell_states_before)
            initial_cell_states_after = [
                tf.reshape(self.get_hidden_states_after(initial_cell_states, step + 1, shape, hidden_size),
                           [-1, hidden_size]) for step in range(num_step)]
            initial_cell_states_after = self.sum_together(initial_cell_states_after)

            # reshape for matmul
            initial_hidden_states = tf.reshape(initial_hidden_states, [-1, hidden_size])
            initial_cell_states = tf.reshape(initial_cell_states, [-1, hidden_size])

            # concat before and after hidden states
            concat_before_after = tf.concat([initial_hidden_states_before, initial_hidden_states_after], axis=1)

            # copy dummy node states
            transformed_dummynode_hidden_states = tf.reshape(
                tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1], 1]), [-1, hidden_size])
            transformed_dummynode_cell_states = tf.reshape(
                tf.tile(tf.expand_dims(dummynode_cell_states, axis=1), [1, shape[1], 1]), [-1, hidden_size])

            f1_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf1) + tf.matmul(concat_before_after, Whf1) +
                tf.matmul(embedding_hidden_state, Wif1) + tf.matmul(transformed_dummynode_hidden_states, Wdf1) + bf1
            )

            f2_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf2) + tf.matmul(concat_before_after, Whf2) +
                tf.matmul(embedding_hidden_state, Wif2) + tf.matmul(transformed_dummynode_hidden_states, Wdf2) + bf2
            )

            f3_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf3) + tf.matmul(concat_before_after, Whf3) +
                tf.matmul(embedding_hidden_state, Wif3) + tf.matmul(transformed_dummynode_hidden_states, Wdf3) + bf3
            )

            f4_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxf4) + tf.matmul(concat_before_after, Whf4) +
                tf.matmul(embedding_hidden_state, Wif4) + tf.matmul(transformed_dummynode_hidden_states, Wdf4) + bf4
            )

            i_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxi) + tf.matmul(concat_before_after, Whi) +
                tf.matmul(embedding_hidden_state, Wii) + tf.matmul(transformed_dummynode_hidden_states, Wdi) + bi
            )

            o_t = tf.nn.sigmoid(
                tf.matmul(initial_hidden_states, Wxo) + tf.matmul(concat_before_after, Who) +
                tf.matmul(embedding_hidden_state, Wio) + tf.matmul(transformed_dummynode_hidden_states, Wdo) + bo
            )

            f1_t, f2_t, f3_t, f4_t, i_t = tf.expand_dims(f1_t, axis=1), tf.expand_dims(f2_t, axis=1), tf.expand_dims(
                f3_t, axis=1), tf.expand_dims(f4_t, axis=1), tf.expand_dims(i_t, axis=1)

            five_gates = tf.concat([f1_t, f2_t, f3_t, f4_t, i_t], axis=1)
            five_gates = tf.nn.softmax(five_gates, dim=1)
            f1_t, f2_t, f3_t, f4_t, i_t = tf.split(five_gates, num_or_size_splits=5, axis=1)

            f1_t, f2_t, f3_t, f4_t, i_t = tf.squeeze(f1_t, axis=1), tf.squeeze(f2_t, axis=1), tf.squeeze(f3_t,
                                                                                                         axis=1), tf.squeeze(
                f4_t, axis=1), tf.squeeze(i_t, axis=1)

            c_t = (f1_t * initial_cell_states_before) + (f2_t * initial_cell_states_after) + (
                    f3_t * embedding_cell_state) + (f4_t * transformed_dummynode_cell_states) + (
                          i_t * initial_cell_states)

            h_t = o_t * tf.nn.tanh(c_t)

            # update states
            initial_hidden_states = tf.reshape(h_t, [shape[0], shape[1], hidden_size])
            initial_cell_states = tf.reshape(c_t, [shape[0], shape[1], hidden_size])
            initial_hidden_states = initial_hidden_states * sequence_mask
            initial_cell_states = initial_cell_states * sequence_mask

            dummynode_hidden_states = dummy_h_t
            dummynode_cell_states = dummy_c_t

        initial_hidden_states = tf.nn.dropout(initial_hidden_states, rate)
        initial_cell_states = tf.nn.dropout(initial_cell_states, rate)

        return initial_hidden_states, initial_cell_states, dummynode_hidden_states

    def word_embedding(self, inputs, lengths):
        def cell():
            return tf.nn.rnn_cell.GRUCell(128)

        print(inputs.shape)
        print(lengths.shape)
        inputs.set_shape([6 * FLAGS.batch_size, None, 1024 + FLAGS.char_embed_dim + FLAGS.embedding_size])
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.rnn_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.rnn_size)
        outputs, final = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                         sequence_length=lengths, dtype=tf.float32,
                                                         scope='word_embedding')
        return final

    def sen_embedding(self, inputs):
        def cell():
            return tf.nn.rnn_cell.GRUCell(128)

        print('sen_embedding', inputs.shape)
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.rnn_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.rnn_size)
        inputs.set_shape([FLAGS.batch_size, None, 256])
        cell_fw_initial = fw_cell.zero_state(FLAGS.batch_size, tf.float32)
        cell_bw_initial = bw_cell.zero_state(FLAGS.batch_size, tf.float32)
        outputs, final = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                         initial_state_fw=cell_fw_initial,
                                                         initial_state_bw=cell_bw_initial,

                                                         scope='sentence_embedding')
        return final

    def train(self):
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, FLAGS.epochs):
            print('---epoch %d---' % epoch)
            if epoch > 1:
                sess.run(self.lr_decay_op, feed_dict={self.lr_decay_factor: FLAGS.weight_decay})

            min_loss = float("inf")
            pre_counter = 0
            for iteration in range(500):
                if FLAGS.classes == 4:
                    arg1, arg2, arg1_len, arg2_len, char1, char2, arg1_elmo, arg2_elmo, label, pad_para_chars, pad_para_ids, pad_para_elmo, para_seq_len, conn_label, type_label, supports = data.next_multi_rel(
                        FLAGS.batch_size, 'train')
                else:
                    arg1, arg2, arg1_len, arg2_len, char1, char2, arg1_elmo, arg2_elmo, label, pad_para_chars, pad_para_ids, pad_para_elmo, para_seq_len, conn_label, type_label, supports = data.next_single_rel(
                        FLAGS.batch_size, 'train')

                fd = {self.arg1_ids: arg1,
                      self.arg2_ids: arg2,
                      self.labels: label,
                      self.conn_labels: conn_label,
                      self.type_labels: type_label,
                      self.arg1_len: arg1_len,
                      self.arg2_len: arg2_len,
                      self.char1: char1,
                      self.char2: char2,
                      self.arg1_elmo: arg1_elmo.numpy(),
                      self.arg2_elmo: arg2_elmo.numpy(),
                      self.para_ids: pad_para_ids,
                      self.para_chars: pad_para_chars,
                      self.para_elmo: pad_para_elmo.numpy(),
                      self.para_len: para_seq_len,
                      self.trainable: True,
                      self.supports: supports}
                step = sess.run(self.global_step)
                # print(step)
                v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                # gd = sess.run([self.gd_pre, self.gd_pos], feed_dict=fd)
                loss, _, imp_mask, exp_mask = sess.run([self.loss, self.train_op, self.imp_mask, self.exp_mask],
                                                       feed_dict=fd)
                if iteration > 10:
                    # print(loss, min_loss)
                    if loss >= min_loss:
                        pre_counter += 1
                    else:
                        pre_counter = 0
                        min_loss = loss
                if pre_counter >= 20:
                    sess.run(self.lr_decay_op, feed_dict={self.lr_decay_factor: 0.99})
                    pre_counter = 0
                if step % 10 == 0:
                    sess.run(tf.local_variables_initializer())
                    self._eval(epoch, iteration, loss, 'dev')
                    self._eval(epoch, iteration, loss, 'test')

    def eval(self):
        ckpt = tf.train.get_checkpoint_state("model/")
        print(ckpt)
        self.saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
        self._eval(-1, -1, -1, 'test')

    def _eval(self, epoch, iteration, loss, ds):

        if FLAGS.classes == 4:
            selected_samples = data.next_multi_rel(None, ds)
            label = [label for _, _, _, _, _, _, _, _, _, _, _, _, _, _, label in selected_samples]
        else:
            selected_samples = data.next_single_rel(None, ds)
            label = [label for _, _, _, _, _, _, _, _, _, _, _, _, _, _, label in selected_samples]
        label_multi = []
        for one in label:
            label_multi.append(one)
        for i in range(len(selected_samples) // FLAGS.batch_size):
            this_batch = selected_samples[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
            arg1, arg2, arg1_len, arg2_len, char1, char2, arg1_elmo, arg2_elmo, _, pad_para_chars, pad_para_ids, pad_para_elmo, para_seq_len, conn_label, type_label, supports = data._batch2input(
                this_batch)
            fd = {self.arg1_ids: arg1,
                  self.arg2_ids: arg2,
                  self.arg1_len: arg1_len,
                  self.arg2_len: arg2_len,
                  self.arg1_elmo: arg1_elmo.numpy(),
                  self.arg2_elmo: arg2_elmo.numpy(),
                  self.char1: char1,
                  self.char2: char2,
                  self.para_ids: pad_para_ids,
                  self.para_chars: pad_para_chars,
                  self.para_len: para_seq_len,
                  self.para_elmo: pad_para_elmo.numpy(),
                  self.type_labels: type_label,
                  self.trainable: False,
                  self.supports: supports}
            predict, dense2_out, pre_pro, lr = sess.run([self.predict, self.dense2_out, self.out, self.lr],
                                                        feed_dict=fd)
            if FLAGS.use_mt:
                conn_predict = sess.run([self.conn_predict], feed_dict=fd)
                if i == 0:
                    conn_labels = conn_label
                    conn_predicts = conn_predict
                else:
                    conn_labels = np.concatenate([conn_labels, conn_label], axis=0)
                    conn_predicts = np.concatenate([conn_predicts, conn_predict], axis=0)
            if i == 0:
                predicts = predict
                dense2_outs = dense2_out
                pre_proes = pre_pro
            else:
                predicts = np.concatenate([predicts, predict], axis=0)
                dense2_outs = np.concatenate([dense2_outs, dense2_out], axis=0)
                pre_proes = np.concatenate([pre_proes, pre_pro], axis=0)

        this_batch = selected_samples[-FLAGS.batch_size:]
        arg1, arg2, arg1_len, arg2_len, char1, char2, arg1_elmo, arg2_elmo, _, pad_para_chars, pad_para_ids, pad_para_elmo, para_seq_len, conn_label, type_label, supports = data._batch2input(
            this_batch)

        fd = {self.arg1_ids: arg1,
              self.arg2_ids: arg2,
              self.arg1_len: arg1_len,
              self.arg2_len: arg2_len,
              self.arg1_elmo: arg1_elmo.numpy(),
              self.arg2_elmo: arg2_elmo.numpy(),
              self.para_ids: pad_para_ids,
              self.para_chars: pad_para_chars,
              self.para_len: para_seq_len,
              self.para_elmo: pad_para_elmo.numpy(),
              self.char1: char1,
              self.char2: char2,
              self.type_labels: type_label,
              self.trainable: False,
              self.supports: supports}
        predict, dense2_out, pre_pro, lr = sess.run([self.predict, self.dense2_out, self.out, self.lr],
                                                    feed_dict=fd)

        nums = len(selected_samples)

        predicts = np.concatenate([predicts, predict[-nums % FLAGS.batch_size:]], axis=0)
        '''
        if FLAGS.use_mt:
            conn_predict = sess.run([self.conn_predict], feed_dict=fd)

            conn_predicts = np.concatenate([conn_predicts, conn_predict[-nums % FLAGS.batch_size:]], axis=0)
            conn_labels = np.concatenate([conn_labels, conn_label[-nums % FLAGS.batch_size:]],axis=0)
            conn_acc = metrics.accuracy_score(conn_labels, conn_predicts)
            print('conn_acc', conn_acc)
        '''
        dense2_outs = np.concatenate([dense2_outs, dense2_out[-nums % FLAGS.batch_size:]], axis=0)
        pre_proes = np.concatenate([pre_proes, pre_pro[-nums % FLAGS.batch_size:]], axis=0)

        for index in range(len(label)):
            if predicts[index] in label[index]:
                label[index] = predicts[index]
            else:
                label[index] = label[index][0]
        # test_loss = sess.run(tf.losses.sparse_softmax_cross_entropy(label, dense2_outs))

        acc = metrics.accuracy_score(label, predicts)
        if FLAGS.classes == 2:
            f1 = metrics.f1_score(label, predicts, pos_label=1, average='binary')
            print('epoch:%d iter_num:%d train_loss:%.4f' % (
            epoch, iteration, loss) + ds + ' acc:%.4f f1:%.4f lr: %.4f' % (acc, f1, lr))
        if FLAGS.classes == 4:
            f1 = metrics.f1_score(label, predicts, average='macro')
            print('epoch:%d iter_num:%d train_loss:%.4f ' % (
            epoch, iteration, loss) + ds + '  acc:%.4f f1:%.4f lr: %.4f' % (acc, f1, lr))


if __name__ == '__main__':

    start_time = time.time()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    local_device_protos = device_lib.list_local_devices()
    [print(x) for x in local_device_protos if x.device_type == 'GPU']
    rel = FLAGS.pos_class
    print(FLAGS.classes, '-class')
    if FLAGS.classes == 2:
        print('pos_class:', rel)
    data = pddata(FLAGS, rel)
    if FLAGS.classes == 4:
        data.gen_whole_data()
    else:
        data.gen_rel_data(rel)
    sess = tf.Session(config=config)
    model = CTNET(FLAGS, data.embedding)
    model.train()
    print("time:%.1f(minute)" % ((time.time() - start_time) / 60))
