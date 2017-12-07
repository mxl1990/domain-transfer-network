# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim

#import model file
from model3 import DCGAN
from ops import *
from utils import *


class DTN(object):
    """Domain Transfer Network
    """
    def __init__(self, mode='train', learning_rate=0.0003, 
                    height=96, width=None, batch_size=64,
                    Alpha=0.0, Beta=0.0, Gama=0.0):
        self.mode = mode
        self.learning_rate = learning_rate
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.Alpha = Alpha
        self.Beta = Beta
        self.Gama = Gama
        
                
    def generator(self, inputs, reuse=False):
        # inputs: (batch, 1, 1, 128)
        with tf.variable_scope('Generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,           
                                 stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                     activation_fn=tf.nn.relu, is_training=(self.mode=='train')):

                    net = slim.conv2d_transpose(inputs, 64*4, [5, 5], scope='conv_transpose1')   # (batch_size, 4, 4, 512)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d_transpose(net, 64*2, [5, 5], scope='conv_transpose2')  # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d_transpose(net, 64, [5, 5], scope='conv_transpose3')  # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d_transpose(net, 3, [5, 5], scope='conv_transpose4')   # (batch_size, 32, 32, 1)
                    return net
    
    def discriminator(self, images, reuse=False):
        # images: (batch, 32, 32, 1)
        with tf.variable_scope('Discriminator', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=lrelu,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                # with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                #                     # activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
                #                     activation_fn=lrelu, is_training=(self.mode=='train')):
                    
                    net = slim.conv2d(images, 128, [3, 3], scope='conv1')   # (batch_size, 16, 16, 128)
                    # net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv2')   # (batch_size, 8, 8, 256)
                    # net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 512, [3, 3], scope='conv3')   # (batch_size, 4, 4, 512)
                    # net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, 1, [4, 4], padding='VALID', scope='conv4')   # (batch_size, 1, 1, 1)
                    # net = slim.flatten(net)
                    net = linear(tf.reshape(net, [self.batch_size, -1]), 3, 'final')
                    # net = tf.nn.sigmoid(net)
                    # D1, D2, D3 = tf.reshape(net[:,0],[-1, 1]), tf.reshape(net[:,1],[-1, 1]), tf.reshape(net[:,2],[-1, 1])
                    # return D1, D2, D3
                    return tf.unstack(net, axis=1)

    def TV_loss(self, images):
        _, h, w, _ = images.shape
        y = images[:,0:h-1,:,:] - images[:,1:,:,:]
        x = images[:,:,0:w-1,:] - images[:,:,1:,:]
        # y = tf.slice(images, [0, 0, 0, 0], [-1, h_, -1, -1]) - tf.slice(images, [0, 1, 0, 0], [-1, -1, -1, -1])
        # x = tf.slice(images, [0, 0, 0, 0], [-1, -1, w - 1, -1]) - tf.slice(images, [0, 0, 1, 0], [-1, -1, -1, -1])
        loss = tf.sqrt( tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y)) )
        return loss
        # return 0.0

                
    def build_model(self, sess):
        
        if self.mode == 'pretrain':
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            self.labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
            
            # logits and accuracy
            self.logits = self.content_extractor(self.images)
            self.pred = tf.argmax(self.logits, 1)
            self.correct_pred = tf.equal(self.pred, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            # loss and train op
            self.loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
            
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])

        elif self.mode == 'eval':
            h, w = self.height, self.width
            self.images = tf.placeholder(tf.float32, [self.batch_size, h, w, 3], 'source_images')

            # source domain (svhn to mnist)
            dcgan = DCGAN(sess, input_height=h, input_width=w, is_crop=False,
                 batch_size=self.batch_size, sample_num = 64, output_height=h, output_width=w)

            _, _, self.fx = dcgan.discriminator(self.images, reuse=True)
            self.sampled_images = self.generator(self.fx)

        elif self.mode == 'train':
            h, w = self.height, self.width
            batch_size = self.batch_size

            self.src_images = tf.placeholder(tf.float32, [batch_size, h, w, 3], 'source_images')
            self.trg_images = tf.placeholder(tf.float32, [batch_size, h, w, 3], 'target_images')
            dcgan = DCGAN(sess, input_height=h, input_width=w, is_crop=False,
                 batch_size=batch_size, sample_num =64, output_height=h, output_width=w, load_model=True)
            
            # source domain (svhn to mnist)
            # self.fx = self.content_extractor(self.src_images)
            _, _, self.fx_src = dcgan.discriminator(self.src_images, reuse=True)

            # self.random_fx = tf.random_shuffle(self.fx)
            shape = self.fx_src.get_shape().as_list()
            self.g_input_src = tf.placeholder(dtype=tf.float32, shape=shape, name='g_input_src')
            self.fake_images = self.generator(self.g_input_src)
            
            # self.fake_images = self.generator(self.fx_src)

            self.g_tv_loss_src = self.TV_loss(self.fake_images)
            print("get source tv loss")
            self.logits1, self.logits2, self.logits3 = self.discriminator(self.fake_images)

            # self.src_logits1, self.src_logits2, self.src_logits3 = self.discriminator(self.src_images)
            _, _, self.fgfx = dcgan.discriminator(self.fake_images, reuse=True)

            # loss
            # self.d_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.zeros_like(self.logits))
            # self.g_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.ones_like(self.logits))
            # self.d_loss_src = tf.reduce_mean(-tf.log(self.logits1))
            # self.g_loss_src = tf.reduce_mean(-tf.log(self.logits3))
            
            # self.d_loss_src = tf.reduce_mean(
            #                     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits1, labels=tf.ones_like(self.logits1)))
            self.d_loss_src = tf.losses.sigmoid_cross_entropy(logits=self.logits1, multi_class_labels=tf.ones_like(self.logits1))
            # self.g_loss_src = tf.reduce_mean(
            #                     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits3, labels=tf.ones_like(self.logits3)))

            # self.d_loss_src = -tf.reduce_mean(self.logits1)
            # self.g_loss_src = tf.reduce_mean(-self.logits3)

            # self.d_loss_src = tf.losses.sigmoid_cross_entropy(logits=self.logits1, multi_class_labels=tf.zeros_like(self.logits1))
            # self.g_loss_src = tf.losses.sigmoid_cross_entropy(logits=self.logits3, multi_class_labels=tf.ones_like(self.logits3)) + self.Gama * self.g_tv_loss_src



            # self.fx_input = tf.placeholder(dtype=tf.float32,shape=shape)
            # self.f_loss_src = tf.reduce_mean(tf.square(self.fx_input - self.fgfx)) * self.Alpha
            self.f_loss_src = tf.sqrt(tf.reduce_mean(tf.square(self.fx_src-self.fgfx))) * self.Alpha

            self.g_loss_src = -tf.reduce_mean(self.logits3) + self.Gama * self.g_tv_loss_src + self.f_loss_src

            
            # optimizer
            # self.d_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            # self.g_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            # self.f_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'Discriminator' in var.name]
            g_vars = [var for var in t_vars if 'Generator' in var.name]
            # f_vars = [var for var in t_vars if 'discriminator' in var.name]

            
            # train op
            # with tf.name_scope('source_train_op'):
            # with tf.variable_scope('source_train_op', reuse=False):
            #     self.d_train_op_src = slim.learning.create_train_op(self.d_loss_src, self.d_optimizer_src, variables_to_train=d_vars)
            #     self.g_train_op_src = slim.learning.create_train_op(self.g_loss_src, self.g_optimizer_src, variables_to_train=g_vars)
            #     self.f_train_op_src = slim.learning.create_train_op(self.f_loss_src, self.f_optimizer_src, variables_to_train=f_vars)
            
            # # summary op
            # d_loss_src_summary = tf.summary.scalar('src_d_loss', self.d_loss_src)
            # g_loss_src_summary = tf.summary.scalar('src_g_loss', self.g_loss_src)
            # f_loss_src_summary = tf.summary.scalar('src_f_loss', self.f_loss_src)
            # origin_images_summary = tf.summary.image('src_origin_images', self.src_images)
            # sampled_images_summary = tf.summary.image('src_sampled_images', self.fake_images)
            # self.summary_op_src = tf.summary.merge([d_loss_src_summary, g_loss_src_summary, 
            #                                         f_loss_src_summary, origin_images_summary, 
            #                                         sampled_images_summary])
            
            # target domain (mnist)
            # self.fx = self.content_extractor(self.trg_images, reuse=True)
            _, _, self.fx_trg = dcgan.discriminator(self.trg_images, reuse=True)
            # self.random_fx = tf.random_shuffle(self.fx)

            self.g_input_trg = tf.placeholder(dtype=tf.float32,shape=shape)
            self.reconst_images = self.generator(self.g_input_trg, reuse=True)
            
            # self.reconst_images = self.generator(self.fx_trg,reuse=True)
            self.logits1_fake, self.logits2_fake, self.logits3_fake = self.discriminator(self.reconst_images, reuse=True)
            self.logits1_real, self.logits2_real, self.logits3_real = self.discriminator(self.trg_images, reuse=True)
            
            # loss
            # self.d_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_fake))
            # self.d_loss_real_trg = slim.losses.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real))

            self.d_loss_fake_trg = tf.losses.sigmoid_cross_entropy(logits=self.logits2_fake, multi_class_labels=tf.zeros_like(self.logits2_fake))
            # self.d_loss_real_trg = tf.losses.sigmoid_cross_entropy(logits=self.logits_real, multi_class_labels=tf.ones_like(self.logits_real))
            
            # self.d_loss_trg = self.d_loss_fake_trg + self.d_loss_real_trg
            # self.d_loss_fake_trg = tf.reduce_mean(
            #                         tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits2_fake, 
            #                         labels=tf.ones_like(self.logits2_fake)))
            # self.d_loss_real_trg = tf.reduce_mean(
            #                         tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits3_real, 
            #                         labels=tf.ones_like(self.logits3_real)))
            # self.d_loss_fake_trg = tf.reduce_mean(-self.logits2_fake-self.logits3_real)
            # self.d_loss_trg = self.d_loss_fake_trg # + self.d_loss_real_trg
            self.d_loss_trg =self.d_loss_fake_trg + tf.reduce_mean(self.logits3-self.logits3_real)

            lam = 10.
            # alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
            # alpha = alpha_dist.sample((batch_size, 1, 1, 1))
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)

            # interpolated = real_data + alpha*(train-real_data)  
            interpolated = self.trg_images + alpha*(self.fake_images-self.trg_images)                                 
            inte_logit = self.discriminator(interpolated, reuse=True)
            gradients = tf.gradients(inte_logit, interpolated)[0]
            grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
            gradient_penalty = tf.reduce_mean((grad_l2-1.0)**2)
            # gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
            # grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
            self.d_loss_trg += lam*gradient_penalty


            self.g_tv_loss_trg = self.TV_loss(self.reconst_images)
            print("get target tv loss")

            # self.g_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.ones_like(self.logits_fake))

            # self.g_loss_fake_trg = tf.losses.sigmoid_cross_entropy(logits=self.logits_fake, multi_class_labels=tf.ones_like(self.logits_fake))
            # self.g_loss_const_trg = tf.reduce_mean(tf.square(self.trg_images - self.reconst_images)) * self.Beta
            # self.g_loss_trg = self.g_loss_fake_trg + self.g_loss_const_trg + self.Gama * self.g_tv_loss_trg
            
            # self.g_loss_fake_trg = tf.reduce_mean(-tf.log(self.logits3_fake))
            # self.g_loss_fake_trg = tf.reduce_mean(
            #                         tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits3_fake, labels=tf.ones_like(self.logits3_fake)))
            # self.g_loss_fake_trg = tf.reduce_mean(-self.logits3_fake)
            # self.g_loss_const_trg = tf.reduce_mean(tf.square(self.trg_images - self.reconst_images)) * self.Beta
            # self.g_loss_trg = self.g_loss_fake_trg + self.g_loss_const_trg + self.Gama * self.g_tv_loss_trg
            self.g_loss_trg = -tf.reduce_mean(self.logits3_fake) + tf.sqrt(tf.reduce_mean(tf.square(self.trg_images - self.reconst_images))) * self.Beta + self.Gama * self.g_tv_loss_trg
            
            # optimizer
            # self.d_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            # self.g_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0., beta2=0.9)
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0., beta2=0.9)

            self.d_loss = self.d_loss_src + self.d_loss_trg
            self.g_loss = self.g_loss_src + self.g_loss_trg


            # train op
            # with tf.name_scope('target_train_op'):
            with tf.variable_scope('train_op', reuse=False):
                self.d_train_op = slim.learning.create_train_op(self.d_loss, self.d_optimizer, variables_to_train=d_vars, colocate_gradients_with_ops=True)
                self.g_train_op = slim.learning.create_train_op(self.g_loss, self.g_optimizer, variables_to_train=g_vars, colocate_gradients_with_ops=True)


            # clipped_var_c = [tf.assign(var, tf.clip_by_value(var, -1.0, 1.0)) for var in d_vars]
            # # merge the clip operations on critic variables
            # with tf.control_dependencies([self.d_optimizer]):
            #     opt_c = tf.tuple(clipped_var_c)
            # self.clip = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in d_vars]
            
            # summary op
            # d_loss_fake_trg_summary = tf.summary.scalar('trg_d_loss_fake', self.d_loss_fake_trg)
            # d_loss_real_trg_summary = tf.summary.scalar('trg_d_loss_real', self.d_loss_real_trg)
            # d_loss_trg_summary = tf.summary.scalar('trg_d_loss', self.d_loss_trg)
            # g_loss_fake_trg_summary = tf.summary.scalar('trg_g_loss_fake', self.g_loss_fake_trg)
            # g_loss_const_trg_summary = tf.summary.scalar('trg_g_loss_const', self.g_loss_const_trg)
            # g_loss_trg_summary = tf.summary.scalar('trg_g_loss', self.g_loss_trg)
            # origin_images_summary = tf.summary.image('trg_origin_images', self.trg_images)
            # sampled_images_summary = tf.summary.image('trg_reconstructed_images', self.reconst_images)
            # self.summary_op_trg = tf.summary.merge([d_loss_trg_summary, g_loss_trg_summary, 
            #                                         # d_loss_fake_trg_summary, d_loss_real_trg_summary,
            #                                         g_loss_fake_trg_summary, g_loss_const_trg_summary,
            #                                         origin_images_summary, sampled_images_summary])
            # for var in tf.trainable_variables():
            #     tf.summary.histogram(var.op.name, var)
            