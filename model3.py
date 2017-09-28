# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
	# size/stride 得出有多少行？
	return int(math.ceil(float(size) / float(stride)))

# 主要的类，用以实现DCGAN
class DCGAN(object):
	def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
				 batch_size=64, sample_num = 64, output_height=64, output_width=64,
				 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
				 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
				 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
		"""
		Args:
			sess: TensorFlow session
			batch_size: 批处理的大小
			y_dim: (optional) Dimension of dim for y. [None]
			z_dim: (optional) 噪声数据的维度. [100]
			gf_dim: (optional) gen filter第一层卷积层的卷积核个数. [64]
			df_dim: (optional) discrim filter第一层卷积层的卷积核个数. [64]
			gfc_dim: (optional) gen全连接层的个数. [1024]
			dfc_dim: (optional) discrim全连接层的个数. [1024]
			c_dim: (optional) 图片颜色的维度. 灰度图可以设置为1. [3]
		"""
		self.sess = sess
		self.is_crop = is_crop
		self.is_grayscale = (c_dim == 1)

		self.batch_size = batch_size
		self.sample_num = sample_num

		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width

		self.y_dim = y_dim
		self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.gfc_dim = gfc_dim
		self.dfc_dim = dfc_dim

		self.c_dim = c_dim

		# batch normalization : deals with poor initialization helps gradient flow
		# batch normalization的实现部分在ops.py中
		# 如下部分都是先初始化一个指定名字的batch_norm对象
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')

		if not self.y_dim:
			self.d_bn3 = batch_norm(name='d_bn3')

		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')

		if not self.y_dim:
			self.g_bn3 = batch_norm(name='g_bn3')

		self.dataset_name = dataset_name
		self.input_fname_pattern = input_fname_pattern
		self.checkpoint_dir = checkpoint_dir
		# 构建模型
		self.build_model()

	def build_model(self):
		'''
		建立模型
		'''
		print("in model")
		if self.y_dim:
			self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

		# is_crop影响image_dims维度的确定
		if self.is_crop:
			image_dims = [self.output_height, self.output_width, self.c_dim]
		else:
			image_dims = [self.input_height, self.input_width, self.c_dim]

		# 输入的维度是batch_size*input_height*input_width*c_dim
		# = batch_size * 图像大小
		# inputs为真实图像输入
		self.inputs = tf.placeholder(
			tf.float32, [self.batch_size] + image_dims, name='real_images')
		# sample_inputs为生成的样本的输入
		self.sample_inputs = tf.placeholder(
			tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

		inputs = self.inputs
		sample_inputs = self.sample_inputs

		# z是用以生成数据的噪声数据
		self.z = tf.placeholder(
			tf.float32, [None, self.z_dim], name='z')
		self.z_sum = histogram_summary("z", self.z)

		# 
		if self.y_dim:
			self.G = self.generator(self.z, self.y)
			self.D, self.D_logits, _ = \
					self.discriminator(inputs, self.y, reuse=False)

			self.sampler = self.sampler(self.z, self.y)
			self.D_, self.D_logits_, _ = \
					self.discriminator(self.G, self.y, reuse=True)
		else:
			# 这里可以看到用z生成G
			self.G = self.generator(self.z)
			# 用discrim判别输入图像的真假
			self.D, self.D_logits, _ = self.discriminator(inputs)

			self.sampler = self.sampler(self.z)
			# 用discrim判别生成图像的真假
			self.D_, self.D_logits_, _ = self.discriminator(self.G, reuse=True)

		self.d_sum = histogram_summary("d", self.D)
		self.d__sum = histogram_summary("d_", self.D_)
		self.G_sum = image_summary("G", self.G)

		def sigmoid_cross_entropy_with_logits(x, y):
			# 应该是尝试兼容不同版本的此函数
			try:
				return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
			except:
				return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

		# D的loss将真的判断为假的部分
		self.d_loss_real = tf.reduce_mean(
			sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))

		# D的loss将假的判断为真的部分
		self.d_loss_fake = tf.reduce_mean(
			sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))


		# G的loss，就是将其判断为真的部分，即论文中1-D(G(z))
		self.g_loss = tf.reduce_mean(
			sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

		self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
													
		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		# discrim的变量
		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		# gen的变量
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		# 用于保存模型结果
		self.saver = tf.train.Saver()

	def train(self, config):
		print("in train")
		"""
		训练DCGAN
		"""

		# 定义D和G的优化器为Adam优化器
		d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
							.minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
							.minimize(self.g_loss, var_list=self.g_vars)
		# 初始化变量
		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()
		

		self.g_sum = merge_summary([self.z_sum, self.d__sum,
			self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
		self.d_sum = merge_summary(
				[self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
		self.writer = SummaryWriter(config.log_dir, self.sess.graph)

		
		# 每次开始之前检查是否存在已经训练的模型
		counter = 0
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		if config.dataset == "mnist":
			return self.train_mnist(config, d_optim, g_optim, counter)

		# 导入数据，这里可以看到默认数据路径为./data/config.dataset位置
		# glob查找指定目录下的文件
		data = glob(os.path.join(config.dataset_dir, self.input_fname_pattern))
		file_queue = tf.train.string_input_producer(data, shuffle=False,capacity=len(data))
		batch_image = get_batch_image(file_queue,
									batch_size = config.batch_size,
									input_height=self.input_height,
									input_width=self.input_width,
									resize_height=self.output_height,
									resize_width=self.output_width,
									is_crop=self.is_crop,
									is_grayscale=self.is_grayscale
									)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

		#np.random.shuffle(data)
		# 取前sample_num个样本
		sample_files = data[0:self.sample_num] # 这里sample仅用于每次训练完成后生成样本查看loss所用
		# 获取图片数据
		sample = [
				get_image(sample_file,
									input_height=self.input_height,
									input_width=self.input_width,
									resize_height=self.output_height,
									resize_width=self.output_width,
									is_crop=self.is_crop,
									is_grayscale=self.is_grayscale) for sample_file in sample_files]
		# 这里创建numpy数组(矩阵)
		if (self.is_grayscale):
			sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
		else:
			sample_inputs = np.array(sample).astype(np.float32)
		# 随机生成用以生成样本的噪声z
		sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

		start_time = time.time()
		batch_idxs = min(len(data), config.train_size) // config.batch_size
		
		for epoch in xrange(config.epoch):
			# # 总数据量/每次训练量=训练次数index    
			# data = glob(os.path.join(
			# 		config.dataset_dir, self.input_fname_pattern))
			# batch_idxs = min(len(data), config.train_size) // config.batch_size

			for idx in xrange(0, batch_idxs):
				counter = counter + 1
				# print("get batch file")
				batch_images = self.sess.run(batch_image)
				# self.inputs.assign(image_batch)
				# self.input = image_batch
				# print("finish get batch file")
				# batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
				# batch = [
				# 		get_image(batch_file,
				# 							input_height=self.input_height,
				# 							input_width=self.input_width,
				# 							resize_height=self.output_height,
				# 							resize_width=self.output_width,
				# 							is_crop=self.is_crop,
				# 							is_grayscale=self.is_grayscale) for batch_file in batch_files]
				# if (self.is_grayscale):
				# 	batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
				# else:
				# 	batch_images = np.array(batch).astype(np.float32)

				batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
							.astype(np.float32)

				# Update D network
				# 更新使用一般数据集的Discrim
				_, summary_str = self.sess.run([d_optim, self.d_sum],
					feed_dict={ self.inputs: batch_images, self.z: batch_z })
					# feed_dict={ self.z: batch_z })
				self.writer.add_summary(summary_str, counter)

				# Update G network
				# 更新使用一般数据集的Gen
				_, summary_str = self.sess.run([g_optim, self.g_sum],
					feed_dict={ self.z: batch_z })
				self.writer.add_summary(summary_str, counter)

				# 与论文中不同的地方
				# 一批次中运行两次g_optim
				_, summary_str = self.sess.run([g_optim, self.g_sum],
					feed_dict={ self.z: batch_z })
				self.writer.add_summary(summary_str, counter)
					
				errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
				errD_real = self.d_loss_real.eval({self.inputs:batch_images})
				errG = self.g_loss.eval({self.z: batch_z})

				# 输出本次批次的训练信息
				# 可屏蔽以加速
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
					% (epoch, idx, batch_idxs,
						time.time() - start_time, errD_fake+errD_real, errG))
				
			# 每个epoch后对生成的图片进行保存
			try:
				samples, d_loss, g_loss = self.sess.run(
					[self.sampler, self.d_loss, self.g_loss],
					feed_dict={
								self.z: sample_z,
								self.inputs: sample_inputs,
							},
						)	
				manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
				manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
				save_images(samples, [manifold_h, manifold_w],
							'{}/train_{:02d}.png'.format(config.sample_dir, idx))
				print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
			except:
				print("one pic error!...")

			# 每间隔500epoch保存一次当前模型
			# 从第2个epoch开始
			if np.mod(epoch, 500) == 2:
				self.save(config.checkpoint_dir, counter)


	# 分离出训练mnist数据集不一致的代码
	# 让训练代码更易读
	def train_mnist(self, config, d_optim, g_optim, counter):
		data_X, data_y = self.load_mnist()
		

		# 用于每次生成图片时记录模型loss的样本数据
		sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
		sample_inputs = data_X[0:self.sample_num]
		sample_labels = data_y[0:self.sample_num]

		start_time = time.time()
		for epoch in xrange(config.epoch):
			batch_idxs = min(len(data_X), config.train_size) // config.batch_size

			for idx in xrange(0, batch_idxs):
				counter = counter + 1
				batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
				batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]

				batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
							.astype(np.float32)

				_, summary_str = self.sess.run([d_optim, self.d_sum],
					feed_dict={ 
						self.inputs: batch_images,
						self.z: batch_z,
						self.y:batch_labels,
					})
				self.writer.add_summary(summary_str, counter)

				# Update G network
				_, summary_str = self.sess.run([g_optim, self.g_sum],
					feed_dict={
						self.z: batch_z, 
						self.y:batch_labels,
					})
				self.writer.add_summary(summary_str, counter)

				# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
				_, summary_str = self.sess.run([g_optim, self.g_sum],
					feed_dict={ self.z: batch_z, self.y:batch_labels })
				self.writer.add_summary(summary_str, counter)
				 
				errD_fake = self.d_loss_fake.eval({
						self.z: batch_z, 
						self.y:batch_labels
				})
				errD_real = self.d_loss_real.eval({
						self.inputs: batch_images,
						self.y:batch_labels
				})
				errG = self.g_loss.eval({
						self.z: batch_z,
						self.y: batch_labels
				})

				# 输出本次批次的训练信息
				# 可屏蔽以加速
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
					% (epoch, idx, batch_idxs,
						time.time() - start_time, errD_fake+errD_real, errG))

			# 每个epoch结束后，保存一次图片
			try:
				samples, d_loss, g_loss = self.sess.run(
						[self.sampler, self.d_loss, self.g_loss],
						feed_dict={
								self.z: sample_z,
								self.inputs: sample_inputs,
								self.y:sample_labels,
						}
					)
				manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
				manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
				save_images(samples, [manifold_h, manifold_w],
							'{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
				print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
			except:
				print("one pic error!...")

			# 每个epoch保存一次模型
			self.save(config.checkpoint_dir, epoch)



	def discriminator(self, image, y=None, reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()

			if not self.y_dim:
				h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
				h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
				h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
				h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
				h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

				return tf.nn.sigmoid(h4), h4, h3
			else:
				yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
				x = conv_cond_concat(image, yb)

				h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
				h0 = conv_cond_concat(h0, yb)

				h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
				h1 = tf.reshape(h1, [self.batch_size, -1])      
				h1 = concat([h1, y], 1)
				
				h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
				features = h2
				h2 = concat([h2, y], 1)

				h3 = linear(h2, 1, 'd_h3_lin')
				
				return tf.nn.sigmoid(h3), h3, features

	def generator(self, z, y=None):
		with tf.variable_scope("generator") as scope:
			if not self.y_dim:
				s_h, s_w = self.output_height, self.output_width
				s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
				s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
				s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
				s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

				# project `z` and reshape
				self.z_, self.h0_w, self.h0_b = linear(
						z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

				self.h0 = tf.reshape(
						self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
				h0 = tf.nn.relu(self.g_bn0(self.h0))

				self.h1, self.h1_w, self.h1_b = deconv2d(
						h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
				h1 = tf.nn.relu(self.g_bn1(self.h1))

				h2, self.h2_w, self.h2_b = deconv2d(
						h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
				h2 = tf.nn.relu(self.g_bn2(h2))

				h3, self.h3_w, self.h3_b = deconv2d(
						h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
				h3 = tf.nn.relu(self.g_bn3(h3))

				h4, self.h4_w, self.h4_b = deconv2d(
						h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

				return tf.nn.tanh(h4)
			else:
				s_h, s_w = self.output_height, self.output_width
				s_h2, s_h4 = int(s_h/2), int(s_h/4)
				s_w2, s_w4 = int(s_w/2), int(s_w/4)

				# yb = tf.expand_dims(tf.expand_dims(y, 1),2)
				yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
				z = concat([z, y], 1)

				h0 = tf.nn.relu(
						self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
				h0 = concat([h0, y], 1)

				h1 = tf.nn.relu(self.g_bn1(
						linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
				h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

				h1 = conv_cond_concat(h1, yb)

				h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
						[self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
				h2 = conv_cond_concat(h2, yb)

				return tf.nn.sigmoid(
						deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

	def sampler(self, z, y=None):
		with tf.variable_scope("generator") as scope:
			scope.reuse_variables()

			if not self.y_dim:
				s_h, s_w = self.output_height, self.output_width
				s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
				s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
				s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
				s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

				# project `z` and reshape
				h0 = tf.reshape(
						linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
						[-1, s_h16, s_w16, self.gf_dim * 8])
				h0 = tf.nn.relu(self.g_bn0(h0, train=False))

				h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
				h1 = tf.nn.relu(self.g_bn1(h1, train=False))

				h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
				h2 = tf.nn.relu(self.g_bn2(h2, train=False))

				h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
				h3 = tf.nn.relu(self.g_bn3(h3, train=False))

				h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

				return tf.nn.tanh(h4)
			else:
				s_h, s_w = self.output_height, self.output_width
				s_h2, s_h4 = int(s_h/2), int(s_h/4)
				s_w2, s_w4 = int(s_w/2), int(s_w/4)

				# yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
				yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
				z = concat([z, y], 1)

				h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
				h0 = concat([h0, y], 1)

				h1 = tf.nn.relu(self.g_bn1(
						linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
				h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
				h1 = conv_cond_concat(h1, yb)

				h2 = tf.nn.relu(self.g_bn2(
						deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
				h2 = conv_cond_concat(h2, yb)

				return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

	def load_mnist(self):
		data_dir = os.path.join("./data", self.dataset_name)
		
		fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

		fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		trY = loaded[8:].reshape((60000)).astype(np.float)

		fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

		fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
		loaded = np.fromfile(file=fd,dtype=np.uint8)
		teY = loaded[8:].reshape((10000)).astype(np.float)

		trY = np.asarray(trY)
		teY = np.asarray(teY)
		
		X = np.concatenate((trX, teX), axis=0)
		y = np.concatenate((trY, teY), axis=0).astype(np.int)
		
		seed = 547
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)
		
		y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
		for i, label in enumerate(y):
			y_vec[i,y[i]] = 1.0
		
		return X/255.,y_vec

	@property
	def model_dir(self):
		return "{}_{}_{}_{}".format(
				self.dataset_name, self.batch_size,
				self.output_height, self.output_width)
			
	def save(self, checkpoint_dir, step):
		model_name = "DCGAN.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir, path_parse=True):
		import re
		print(" [*] Reading checkpoints...")
		if path_parse:
			checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print("result is ",ckpt)
			print(" [*] Failed to find a checkpoint")
			return False, 0

	def extract_features(self, config):
		import os
		# import json
		# import matio
		# 先导入已经训练好的模型
		could_load, checkpoint_counter = self.load(self.checkpoint_dir, path_parse=False)
		if could_load:
			counter = checkpoint_counter
			print(" [*] Load SUCCESS")
		else:
			raise Exception(" [!] Load failed, please Train First!")

		# if not config.img_file or not config.dataset_dir:
		# 	raise Exception("Have no file to get image")

		# with open(config.img_file) as fp:
		# 	imglist = json.load(fp)['path']


		# # 导入数据
		# if config.dataset == 'mnist':
		# 	data_X, data_y = self.load_mnist()
		# else:
		# 	data = glob(os.path.join(".\\data", config.extr_dir, self.input_fname_pattern))

		# 单张处理
		# # 导入数据
		if config.dataset == 'mnist':
			data_X, data_y = self.load_mnist()
		else:
			data = glob(os.path.join("./data", config.extr_dir, self.input_fname_pattern))

		print("begin to get image data")

		print("finish image get")
		if self.is_crop:
			image_dims = [self.output_height, self.output_width, self.c_dim]
		else:
			image_dims = [self.input_height, self.input_width, self.c_dim]
		self.batch_size = 1

		# image_pos = tf.placeholder(tf.float32, image_dims)

		for datum in data:
			sample = [get_image(datum,
										input_height=self.input_height,
										input_width=self.input_width,
										resize_height=self.output_height,
										resize_width=self.output_width,
										is_crop=self.is_crop,
										is_grayscale=self.is_grayscale)]

			if (self.is_grayscale):
				image = np.array(sample).astype(np.float32)[:, :, :, None]
			else:
				image = np.array(sample).astype(np.float32)


			print("current image is", image.shape )
			image = tf.constant(image)
			
			if self.y_dim:
				_, _, feature = self.discriminator(image, self.y, reuse=True)
			else:
				_, _, feature  = self.discriminator(image,reuse=True)

			# self.sess.run(feature, feed_dict={image_pos:image})

			# 输出特征信息
			print("feature is", self.sess.run(feature))




