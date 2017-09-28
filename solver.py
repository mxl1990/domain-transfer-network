import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc

from glob import glob


class Solver(object):

    def __init__(self, sess, model, batch_size=64, pretrain_iter=20000, train_iter=2000, sample_iter=100, 
                 source_dir='source', target_dir='target', log_dir='logs', sample_save_path='sample', 
                 model_save_path='model', 
                 pretrained_model='D:\\workspace\\tensorflow-101.git\\DTN\\trunk\\checkpoint\\celebA_64_96_96\\DCGAN.model-18990', 
                 test_model='model/dtn-2000'):
        
        self.sess = sess
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        self.config.operation_timeout_in_ms = 1000000



    # def load_source(self, image_dir):
    #   return self._load_images(image_dir)

    # def load_target(self, image_dir):
    #   return self._load_images(image_dir)

    def load_images(self, image_dir):
        import os
        paths = glob(os.path.join(image_dir, "*.jpg"))
        file_queue = tf.train.string_input_producer(paths, capacity=len(paths))
        images = []
        num_threads = 8
        for _ in range(num_threads):
            reader = tf.WholeFileReader()
            key, image_content = reader.read(file_queue)
            image = tf.image.decode_image(image_content)
            image = tf.cast(image, tf.float32)
            image = image / 127.5 - 1
            images.append([image])

        return tf.train.batch_join(
                    images,
                    batch_size=self.batch_size,
                    shapes=[96, 96, 3],
                    enqueue_many=False,
                    capacity=4*num_threads*self.batch_size,
                    allow_smaller_final_batch=True,
            )


    def load_svhn(self, image_dir, split='train'):
        print ('loading svhn image dataset..')
        
        if self.model.mode == 'pretrain':
            image_file = 'extra_32x32.mat' if split=='train' else 'test_32x32.mat'
        else:
            image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'
            
        image_dir = os.path.join(image_dir, image_file)
        svhn = scipy.io.loadmat(image_dir)
        images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels==10)] = 0
        print ('finished loading svhn image dataset..!')
        return images, labels

    def load_mnist(self, image_dir, split='train'):
        print ('loading mnist image dataset..')
        image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
        print ('finished loading mnist image dataset..!')
        return images, labels

    def merge_images(self, sources, targets, k=10):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([row*h, row*w*2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[i*h:(i+1)*h, (j*2)*h:(j*2+1)*h, :] = s
            merged[i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h, :] = t
        return merged

    def pretrain(self):
        # load svhn dataset
        train_images, train_labels = self.load_svhn(self.svhn_dir, split='train')
        test_images, test_labels = self.load_svhn(self.svhn_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()
        
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            for step in range(self.pretrain_iter+1):
                i = step % int(train_images.shape[0] / self.batch_size)
                batch_images = train_images[i*self.batch_size:(i+1)*self.batch_size]
                batch_labels = train_labels[i*self.batch_size:(i+1)*self.batch_size] 
                feed_dict = {model.images: batch_images, model.labels: batch_labels}
                sess.run(model.train_op, feed_dict) 

                if (step+1) % 10 == 0:
                    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
                    rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
                    test_acc, _ = sess.run(fetches=[model.accuracy, model.loss], 
                                           feed_dict={model.images: test_images[rand_idxs], 
                                                      model.labels: test_labels[rand_idxs]})
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' \
                               %(step+1, self.pretrain_iter, l, acc, test_acc))

                if (step+1) % 1000 == 0:  
                    saver.save(sess, os.path.join(self.model_save_path, 'svhn_model'), global_step=step+1) 
                    print ('svhn_model-%d saved..!' %(step+1))

    def train(self):
        sess = self.sess
        # load svhn dataset
        # svhn_images, _ = self.load_svhn(self.svhn_dir, split='train')
        # mnist_images, _ = self.load_mnist(self.mnist_dir, split='train')
        source_image = self.load_images(self.source_dir)
        target_image = self.load_images(self.target_dir)

        # build a graph
        model = self.model
        model.build_model(sess)

        #f函数用DCGAN代替
        

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        # with tf.Session(config=self.config) as sess:
        if True:
            sess = self.sess
            # initialize G and D
            tf.global_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            # restore variables of F
            print ('loading pretrained model F..')
            variables_to_restore = slim.get_model_variables(scope='discriminator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print ('start training..!')
            f_interval = 15
            for step in range(self.train_iter+1):

                src_images = sess.run(source_image)
                trg_images = sess.run(target_image)
                

                feed_dict = {model.src_images: src_images}
                
                sess.run(model.d_train_op_src, feed_dict) 
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict)
                
                if step > 1600:
                    f_interval = 30
                
                if step % f_interval == 0:
                    sess.run(model.f_train_op_src, feed_dict)
                
                if (step+1) % 10 == 0:
                    summary, dl, gl, fl = sess.run([model.summary_op_src, \
                        model.d_loss_src, model.g_loss_src, model.f_loss_src], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Source] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f] f_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl, fl))
                
                # train the model for target domain T
                feed_dict = {model.src_images: src_images, model.trg_images: trg_images}
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)

                if (step+1) % 10 == 0:
                    summary, dl, gl = sess.run([model.summary_op_trg, \
                        model.d_loss_trg, model.g_loss_trg], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Target] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl))

                if (step+1) % 200 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step+1)
                    print ('model/dtn-%d saved' %(step+1))
                
    def eval(self):
        # build model
        # with tf.Session(config=self.config) as sess:
        if True:    
            sess = self.sess
            model = self.model
            model.build_model(sess)

            # load svhn dataset
            source_image = self.load_images(self.source_dir)   
            print("source_image", source_image)     
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

            print ('start sampling..!')
            for i in range(self.sample_iter):
                # train model for source domain S
                batch_images = sess.run(source_image)
                print("get batch images")
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images, feed_dict)
                print("generate images")

                # merge and save source images and sampled target images
                merged = self.merge_images(batch_images, sampled_batch_images)
                path = os.path.join(self.sample_save_path, 'sample-%d-to-%d.png' %(i*self.batch_size, (i+1)*self.batch_size))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)