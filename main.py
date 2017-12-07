# -*- coding: utf-8 -*-
import tensorflow as tf
from model import DTN
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('fmodel_path', 'checkpoint', "directory for pretrained f model path")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_string('source_img_path', 'source', "directory for source images")
flags.DEFINE_string('target_img_path', 'target', "directory for target images")
flags.DEFINE_integer("input_height", 96, "input image height")
flags.DEFINE_integer("input_width", None, "input image width")
flags.DEFINE_integer("batch_size", 64, "the size of batch")
flags.DEFINE_float("alpha", 0.0, "the multiplier of Consistent Loss")
flags.DEFINE_float("beta", 0.0, "the multiplier of TID Loss")
flags.DEFINE_float("gama", 0.0, "the multiplier of Total Variation Loss")
FLAGS = flags.FLAGS

def main(_):
    run_config = tf.ConfigProto()
    # run_config.allow_soft_placement = True
    run_config.gpu_options.allow_growth=True
    # run_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # run_config.operation_timeout_in_ms = 100000
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height

    with tf.Session(config=run_config) as sess:

        model = DTN(mode=FLAGS.mode, learning_rate=0.00005, height=FLAGS.input_height, width=FLAGS.input_width, 
                    batch_size=FLAGS.batch_size, Alpha=FLAGS.alpha, Beta=FLAGS.beta, Gama=FLAGS.gama)
        solver = Solver(sess, model, batch_size=FLAGS.batch_size, pretrain_iter=20000, train_iter=20000, sample_iter=100, 
                        source_dir=FLAGS.source_img_path, target_dir=FLAGS.target_img_path, 
                        model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path,
                        pretrained_model=FLAGS.fmodel_path)
        
        # create directories if not exist
        if not tf.gfile.Exists(FLAGS.model_save_path):
            tf.gfile.MakeDirs(FLAGS.model_save_path)
        if not tf.gfile.Exists(FLAGS.sample_save_path):
            tf.gfile.MakeDirs(FLAGS.sample_save_path)
        
        if FLAGS.mode == 'pretrain':
            solver.pretrain()
        elif FLAGS.mode == 'train':
            solver.train()
        else:
            solver.eval()
        
if __name__ == '__main__':
    tf.app.run()