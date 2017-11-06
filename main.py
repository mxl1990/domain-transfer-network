import tensorflow as tf
from model import DTN
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_integer("input_height", 96, "input image height")
flags.DEFINE_integer("input_width", None, "input image width")
flags.DEFINE_integer("batch_size", 64, "the size of batch")
FLAGS = flags.FLAGS

def main(_):
    run_config = tf.ConfigProto()
    # run_config.operation_timeout_in_ms = 100000
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height

    with tf.Session(config=run_config) as sess:

        model = DTN(mode=FLAGS.mode, learning_rate=0.0003, height=FLAGS.input_height, width=FLAGS.input_width, 
                    batch_size=FLAGS.batch_size)
        solver = Solver(sess, model, batch_size=FLAGS.batch_size, pretrain_iter=20000, train_iter=20000, sample_iter=100, 
                        source_dir='D:\\workspace\\tensorflow-101.git\\DCGAN\\trunk\\data\\resize', 
                        target_dir='D:\\workspace\\tensorflow-101.git\\DCGAN\\trunk\\data\\faces', 
                        model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
        
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