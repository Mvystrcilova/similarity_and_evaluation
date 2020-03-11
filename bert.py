import tensorflow as tf


saver = tf.train.import_meta_graph('bert_model/bert_model.ckpt.meta')
    # print(sess.run('w1:0'))