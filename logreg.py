import tensorflow as tf
import numpy as np
from reader import *
from feeder import feeder

def build_model(model, vocab_size, dim_embed, class_num, feature_size, learning_rate, gpu_id="0"):
    m = model(vocab_size, dim_embed, class_num, feature_size)
    m.build_model()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(m.loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu_id
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return m, train_op, sess

def train(model, dim_embed, class_num, learning_rate):
    train_data, dev_data, test_data, vocab_size, feature_size = read_dataset()
    m, train_op, sess = build_model(model, vocab_size, dim_embed, class_num, feature_size, learning_rate)
    fd_train, fd_dev, fd_test = [feeder([m.label, m.x, m.size_list], d) for d in [train_data, dev_data, test_data]]

    batch_size, train_data_size = 1000, len(train_data[0])
    start_epoch, max_epoch = 0, 2000
    for epoch in range(start_epoch, max_epoch):
        for kk, start in enumerate(range(0, train_data_size, batch_size)):
            end = start + batch_size
            if end > train_data_size:break
            _, train_loss = sess.run([train_op, m.loss], feed_dict=fd_train.get(start, end, rnd=True))

        train_loss, train_accuracy = sess.run([m.loss, m.accuracy], feed_dict=fd_train.get())
        dev_loss, dev_accuracy = sess.run([m.loss, m.accuracy], feed_dict=fd_dev.get())
        print "@epoch %s, train_loss : %s, dev_loss : %s, train_accuracy : %s, dev_accuracy : %s" % (epoch, train_loss, dev_loss, round(train_accuracy, 4), round(dev_accuracy, 4))

# TODO:implementation of early stoppin
"""
dev_loss_hist, best_dev_loss = [], 1000
patience, stop_patience = 0, 10
    # if save_epoch:saver.save(sess, os.path.join(model_path, 'model.checkpoint'), global_step=epoch)
    dev_loss_hist.append(dev_loss)
    # if dev_loss > best_dev_loss:
    if dev_loss >= np.mean(dev_loss_hist[-20:]):
        patience += 1
    else:
        patience = 0
        best_dev_loss = dev_loss
        # saver.save(sess, os.path.join(model_path, 'model.checkpoint-best_dev_loss'))
    if patience > stop_patience: break
"""

