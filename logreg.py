import tensorflow as tf
import numpy as np
from model_simple import model
from reader import *

def build_model(vocab_size, dim_embed, class_num, feature_size):
    m = model(vocab_size, dim_embed, class_num, feature_size)
    loss, placeholders, elements = m.build_model()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return m, loss, placeholders, elements, train_op, sess

class feeder():
    def __init__(self, k, v):
        self.d = {kk:vv for kk, vv in zip(k, v)}
    def get(self, s, e):
        return {kk:vv[s:e] for kk, vv in self.d.iteritems()}
    def get_all(self):
        return {kk:vv for kk, vv in self.d.iteritems()}

dim_embed = 50
class_num = 2
learning_rate = 0.001

train_data, dev_data, test_data, vocab_size, feature_size = read_dataset()
m, loss, placeholders, elements, train_op, sess = build_model(vocab_size, dim_embed, class_num, feature_size)
x, label, size_list = placeholders
emb, h, logit, pred, cross_entropy, accuracy, precision, recall = elements
fd_train, fd_dev, fd_test = [feeder([label, x, size_list], d) for d in [train_data, dev_data, test_data]]

batch_size, train_data_size = 100, len(train_data[0])
start_epoch, max_epoch = 0, 1000
dev_loss_hist, best_dev_loss = [], 1000
patience, stop_patience = 0, 10
for epoch in range(start_epoch, max_epoch):
    for kk, start in enumerate(range(0, train_data_size, batch_size)):
        end = start + batch_size
        if end > train_data_size:break
        _, train_loss = sess.run([train_op, loss], feed_dict=fd_train.get(start, end))

    train_loss, train_accuracy = sess.run([loss, accuracy], feed_dict=fd_train.get_all())
    dev_loss, dev_accuracy = sess.run([loss, accuracy], feed_dict=fd_dev.get_all())
    print "@epoch %s, train_loss : %s, dev_loss : %s, train_accuracy : %s, dev_accuracy : %s" % (epoch, train_loss, dev_loss, round(train_accuracy, 4), round(dev_accuracy, 4))


"""
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

