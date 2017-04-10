import tensorflow as tf

class model():
    def __init__(self, vocab_size, dim_embed, class_num, feature_size):
        self.vocab_size, self.dim_embed, self.class_num, self.feature_size = vocab_size, dim_embed, class_num, feature_size
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.dim_embed], -1.0, 1.0), name='embeddings')
        self.W = tf.Variable(tf.random_uniform([self.dim_embed, self.class_num], -1.0, 1.0), name='W')
        self.b = tf.Variable(tf.random_uniform([self.class_num], -1.0, 1.0), name='b')

    def build_model(self):
        x = tf.placeholder(tf.int64, [None, self.feature_size])
        label = tf.placeholder(tf.int64, [None])
        size_list = tf.placeholder(tf.float32, [None])

        emb = tf.nn.embedding_lookup(self.embeddings, x)
        h = tf.reduce_sum(emb, 1) / tf.expand_dims(size_list, 1)
        with tf.variable_scope("prediction"):
            logit = tf.matmul(h, self.W) + self.b
            pred = tf.argmax(logit, 1)
        logit = tf.matmul(tf.nn.dropout(h, 0.5), self.W) + self.b
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label)
        self.loss = tf.reduce_sum(cross_entropy) + tf.nn.l2_loss(self.W)

        with tf.variable_scope("metrics"): # something is strange here
            self.accuracy = tf.contrib.metrics.accuracy(pred, label)
            self.precision = tf.contrib.metrics.streaming_precision(pred, label)
            self.recall = tf.contrib.metrics.streaming_recall(pred, label)

        self.x, self.label, self.size_list = x, label, size_list
        self.emb, self.h, self.logit, self.cross_entropy, self.pred = emb, h, logit, cross_entropy, pred

