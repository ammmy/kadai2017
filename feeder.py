import numpy as np

class feeder():
    def __init__(self, k, v):
        self.d = {kk:vv for kk, vv in zip(k, v)}
    def get(self, s=None, e=None, rnd=False):
        if s == None and e == None:
            return self.get_all()
        elif rnd:
            return self.get_batch_rnd(s, e)
        else:
            return self.get_batch(s, e)
    def get_batch(self, s, e):
        return {kk:vv[s:e] for kk, vv in self.d.iteritems()}
    def get_all(self):
        return {kk:vv for kk, vv in self.d.iteritems()}
    def get_batch_rnd(self, s, e):
        if s == 0:
            self.rnd_idx = range(len(self.d[self.d.keys()[0]]))
            np.random.shuffle(self.rnd_idx)
        return {kk:vv[self.rnd_idx][s:e] for kk, vv in self.d.iteritems()}

