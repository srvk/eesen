


class Reader():
    def three_stirede(self, feat):
        feat_stride = np.concatenate((np.roll(feat,1,axis=0), feat, np.roll(feat,-1,axis=0)), 1)[shift::stride,]
        return feat_stride;


