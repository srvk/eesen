import tf, sys
from main import load_feat

if __name__ == "__main__":
    use_cudnn = True
    cv_x, _0, cv_y, _ = load_feat(use_cudnn)
    nfeat = cv_x[0].shape[-1]
    config = {
        "nfeat": nfeat, 
        "nclass": 36,
        "nepoch": 1,
        "lr_rate": 3e-2, 
        "l2": 0.0,
        "clip": 0.1,
        "nlayer": 6,
        "nhidden": 140,
        "nproj": 0,
	"cudnn": use_cudnn,
        "grad_opt": "grad"
    }
    data = (cv_x, cv_y)
    model_path = sys.argv[1] 
    print(model_path)
    tf.eval(data, config, model_path)
