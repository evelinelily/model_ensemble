#
# Definitions of  special metrics for evaluators
#
import numpy as np

def compute_r_at_p(y_true, y_score, precision_thresh):
    labels = list(y_true)
    scores = list(y_score)
    total_yes = sum(labels)

    # sort scores in descend order
    bundle = sorted(zip(scores, labels), key=lambda x: -x[0])

    recalls, precisions = list(), list()
    yes_counter = 0
    for ix, (_, lb) in enumerate(bundle):
        all_counter = ix+1
        yes_counter += lb  # when lb = 1, it yelps "YES!"
        recalls.append(yes_counter / total_yes)
        precisions.append(yes_counter/all_counter)

    # import matplotlib.pyplot as plt
    # plt.plot(recalls, precisions, '.')
    # plt.show()

    mask = np.array(precisions) >= precision_thresh
    if any(mask) > 0:
        return max(np.array(recalls)[mask])
    else:
        bundle = sorted(zip(precisions, recalls), key=lambda x: '{:.5f}-{:.5f}'.format(*x), reverse=True)

        assert len(bundle) > 1, "Only 1 sample in the list"

        # find the maximum recall at precision X
        target = bundle[0]
        for prec, rec in bundle:
            # print('prec:', prec, 'rec:', rec)
            if prec < precision_thresh:
                break
            if prec != target[0]:
                target = (prec, rec)

        # print(target)

        # precs, recs = list(zip(*bundle))
        # import matplotlib.pyplot as plt
        # plt.plot(precs, recs, '.')
        # plt.show()


        # add punishment to the metric if precision is not close to precision_thresh
        prec, rec = target
        delta_prec = -min(prec-precision_thresh, 0)

        # import matplotlib.pyplot as plt
        # x = np.linspace(0,1,100)
        # y = np.power(x, 0.2)
        # plt.plot(x,y)
        # plt.show()

        punishment = np.power(delta_prec, 0.2)

        return rec - punishment



if __name__ == "__main__":
    N = 1000
    np.random.seed(123)
    y_true = (np.random.rand(N) > .9).astype(np.int32)
    y_score = np.random.rand(N)

    rec = compute_r_at_p(y_true, y_score, precision_thresh=.9)
    print(rec)
