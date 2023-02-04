import numpy as np
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

means = np.array([[-0.04193198,1.11485344,22.53207206,-0.08767917,0.23441203,3.22835194],[-0.11536963,1.60988842,31.13567036,-0.055891,0.12901063,3.04363133],[0.11299296,1.54344841,27.29784195,0.06019754,0.09676839,3.9023696,],[0.18788007,1.44682264,25.4480972,0.09091709,1.88950034,4.72486333]])

cov0 = np.array([[1.83791745e-02,-4.17533457e-05,4.31252886e-02,1.53180272e-02,1.79897652e-02,-3.82045193e-02],[-4.17533457e-05,3.42521995e-02,1.92525116e-01,3.33387781e-03,2.36566065e-02,1.59020937e-02],[4.31252886e-02,1.92525116e-01,3.46683724e+00,7.27749443e-02,-1.75223309e-01,1.26626816e-01],[1.53180272e-02,3.33387781e-03,7.27749443e-02,1.41577817e-02,1.50638645e-02,-4.12333447e-02],[1.79897652e-02,2.36566065e-02,-1.75223309e-01,1.50638645e-02,1.96835879e-01,-1.01209421e-01],[-3.82045193e-02,1.59020937e-02,1.26626816e-01,-4.12333447e-02,-1.01209421e-01,6.19909298e-01]])

cov1 = np.array([[5.28672473e-02,-5.93058725e-02,-2.66495058e-01,2.78770099e-02,1.99280947e-02,-1.67176833e-02],[-5.93058725e-02,1.04229593e-01,5.98204650e-01,-2.83650476e-02,-7.04457119e-03,5.01087584e-02],[-2.66495058e-01,5.98204650e-01,8.65902471e+00,-8.10645489e-02,-3.56901361e-01,-3.97971091e-01],[2.78770099e-02,-2.83650476e-02,-8.10645489e-02,1.56402534e-02,1.03962277e-02,-1.53444878e-02],[1.99280947e-02,-7.04457119e-03,-3.56901361e-01,1.03962277e-02,1.40646384e-01,8.94053089e-02],[-1.67176833e-02,5.01087584e-02,-3.97971091e-01,-1.53444878e-02,8.94053089e-02,2.95198661e-01]])

cov2 = np.array([[6.02348349e-02,-9.86748943e-03,-2.78588256e-02,3.44697655e-02,7.89222264e-03,-5.05464547e-02],[-9.86748943e-03,1.04816669e-01,3.30098022e-01,-1.33773837e-02,-3.40435931e-03,2.51987139e-01],[-2.78588256e-02,3.30098022e-01,3.04080826e+00,-1.50912327e-03,-1.52185221e-01,4.56069308e-01],[3.44697655e-02,-1.33773837e-02,-1.50912327e-03,2.16572723e-02,4.24825013e-03,-5.58765740e-02],[7.89222264e-03,-3.40435931e-03,-1.52185221e-01,4.24825013e-03,1.26409787e-01,4.25383383e-02],[-5.05464547e-02,2.51987139e-01,4.56069308e-01,-5.58765740e-02,4.25383383e-02,8.32573255e-01]])

cov3 = np.array([[2.33486370e-02,1.63393056e-04,-3.60046458e-02,1.45586485e-02,9.13635344e-02,6.15897870e-02],[1.63393056e-04,3.25649452e-02,2.04282481e-01,1.70951778e-03,4.48130940e-02,6.82207127e-02],[-3.60046458e-02,2.04282481e-01,3.15735136e+00,7.36631563e-03,-2.16151226e-01,-6.08526178e-02],[1.45586485e-02,1.70951778e-03,7.36631563e-03,9.81194874e-03,5.86924208e-02,3.52841009e-02],[9.13635344e-02,4.48130940e-02,-2.16151226e-01,5.86924208e-02,7.91495055e-01,5.78903988e-01],[6.15897870e-02,6.82207127e-02,-6.08526178e-02,3.52841009e-02,5.78903988e-01,6.48033348e-01]])

covariances = np.vstack((cov0,cov1,cov2,cov3))
np.savetxt("meandata.csv", means, fmt='%f', delimiter=",")
np.savetxt("covdata.csv", covariances, fmt='%f', delimiter=",")
color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

def plot_results(X, Y, means, covariances, index, title):
    splot = plt.subplot(5, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6.0, 4.0 * np.pi - 6.0)
    plt.ylim(-5.0, 5.0)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

def plot_samples(X, Y, n_components, index, title):
    plt.subplot(5, 1, 4 + index)
    for i, color in zip(range(n_components), color_iter):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)

    plt.xlim(-6.0, 4.0 * np.pi - 6.0)
    plt.ylim(-5.0, 5.0)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())