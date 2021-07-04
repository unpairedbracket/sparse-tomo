import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

from skimage.metrics import structural_similarity, normalized_root_mse

from tomography_3d import project, angular_interp, reconstruct

def plot_results(F, n_views, aspect, plot_aspect=False, phi_0=0, filter_func=None, plot=True):

    aspect = np.array(aspect)
    dx, dy = aspect[1:]
    maxpect = aspect[1:].max()
    ratio = maxpect**2 / (dx * dy)
    k_ratio = np.sqrt( (ratio**2 + np.tan(np.pi/2 * (1 - 2./n_views))) /
                       (  1.0**2 + np.tan(np.pi/2 * (1 - 2./n_views))))

    P0 = project(F, n_views, 0, np.ones(3), phi_0)
    Pi = angular_interp(P0, 512)
    R0 = reconstruct(P0, 0, [1,1,1], phi_0, filter_func)
    Ri = reconstruct(Pi, 0, [1,1,1], phi_0, filter_func)

    P0a = project(F, n_views, 0, aspect, phi_0)
    Pia = angular_interp(P0a, 512)
    R0a = reconstruct(P0a, 0, aspect, phi_0, filter_func, k_ratio)
    Ria = reconstruct(Pia, 0, aspect, phi_0, filter_func, k_ratio)

    fmin = F.min()

    if fmin < 0:
        vmax = np.max(np.abs(F))
        vmin = -vmax
        cmap = 'RdBu_r'
    else:
        vmax = np.max(F)
        vmin = 0
        cmap = 'plasma'

    xF = np.linspace(-1, 1, F.shape[1]+1) * (F.shape[1])/2
    yF = np.linspace(-1, 1, F.shape[2]+1) * (F.shape[2])/2

    xR = np.linspace(-1, 1, R0.shape[1]+1) * (R0.shape[1])/2
    yR = np.linspace(-1, 1, R0.shape[2]+1) * (R0.shape[2])/2

    xa = np.linspace(-1, 1, R0a.shape[1]+1) * (R0a.shape[1])/2 * dx / maxpect
    ya = np.linspace(-1, 1, R0a.shape[2]+1) * (R0a.shape[2])/2 * dy / maxpect

    if plot_aspect:
        xF /= dx; xR /= dx; xa /= dx
        yF /= dy; yR /= dy; ya /= dy

    if plot:
        fig = plt.figure()

        #axF = fig.add_subplot(2,3,1)
        #axF = fig.add_subplot(4,1,1)
        #axF.imshow(F[0].T, extent=(xF[0], xF[-1], yF[0], yF[-1]), vmin=vmin, vmax=vmax, cmap=cmap)

        #axR0 = fig.add_subplot(2,3,2)
        #axRi = fig.add_subplot(2,3,3)
        axR0 = fig.add_subplot(4,1,1)
        axRi = fig.add_subplot(4,1,2)
        axR0.imshow(R0[0].T, extent=(xR[0], xR[-1], yR[0], yR[-1]), vmin=vmin, vmax=vmax, cmap=cmap)
        axRi.imshow(Ri[0].T, extent=(xR[0], xR[-1], yR[0], yR[-1]), vmin=vmin, vmax=vmax, cmap=cmap)
        axR0.set_xlim(xF[[0, -1]])
        axR0.set_ylim(yF[[0, -1]])
        axRi.set_xlim(xF[[0, -1]])
        axRi.set_ylim(yF[[0, -1]])

        #axR0a = fig.add_subplot(2,3,5)
        #axRia = fig.add_subplot(2,3,6)
        axR0a = fig.add_subplot(4,1,3)
        axRia = fig.add_subplot(4,1,4)
        axR0a.imshow(R0a[0].T, extent=(xa[0], xa[-1], ya[0], ya[-1]), vmin=vmin, vmax=vmax, cmap=cmap)
        axRia.imshow(Ria[0].T, extent=(xa[0], xa[-1], ya[0], ya[-1]), vmin=vmin, vmax=vmax, cmap=cmap)
        axR0a.set_xlim(xF[[0, -1]])
        axR0a.set_ylim(yF[[0, -1]])
        axRia.set_xlim(xF[[0, -1]])
        axRia.set_ylim(yF[[0, -1]])

    c = lambda a: (a[1:] + a[:-1])/2
    F_interp = interp2d(c(yF), c(xF), F[0], fill_value=0)(c(yF), c(xF))
    R0_interp = interp2d(c(yR), c(xR), R0[0], fill_value=0)(c(yF), c(xF))
    Ri_interp = interp2d(c(yR), c(xR), Ri[0], fill_value=0)(c(yF), c(xF))
    R0a_interp = interp2d(c(ya), c(xa), R0a[0], fill_value=0)(c(yF), c(xF))
    Ria_interp = interp2d(c(ya), c(xa), Ria[0], fill_value=0)(c(yF), c(xF))
    return F_interp, R0_interp, Ri_interp, R0a_interp, Ria_interp

def plot_multiple_views(F, n_views, subplot_shape, n_interp=None, aspect=(1,1,1), phi_0=0, filter_func=None):

    aspect = np.array(aspect)
    do_plot = True
    if subplot_shape is None:
        do_plot = False
        axes = np.zeros(len(n_views))
    elif len(n_views) != np.product(subplot_shape):
        print(f'Number of view numbers specified {len(n_views)} does not match number of subplots requested {np.product(subplot_shape)}')
        return
    else:
        _, axes = plt.subplots(*subplot_shape)
        
        fmin = F.min()

        if fmin < 0:
            vmax = np.max(np.abs(F))
            vmin = -vmax
            cmap = 'RdBu_r'
        else:
            vmax = np.max(F)
            vmin = 0
            cmap = 'plasma'

    xF = np.linspace(-1, 1, F.shape[1]+1) #* (F.shape[1])/2
    yF = np.linspace(-1, 1, F.shape[2]+1) #* (F.shape[2])/2
   
    dx, dy = aspect[1:]
    maxpect = aspect[1:].max()
    ratio = maxpect**2 / (dx * dy)

    c = lambda a: (a[1:] + a[:-1])/2
    ssim = np.zeros(len(n_views))
    rmse = np.zeros(len(n_views))
    f = F[0].astype('float')
    for j, (N, axis) in enumerate(zip(n_views, axes.flat)):
        
        P = project(F, N, 0, aspect, phi_0)

        k_ratio = np.sqrt( (ratio**2 + np.tan(np.pi/2 * (1 - 2./N))) /
                       (  1.0**2 + np.tan(np.pi/2 * (1 - 2./N))))

        if n_interp:
            P = angular_interp(P, n_interp)
        
        R = reconstruct(P, 0, aspect, phi_0, filter_func, k_ratio)

        xa = np.linspace(-1, 1, R.shape[1]+1) * dx / maxpect * R.shape[1]/F.shape[1]
        ya = np.linspace(-1, 1, R.shape[2]+1) * dy / maxpect * R.shape[2]/F.shape[2]

        if do_plot:
            axis.imshow(R[0].T, extent=(xa[0], xa[-1], ya[0], ya[-1]), vmin=vmin, vmax=vmax, cmap=cmap, interpolation='none')
            axis.set_xlim(xF[[0, -1]])
            axis.set_ylim(yF[[0, -1]])

        r = interp2d(c(ya), c(xa), R[0], fill_value=0)(c(yF), c(xF))
        ssim[j] = structural_similarity(f, r)
        rmse[j] = normalized_root_mse(f, r)

    return ssim, rmse


def plot_multiple_aspects(F, n_views, aspects, phi_0=0, filter_func=None):

    _, axes = plt.subplots(1, len(aspects))
        
    fmin = F.min()

    if fmin < 0:
        vmax = np.max(np.abs(F))
        vmin = -vmax
        cmap = 'RdBu_r'
    else:
        vmax = np.max(F)
        vmin = 0
        cmap = 'plasma'

    xF = np.linspace(-1, 1, F.shape[1]+1) #* (F.shape[1])/2
    yF = np.linspace(-1, 1, F.shape[2]+1) #* (F.shape[2])/2

    c = lambda a: (a[1:] + a[:-1])/2
    ret = []
    for aspect, axis in zip(aspects, axes.flat):
        aspect = np.array(aspect)
        dx, dy = aspect[1:]
        maxpect = aspect[1:].max()
        ratio = maxpect**2 / (dx * dy)

        k_ratio = np.sqrt( (ratio**2 + np.tan(np.pi/2 * (1 - 2./n_views))) /
                           (  1.0**2 + np.tan(np.pi/2 * (1 - 2./n_views))))

        P0a = project(F, n_views, 0, aspect, phi_0)
        Pia = angular_interp(P0a, 360)
        Ria = reconstruct(Pia, 0, aspect, phi_0, filter_func, k_ratio)

        xa = np.linspace(-1, 1, Ria.shape[1]+1) * dx / maxpect * Ria.shape[1]/F.shape[1]
        ya = np.linspace(-1, 1, Ria.shape[2]+1) * dy / maxpect * Ria.shape[2]/F.shape[2]

        axis.imshow(Ria[0].T, extent=(xa[0], xa[-1], ya[0], ya[-1]), vmin=vmin, vmax=vmax, cmap=cmap, interpolation='none')
        axis.set_xlim(xF[[0, -1]])
        axis.set_ylim(yF[[0, -1]])

        R_interp = interp2d(c(ya), c(xa), Ria[0], fill_value=0)(c(yF), c(xF))
        ret.append(R_interp)

    return ret

def plot_multiple(F, n_views, aspect, plot_aspect=False, phi_0=0, filter_func=None):

    dx, dy = aspect[1:]
    maxpect = aspect[1:].max()
    ratio = maxpect**2 / (dx * dy)

    fig, axes = plt.subplots(len(n_views), 4)
        
    fmin = F.min()

    if fmin < 0:
        vmax = np.max(np.abs(F))
        vmin = -vmax
        cmap = 'RdBu_r'
    else:
        vmax = np.max(F)
        vmin = 0
        cmap = 'plasma'

    xF = np.linspace(-1, 1, F.shape[1]+1) * (F.shape[1])/2
    yF = np.linspace(-1, 1, F.shape[2]+1) * (F.shape[2])/2

    if plot_aspect:
        xF /= dx
        yF /= dy

    for N, axs in zip(n_views, axes):
        k_ratio = np.sqrt( (ratio**2 + np.tan(np.pi/2 * (1 - 2./N))) /
                           (  1.0**2 + np.tan(np.pi/2 * (1 - 2./N))))

        P0 = project(F, N, 0, np.ones(3), phi_0)
        Pi = angular_interp(P0, 360)
        R0 = reconstruct(P0, 0, np.ones(3), phi_0, filter_func)
        Ri = reconstruct(Pi, 0, np.ones(3), phi_0, filter_func)

        P0a = project(F, N, 0, aspect, phi_0)
        Pia = angular_interp(P0a, 360)
        R0a = reconstruct(P0a, 0, aspect, phi_0, filter_func, k_ratio)
        Ria = reconstruct(Pia, 0, aspect, phi_0, filter_func, k_ratio)

        xR = np.linspace(-1, 1, R0.shape[1]+1) * (R0.shape[1])/2
        yR = np.linspace(-1, 1, R0.shape[2]+1) * (R0.shape[2])/2

        xa = np.linspace(-1, 1, R0a.shape[1]+1) * (R0a.shape[1])/2 * dx / maxpect
        ya = np.linspace(-1, 1, R0a.shape[2]+1) * (R0a.shape[2])/2 * dy / maxpect

        if plot_aspect:
            xR /= dx; xa /= dx
            yR /= dy; ya /= dy

        for axis, x, y, R in zip(axs, [xR, xa, xR, xa], [yR, ya, yR, ya], [R0, R0a, Ri, Ria]):
            axis.imshow(R[0].T, extent=(x[0], x[-1], y[0], y[-1]), vmin=vmin, vmax=vmax, cmap=cmap)
            axis.set_xlim(xF[[0, -1]])
            axis.set_ylim(yF[[0, -1]])


def run():
    import tomopy as tp
    F = tp.shepp2d()
    plot_results(F, 16, [1,4,3], False, 0, 0)


if __name__ == '__main__':
    run()
    plt.show()
