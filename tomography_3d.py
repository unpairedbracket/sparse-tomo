import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import tomopy as tp

from angular_interpolation import angular_interpolate

def project(field, N_angles, rot_axis, scale_factors, phi_0=0):
    angles = phi_0 + np.linspace(0, np.pi, N_angles, endpoint=False)
    field = field.swapaxes(0, rot_axis)
    scale_factors[[0, rot_axis]] = scale_factors[[rot_axis, 0]]
    if field.shape[1] < field.shape[2]:
        field = np.rot90(field, k=1, axes=(1,2))
        angles += np.pi/2
        
    angles_warp = np.arctan2(scale_factors[2]*np.sin(angles), scale_factors[1]*np.cos(angles))

    sino = tp.project(field, angles_warp, pad=False)

    maxscale = np.max(scale_factors[1:])
    s0 = np.linspace(-1, 1, sino.shape[2]) * maxscale
    s = np.linspace(-2, 2, 2*sino.shape[2])
    S = np.zeros((sino.shape[0], sino.shape[1], s.shape[0]))
    for j, (theta, theta_warp) in enumerate(zip(angles, angles_warp)):
        factor = np.sqrt((scale_factors[2]*np.cos(theta_warp))**2 
                       + (scale_factors[1]*np.sin(theta_warp))**2)
        S[j] = interp1d(s0, sino[j], fill_value=0, bounds_error=False)(s*factor) * factor / maxscale
    return S


def angular_interp(sinogram, N_interp, n_only=None):
    
    sino_2pi = pi_to_2pi(sinogram)
    _, sino_terp = angular_interpolate(sino_2pi, N_interp, n_only)
    return pi_from_2pi(sino_terp)


def reconstruct(sinogram, rot_axis, scale_factors, filter_func=None):
    N_angles = sinogram.shape[0]

    _, sino_filt = filter_sinogram(sinogram, filter_func)

    angles = np.linspace(0, np.pi, N_angles, endpoint=False)

    recon = tp.recon(sino_filt, angles, algorithm='fbp') * np.pi / (2*N_angles)

    recon.swapaxes(0, rot_axis)

    return recon


def filter_sinogram(sinogram, filter_func=None):
    N_fft = int(2**(1 + np.floor(np.log2(sinogram.shape[2]))))

    filt = ram_lak(N_fft)
    if filter_func is not None:
        k = 2*np.fft.fftfreq(N_fft)
        if filter_func == 'shepp':
            filt *= np.sinc(k/2)  # numpy sinc is sin(pi x)/(pi x)
        elif filter_func == 'cosine':
            filt *= np.cos(np.pi/2 * k)
        elif filter_func == 'hann':
            filt *= (1 + np.cos(np.pi * k))/2    
        elif callable(filter_func):
            filt *= filter_func(k)
        else:
            print(f"Filter function {filter_func} unknown. Falling back to Ramachandran-Lakshminarayanan filter.")

    sino_filt = np.fft.ifft(
        filt * np.fft.fft(
            sinogram, n=N_fft, axis=2
        ), axis=2
    ).real
    return filt, sino_filt[:, :, :sinogram.shape[2]]


def pi_to_2pi(sinogram_half):
    sinogram_full = np.concatenate((
        sinogram_half,
        np.flip(sinogram_half, axis=2)
    ), axis=0)
    return sinogram_full


def pi_from_2pi(sinogram_full):
    return sinogram_full[:sinogram_full.shape[0]//2]


def ram_lak(N):
    realspace = np.zeros(N)
    realspace[0] = 1/4
    realspace[ 1:N//2: 2] = - ( np.pi * np.arange(N//2)[1::2])**-2
    realspace[-1:N//2:-2] = - ( np.pi * np.arange(N//2)[1::2])**-2

    return 2 * np.fft.fft(realspace)


if __name__ == '__main__':
    run()
    plt.show()

def run():
    F = tp.shepp2d()
    plot_results(F, 16, [1,4,3], False, 0, 0)


def plot_results(F, n_views, aspect, plot_aspect=False, axis=0, phi_0=0):

    P0 = project(F, n_views, axis, np.ones(3), phi_0)
    Pi = angular_interp(P0, 360)
    R0 = reconstruct(P0, 0, [1,1,1])
    Ri = reconstruct(Pi, 0, [1,1,1])

    P0a = project(F, n_views, axis, aspect, phi_0)
    Pia = angular_interp(P0a, 360)
    R0a = reconstruct(P0a, 0, [1,1,1])
    Ria = reconstruct(Pia, 0, [1,1,1])

    fmin = F.min()

    if fmin < 0:
        vmax = np.max(np.abs(F))
        vmin = -vmax
        cmap = 'RdBu_r'
    else:
        vmax = np.max(F)
        vmin = 0
        cmap = 'plasma'

    fig = plt.figure()

    axF = fig.add_subplot(2,3,1)
    axF.pcolormesh(F[0], vmin=vmin, vmax=vmax, cmap=cmap)
    axF.axis('image')

    axR0 = fig.add_subplot(2,3,2)
    axRi = fig.add_subplot(2,3,3)
    axR0.pcolormesh(R0[0], vmin=vmin, vmax=vmax, cmap=cmap)
    axRi.pcolormesh(Ri[0], vmin=vmin, vmax=vmax, cmap=cmap)
    axR0.axis('image')
    axRi.axis('image')

    axR0a = fig.add_subplot(2,3,5)
    axRia = fig.add_subplot(2,3,6)
    axR0a.pcolormesh(R0a[0], vmin=vmin, vmax=vmax, cmap=cmap)
    axRia.pcolormesh(Ria[0], vmin=vmin, vmax=vmax, cmap=cmap)
    axR0a.axis('image')
    axRia.axis('image')
