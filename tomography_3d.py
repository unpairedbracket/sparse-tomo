import numpy as np
from scipy.interpolate import interp1d

import tomopy as tp

from angular_interpolation import angular_interpolate

def project(field, N_angles, rot_axis, scale_factors=(1,1,1), phi_0=0):
    scale_factors=np.array(scale_factors)
    angles = phi_0 + np.linspace(0, np.pi, N_angles, endpoint=False)
    field = field.swapaxes(0, rot_axis)
    scale_factors[[0, rot_axis]] = scale_factors[[rot_axis, 0]]

    dx, dy = scale_factors[1:].astype('float')
    Nx, Ny = np.array(field.shape[1:], dtype='float')

    angles_warp = np.arctan2(dy*np.sin(angles), dx*np.cos(angles))

    sino = tp.project(field, angles_warp, pad=True)

    scale_max = np.fmax(dx, dy)
    s_diag = np.sqrt((Nx/dx)**2 + (Ny/dy)**2)
    N_interp = int(2 * np.ceil(s_diag * scale_max / 2))

    s0 = np.linspace(-1., 1., sino.shape[2]) * (float(sino.shape[2])-1)/2 # M^0
    s = np.linspace(-1., 1., N_interp) * (N_interp - 1) / (2 * scale_max)# 1/M
    S = np.zeros((sino.shape[0], sino.shape[1], N_interp))
    for j, theta_warp in enumerate(angles_warp):
        factor = np.sqrt((dy*np.cos(theta_warp))**2
                       + (dx*np.sin(theta_warp))**2) # M
        S[j] = interp1d(s0, sino[j], fill_value=0, bounds_error=False, kind='linear')(s*factor) * factor / scale_max
    return S / ( dx * dy / scale_max**2 )


def angular_interp(sinogram, N_interp, n_only=None):

    sino_2pi = pi_to_2pi(sinogram)
    _, sino_terp = angular_interpolate(sino_2pi, N_interp, n_only)
    return pi_from_2pi(sino_terp)


def reconstruct(sinogram, rot_axis, scale_factors, phi_0=0, filter_func=None, k_ratio=1):
    N_angles = sinogram.shape[0]

    _, sino_filt = filter_sinogram(sinogram, filter_func, k_ratio)

    angles = phi_0 + np.linspace(0, np.pi, N_angles, endpoint=False)

    recon = tp.recon(sino_filt, angles, algorithm='fbp') * np.pi / (2*N_angles)

    recon.swapaxes(0, rot_axis)

    return recon


def filter_sinogram(sinogram, filter_func=None, k_ratio=1):
    N_fft = int(2**(1 + np.floor(np.log2(sinogram.shape[2]))))

    filt = ram_lak(N_fft)
    k = k_ratio * 2*np.fft.fftfreq(N_fft)
    filt *= (np.abs(k) <= 1)
    if filter_func is not None:
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
