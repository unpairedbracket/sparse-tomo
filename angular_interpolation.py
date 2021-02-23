from numpy import ceil, floor, concatenate, zeros, linspace, pi
from numpy.fft import fft, ifft, fftshift, ifftshift

def angular_interpolate(f, N_interp, n_only=None):
    '''
    Interpolate a function f(phi_in) to a band-limited function f(phi)
    phi_in = phi_0 + 2*pi*(0..N-1)/N
    phi = phi_0 + 2*pi*(0..N_interp-1)/N_interp
    Arguments:
      f: array, shape (..., N): Array to be interpolated
      N_interp: int > N: Number of points in interpolated array
    Returns:
      phi: array, shape (N_interp,): interpolated angular variable.
            Zero-offset is the same as angular axis of input variable.
      F: array, shape (..., N_interp): interpolated function,
            defined at angles given in phi
    '''
    N = f.shape[0]
    g = fftshift(fft(f,axis=0), axes=0) / N
    if N % 2 == 0:
        g[0] /= 2
        g = concatenate((g, g[[0]]), axis=0)
        N += 1

    if n_only is not None:
        g_temp = 0*g
        zero_pos = (N-1)//2
        g_temp[zero_pos + n_only] = g[zero_pos + n_only]
        g_temp[zero_pos - n_only] = g[zero_pos - n_only]
        g = g_temp

    lzeroshape = list(f.shape)
    rzeroshape = list(f.shape)
    lzeroshape[0] = int(ceil((N_interp - N) / 2.0))
    rzeroshape[0] = int(floor((N_interp - N) / 2.0))
    lzeros = zeros(lzeroshape, 'c16')
    rzeros = zeros(rzeroshape, 'c16')
    G = ifftshift(concatenate((lzeros, g, rzeros), axis=0), axes=0)
    F = ifft(G, axis=0) * N_interp
    phi = linspace(0, 2*pi, N_interp, endpoint=False)
    return phi, F.real
