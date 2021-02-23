from numpy import sin, cos, pi

from tomopy.misc.phantom import phantom, _totuple, _array_to_params


def bshepp_2d(size):
    size = _totuple(size, 2)
    size = (3, size[0], size[1])
    F = _bshepp(size)
    return F[[1], :, :]


def bshepp_3d(size):
    size = _totuple(size, 3)
    return _bshepp(size)


def _bshepp(size):
    x0 = 0.22 * cos(pi/10)
    y0 = 0.22 * sin(pi/10)
    shepp_array = [
        [+1., .6900, .920, .810,   0.,     0.,   0.,   90.,   90.,   90.],
        [-1., .6624, .874, .780,   0., -.0184,   0.,   90.,   90.,   90.],
        [+3., .1100, .310, .220,   x0,     y0,   0.,   90.,   90.,   72.],
        [-3., .1600, .410, .280,  -x0,     y0,   0.,   90.,   90.,  108.],
        [+1., .2100, .250, .410,   0.,    .35,   0.,   90.,   90.,   90.],
        [-3., .0460, .046, .050,   0.,     .1,   0.,   90.,   90.,   90.],
        [+1., .0460, .046, .050,   0.,    -.1,   0.,   90.,   90.,   90.],
        [-5., .0460, .023, .050, -.08,  -.605,   0.,   90.,   90.,   90.],
        [+1., .0230, .023, .020,   0.,  -.606,   0.,   90.,   90.,   90.],
        [-5., .0230, .046, .020,  .06,  -.605,   0.,   90.,   90.,   90.]
    ]

    shepp_params = _array_to_params(shepp_array)
    return phantom(size, shepp_params)

