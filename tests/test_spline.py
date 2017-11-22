import splinart as spl
import numpy as np
import pytest

@pytest.mark.parametrize("beg, end", [
    (0 , 1),
    (-2, -1),
    (-20, 10),
])
@pytest.mark.parametrize("nbpoints", [
    (10),
    (20),
    (30)
])
def test_spline_1(beg, end, nbpoints):
    path = spl.line(beg, end, npoints=nbpoints)
    y2s = spl.spline.spline(path[:, 0], path[:, 1])
    assert y2s == pytest.approx(np.zeros(nbpoints))

@pytest.mark.parametrize("beg, end", [
    (0 , 1),
    (-2, -1),
    (-20, 10),
])
@pytest.mark.parametrize("nbpoints", [
    (10),
    (20),
    (30)
])
@pytest.mark.parametrize("slope", [
    (1.),
    (2.),
    (-5.)
])
def test_spline_2(beg, end, slope, nbpoints):
    x = np.linspace(beg, end, nbpoints)
    y = slope * x
    y2s = spl.spline.spline(x, y)
    assert y2s == pytest.approx(np.zeros(nbpoints), abs=1e-6)


@pytest.mark.parametrize("center, radius", [
    ([0, 0] , 1),
    ([0.5, 0.1] , .1),
    ([-0.5, 0.6] , .3),
])
@pytest.mark.parametrize("nbpoints", [
    (10),
    (20),
    (30)
])
def test_spline_circle(center, radius, nbpoints):
    theta, path = spl.circle(center, radius, npoints=nbpoints)
    y2s = spl.spline.spline(theta, path)
    y_new = np.zeros_like(path)
    spl.spline.splint(theta, path, y2s, theta, y_new)
    assert path == pytest.approx(y_new)