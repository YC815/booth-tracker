import numpy as np

from booth_tracker.filter import JointFilter, OneEuroFilter
from booth_tracker.types import LANDMARKS as JOINTS


def test_常數輸入等於輸出():
    f = OneEuroFilter(1.0, 0.05)
    x = np.array([0.1, 1.2, 2.0])
    for i in range(30):
        out = f(x, i / 30)
    np.testing.assert_allclose(out, x, atol=1e-9)


def test_第一筆原樣輸出():
    f = OneEuroFilter(1.0, 0.05)
    np.testing.assert_allclose(f(np.array([1.0, 2.0, 3.0]), 0.0), [1.0, 2.0, 3.0])


def test_階躍輸入單調趨近():
    f = OneEuroFilter(1.0, 0.0)
    f(np.array([0.0]), 0.0)
    prev = 0.0
    for i in range(1, 60):
        y = float(f(np.array([1.0]), i / 30)[0])
        assert prev < y <= 1.0
        prev = y
    assert prev > 0.9


def test_時間倒退或重複不爆():
    f = OneEuroFilter(1.0, 0.05)
    f(np.array([0.0]), 1.0)
    out = f(np.array([1.0]), 1.0)  # dt=0
    assert np.all(np.isfinite(out))


def test_JointFilter_與reset():
    jf = JointFilter(1.0, 0.05)
    j = {k: np.array([1.0, 2.0, 3.0]) for k in JOINTS}
    out = jf(j, 0.0)
    assert set(out) == set(JOINTS)
    jf(j, 0.03)
    far = {k: np.array([5.0, 5.0, 5.0]) for k in JOINTS}
    smoothed = jf(far, 0.06)
    assert smoothed["nose"][0] < 5.0  # 有平滑
    jf.reset()
    np.testing.assert_allclose(jf(far, 0.09)["nose"], [5.0, 5.0, 5.0])  # reset 後原樣
