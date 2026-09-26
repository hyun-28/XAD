"""W2 rev1 E1 (§6): operator unit tests — input unchanged, identity bit-identical, recon donors clear
of the forbidden regions, constant operators have std 0, legacy equality, config == code metadata."""
import numpy as np
import pytest
import yaml

from src.config import REPO_ROOT
from src.perturb.operators import (META, NAMES, Ctx, NoDonorError, OperatorError, apply, context_len,
                                   exclusion, influence, recon_donor)
from src.perturbations import IDENTITY, PERTURBATIONS


def _setup(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 37) + 0.3 * np.sin(2 * np.pi * t / 9) + 0.1 * rng.standard_normal(n) + 3.0
    ctx = Ctx(train_end=1200, gt=(3000, 3060), w=32)
    return x, ctx


RUNS = [[(1800, 1808)], [(1800, 1832)], [(1500, 1516), (2200, 2232)], [(2500, 2640)]]


@pytest.mark.parametrize("name", NAMES)
def test_input_unchanged_and_only_runs_touched(name):
    x, ctx = _setup()
    for runs in RUNS:
        x0 = x.copy()
        y = apply(name, x, runs, np.random.default_rng(3), ctx)
        assert np.array_equal(x, x0)
        mask = np.zeros(x.size, bool)
        for a, b in runs:
            mask[a:b] = True
        assert np.array_equal(y[~mask], x[~mask])
        assert np.isfinite(y).all()


def test_identity_bit_identical():
    x, ctx = _setup()
    for runs in RUNS:
        y = apply("identity", x, runs, None, ctx)
        assert np.array_equal(y, x) and y is not x


@pytest.mark.parametrize("name", [k for k in NAMES if k.startswith("B")])
def test_single_run_equals_legacy_call(name):
    """One run: apply() IS the legacy function (src/perturbations.py), bit for bit."""
    x, ctx = _setup()
    for a, b in [(1800, 1808), (2500, 2640), (0, 10), (3990, 4000)]:
        legacy = PERTURBATIONS[name](x, np.arange(a, b), rng=np.random.default_rng(7))
        ours = apply(name, x, [(a, b)], np.random.default_rng(7), ctx)
        assert np.array_equal(ours, legacy)


@pytest.mark.parametrize("name", [k for k, v in META.items() if v["canonical_shape"] == "constant"])
def test_constant_operators_have_zero_std(name):
    """Std 0 in the exact sense stumpy uses (ptp == 0, core.py:_rolling_isconstant). np.std itself
    is not used: it recomputes the mean of identical values with rounding and can return 4e-16."""
    x, ctx = _setup()
    for runs in RUNS:
        y = apply(name, x, runs, None, ctx)
        for a, b in runs:
            seg = y[a:b]
            assert np.ptp(seg) == 0.0
            assert np.sqrt(np.mean((seg - seg[0]) ** 2)) == 0.0


@pytest.mark.parametrize("src", ["test", "train"])
def test_recon_donor_avoids_forbidden_regions(src):
    x, ctx = _setup()
    c = context_len(ctx.w)
    for runs in RUNS:
        for run in runs:
            u, _ = recon_donor(x, run, runs, ctx, src)
            L = run[1] - run[0]
            lo, hi = (ctx.train_end, x.size) if src == "test" else (0, ctx.train_end)
            assert lo <= u and u + L <= hi and u - c >= 0 and u + L + c <= x.size
            forbidden = [exclusion(ctx)] + [iv for ra, rb in runs
                                           for iv in (influence(ra, rb, ctx.w), (ra - c, ra), (rb, rb + c))]
            for f0, f1 in forbidden:
                assert not (u < f1 and f0 < u + L), (run, u, (f0, f1))


def test_recon_donor_is_the_best_admissible_match_and_copies_raw_values():
    x, ctx = _setup()
    run, c = (1800, 1832), context_len(ctx.w)
    u, d = recon_donor(x, run, [run], ctx, "test")
    q = np.r_[x[run[0] - c:run[0]], x[run[1]:run[1] + c]]
    L = run[1] - run[0]
    best = None
    for v in range(ctx.train_end, x.size - L + 1):
        if v - c < 0 or v + L + c > x.size:
            continue
        bad = [exclusion(ctx), influence(*run, ctx.w), (run[0] - c, run[0]), (run[1], run[1] + c)]
        if any(v < f1 and f0 < v + L for f0, f1 in bad):
            continue
        dv = float(((np.r_[x[v - c:v], x[v + L:v + L + c]] - q) ** 2).sum())
        if best is None or dv < best[1]:
            best = (v, dv)
    assert u == best[0] and abs(d - best[1]) < 1e-9
    y = apply("recon_test", x, [run], None, ctx)
    assert np.array_equal(y[run[0]:run[1]], x[u:u + L])


def test_recon_train_copies_a_reference_window():
    x, ctx = _setup()
    run = (1800, 1800 + ctx.w)
    u, _ = recon_donor(x, run, [run], ctx, "train")
    assert u + ctx.w <= ctx.train_end
    y = apply("recon_train", x, [run], None, ctx)
    assert np.array_equal(y[run[0]:run[1]], x[u:u + ctx.w])


def test_recon_reports_missing_donor():
    x, ctx = _setup()
    with pytest.raises(NoDonorError):
        recon_donor(x, (3995, 4000), [(3995, 4000)], ctx, "test")       # no right context
    small = Ctx(train_end=30, gt=ctx.gt, w=ctx.w)                        # 30 < c + L = 8 + 32
    with pytest.raises(NoDonorError):
        recon_donor(x, (1800, 1832), [(1800, 1832)], small, "train")    # training region too short


def test_stochastic_operators_need_a_generator_and_are_deterministic():
    x, ctx = _setup()
    for name in ("B5_gaussian", "B6_shuffle"):
        with pytest.raises(OperatorError):
            apply(name, x, [(1800, 1832)], None, ctx)
        a = apply(name, x, [(1800, 1832)], np.random.default_rng(5), ctx)
        b = apply(name, x, [(1800, 1832)], np.random.default_rng(5), ctx)
        assert np.array_equal(a, b)


def test_bad_runs_rejected():
    x, ctx = _setup()
    for runs in ([(10, 10)], [(-1, 5)], [(100, 120), (110, 130)], [(3990, 4001)]):
        with pytest.raises(OperatorError):
            apply("B1_zero", x, runs, None, ctx)


def test_config_matches_code():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "operators.yaml").read_text())["operators"]
    assert list(cfg) == list(META)
    for k, v in META.items():
        assert {f: cfg[k][f] for f in v} == v
    assert IDENTITY is not None


@pytest.mark.parametrize("band", [None, (2940, 3124)])      # segments span [2940, 3124)
@pytest.mark.parametrize("name", ["recon_test", "recon_train"])
def test_recon_cache_equals_apply(name, band):
    """E3 uses ReconCache for speed; it must give exactly the donors `apply` gives — with and without
    the memory band (E3 passes Ω)."""
    from src.perturb.operators import ReconCache, apply_cached
    x, ctx = _setup()
    cache = ReconCache(x, ctx, band=band)
    rng = np.random.default_rng(11)
    segs = [(s, s + 8) for s in range(2940, 3120, 8)]            # segments around the GT, like E3's Ω
    for _ in range(60):
        pick = sorted(rng.choice(len(segs), size=int(rng.integers(1, len(segs))), replace=False))
        runs, cur = [], None
        for k in pick:
            a, b = segs[k]
            if cur and cur[1] == a:
                cur = (cur[0], b)
            else:
                if cur:
                    runs.append(cur)
                cur = (a, b)
        runs.append(cur)
        try:
            want = apply(name, x, runs, None, ctx)
        except NoDonorError:
            with pytest.raises(NoDonorError):
                apply_cached(name, x, runs, None, ctx, cache)
            continue
        assert np.array_equal(apply_cached(name, x, runs, None, ctx, cache), want)


def test_recon_cache_band_bounds_memory_and_rejects_outside_runs():
    from src.perturb.operators import ReconCache, apply_cached
    x, ctx = _setup()
    full, banded = ReconCache(x, ctx), ReconCache(x, ctx, band=(2940, 3124))
    run = (3000, 3024)
    assert full.donor(run, [run], "test") == banded.donor(run, [run], "test")
    n_full = full._order[("test", run)][0].size
    n_band = banded._order[("test", run)][0].size
    assert n_band < n_full and n_band <= n_full
    with pytest.raises(OperatorError):
        apply_cached("recon_test", x, [(2500, 2510)], None, ctx, banded)


def test_order_d35_ties_random_lerf_reverse_and_abs():
    from src.metrics.faithfulness_ad import order, order_d35, tie_rank
    a = np.array([0.0, 2.0, 0.0, -1.0, 2.0, 0.0])
    ties = tie_rank(a.size, np.random.default_rng(3))
    m = order_d35(a, ties, "MoRF")
    assert list(a[m]) == sorted(a, reverse=True)                     # signed descending
    assert np.array_equal(order_d35(a, ties, "LeRF"), m[::-1])        # exact reverse
    for grp in (np.flatnonzero(a == 2.0), np.flatnonzero(a == 0.0)):  # ties follow the random rank
        pos = [int(np.flatnonzero(m == i)[0]) for i in grp]
        assert [grp[k] for k in np.argsort(pos)] == list(grp[np.argsort(ties[grp])])
    ma = order_d35(a, ties, "MoRF", key="abs")
    assert list(np.abs(a)[ma]) == sorted(np.abs(a), reverse=True)
    distinct = np.array([0.3, -0.2, 0.9, 0.1])                        # no ties: same as the index version
    t2 = tie_rank(4, np.random.default_rng(0))
    assert np.array_equal(order_d35(distinct, t2, "MoRF"), order(distinct, "MoRF"))
    assert np.array_equal(order_d35(distinct, t2, "LeRF"), order(distinct, "LeRF"))
