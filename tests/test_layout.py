import pytest

from core.layout import (LayoutSettings, PathInfo, SlotInfo, VariantInfo, layout_path,
                         POINT, SPAN)


def slot(key, length=1.0, minx=None, **kw):
    if minx is None:
        minx = -length / 2.0
    return SlotInfo(key, variants=[VariantInfo(minx, minx + length)], **kw)


def spans(res, key):
    return [p for p in res.placements if p.slot == key]


def covered(placements):
    return sum(p.s_end - p.s_begin for p in placements)


def test_default_adaptive_round_fills_exactly():
    res = layout_path(PathInfo(10.3), {"default": slot("default", 2.0)}, LayoutSettings())
    ds = spans(res, "default")
    assert len(ds) == 5
    assert ds[0].s_begin == pytest.approx(0.0)
    assert ds[-1].s_end == pytest.approx(10.3)
    assert all(p.k == pytest.approx(10.3 / 10.0) for p in ds)


@pytest.mark.parametrize("mode,count", [("CEIL", 6), ("FLOOR", 5), ("ROUND", 5)])
def test_fit_modes(mode, count):
    res = layout_path(PathInfo(10.3), {"default": slot("default", 2.0)}, LayoutSettings(fit_mode=mode))
    assert len(spans(res, "default")) == count


def test_count_mode_and_spacing():
    res = layout_path(PathInfo(10.0), {"default": slot("default", 2.0)},
                      LayoutSettings(fit_mode="COUNT", fit_count=4, spacing=0.4))
    ds = spans(res, "default")
    assert len(ds) == 4
    assert ds[1].s_begin - ds[0].s_end == pytest.approx(0.4)
    assert ds[-1].s_end == pytest.approx(10.0)


def test_start_end_take_the_ends():
    slots = {"default": slot("default", 2.0), "start": slot("start", 0.5), "end": slot("end", 0.25)}
    res = layout_path(PathInfo(10.0), slots, LayoutSettings())
    st, en = spans(res, "start")[0], spans(res, "end")[0]
    assert (st.s_begin, st.s_end) == pytest.approx((0.0, 0.5))
    assert (en.s_begin, en.s_end) == pytest.approx((9.75, 10.0))
    ds = spans(res, "default")
    assert ds[0].s_begin == pytest.approx(0.5)
    assert ds[-1].s_end == pytest.approx(9.75)


def test_clipping():
    res = layout_path(PathInfo(10.0), {"default": slot("default", 1.0)},
                      LayoutSettings(clip_start=1.0, clip_end=2.0))
    ds = spans(res, "default")
    assert ds[0].s_begin == pytest.approx(1.0)
    assert ds[-1].s_end == pytest.approx(8.0)


def test_corners_break_sections_and_place_corner_samples():
    slots = {"default": slot("default", 1.0), "corner": slot("corner", 0.2)}
    res = layout_path(PathInfo(8.0, corners=[4.0]), slots, LayoutSettings())
    cs = spans(res, "corner")
    assert len(cs) == 1 and cs[0].kind == POINT and cs[0].anchor == pytest.approx(4.0)
    ds = spans(res, "default")
    # No default module crosses the corner footprint [3.9, 4.1].
    assert all(p.s_end <= 3.9 + 1e-9 or p.s_begin >= 4.1 - 1e-9 for p in ds)
    assert covered(ds) == pytest.approx(8.0 - 0.2)


def test_corner_without_sample_still_breaks():
    res = layout_path(PathInfo(5.0, corners=[2.5]), {"default": slot("default", 2.0)}, LayoutSettings())
    ds = spans(res, "default")
    assert any(p.s_end == pytest.approx(2.5) for p in ds)


def test_corner_slide_moves_the_break():
    res = layout_path(PathInfo(8.0, corners=[4.0]), {"default": slot("default", 1.0)},
                      LayoutSettings(corner_slide=-0.5))
    assert any(p.s_end == pytest.approx(3.5) for p in spans(res, "default"))


def test_evenly_spacing_with_min_gap():
    slots = {"default": slot("default", 1.0), "evenly": slot("evenly", 0.1)}
    st = LayoutSettings(evenly_mode="SPACING", evenly_spacing=3.0, evenly_min_gap=1.5)
    res = layout_path(PathInfo(10.0), slots, st)
    ev = [p.anchor for p in spans(res, "evenly")]
    assert ev == pytest.approx([3.0, 6.0])  # 9.0 would leave a gap < 1.5 to the end
    st = LayoutSettings(evenly_mode="SPACING", evenly_spacing=3.0, evenly_min_gap=0.5)
    ev = [p.anchor for p in spans(layout_path(PathInfo(10.0), slots, st), "evenly")]
    assert ev == pytest.approx([3.0, 6.0, 9.0])


def test_evenly_fit_and_count():
    slots = {"default": slot("default", 1.0), "evenly": slot("evenly", 0.1)}
    ev = spans(layout_path(PathInfo(10.0), slots, LayoutSettings(evenly_mode="FIT", evenly_spacing=3.0)), "evenly")
    assert [p.anchor for p in ev] == pytest.approx([2.5, 5.0, 7.5])
    ev = spans(layout_path(PathInfo(10.0), slots, LayoutSettings(evenly_mode="COUNT", evenly_count=1)), "evenly")
    assert [p.anchor for p in ev] == pytest.approx([5.0])


def test_evenly_is_per_section_between_corners():
    slots = {"default": slot("default", 1.0), "evenly": slot("evenly", 0.0)}
    st = LayoutSettings(evenly_mode="COUNT", evenly_count=1)
    ev = spans(layout_path(PathInfo(10.0, corners=[4.0]), slots, st), "evenly")
    assert [p.anchor for p in ev] == pytest.approx([2.0, 7.0])


def test_markers_place_samples_and_break():
    slots = {"default": slot("default", 1.0), "marker:0": slot("marker:0", 1.2)}
    res = layout_path(PathInfo(10.0, markers=[(5.0, "marker:0")]), slots, LayoutSettings())
    ms = spans(res, "marker:0")
    assert len(ms) == 1 and ms[0].anchor == pytest.approx(5.0)
    ds = spans(res, "default")
    assert all(p.s_end <= 4.4 + 1e-9 or p.s_begin >= 5.6 - 1e-9 for p in ds)


def test_segment_ids_switch_sample_and_allow_holes():
    bounds = [(0.0, 3.0, -1), (3.0, 6.0, 0), (6.0, 9.0, 1)]
    slots = {"default": slot("default", 1.0), "seg:0": slot("seg:0", 0.5),
             "seg:1": SlotInfo("seg:1", variants=[])}  # empty sample -> hole
    res = layout_path(PathInfo(9.0, seg_bounds=bounds), slots, LayoutSettings())
    assert covered(spans(res, "default")) == pytest.approx(3.0)
    seg0 = spans(res, "seg:0")
    assert len(seg0) == 6 and covered(seg0) == pytest.approx(3.0)
    assert not spans(res, "seg:1")


def test_fixed_mode_slice_and_scale():
    st = LayoutSettings(fit_mode="FIXED", fixed_remainder="SLICE")
    res = layout_path(PathInfo(5.5), {"default": slot("default", 2.0)}, st)
    ds = spans(res, "default")
    assert len(ds) == 3
    assert ds[-1].slice_hi == pytest.approx(0.75)
    assert ds[-1].s_end == pytest.approx(5.5)
    st = LayoutSettings(fit_mode="FIXED", fixed_remainder="SCALE", fixed_align="END")
    ds = spans(layout_path(PathInfo(5.5), {"default": slot("default", 2.0)}, st), "default")
    assert ds[0].s_begin == pytest.approx(0.0) and ds[0].k == pytest.approx(0.75)
    st = LayoutSettings(fit_mode="FIXED", fixed_align="CENTER")
    ds = spans(layout_path(PathInfo(5.0), {"default": slot("default", 2.0)}, st), "default")
    assert ds[0].s_begin == pytest.approx(0.5) and ds[-1].s_end == pytest.approx(4.5)


def test_variants_sequence_and_random_are_deterministic():
    sl = SlotInfo("default", variants=[VariantInfo(0, 1), VariantInfo(0, 1), VariantInfo(0, 1)],
                  pick="SEQUENCE")
    res = layout_path(PathInfo(6.0), {"default": sl}, LayoutSettings())
    assert [p.variant for p in res.placements] == [0, 1, 2, 0, 1, 2]
    sl.pick = "RANDOM"
    a = [p.variant for p in layout_path(PathInfo(30.0), {"default": sl}, LayoutSettings(seed=3)).placements]
    b = [p.variant for p in layout_path(PathInfo(30.0), {"default": sl}, LayoutSettings(seed=3)).placements]
    c = [p.variant for p in layout_path(PathInfo(30.0), {"default": sl}, LayoutSettings(seed=4)).placements]
    assert a == b and a != c and set(a) == {0, 1, 2}


def test_weighted_random_respects_zero_weight():
    sl = SlotInfo("default", variants=[VariantInfo(0, 1), VariantInfo(0, 1)], pick="RANDOM",
                  weights=[1.0, 0.0])
    res = layout_path(PathInfo(50.0), {"default": sl}, LayoutSettings())
    assert {p.variant for p in res.placements} == {0}


def test_cyclic_path_with_corners():
    slots = {"default": slot("default", 1.0), "corner": slot("corner", 0.2)}
    res = layout_path(PathInfo(8.0, cyclic=True, corners=[0.0, 2.0, 4.0, 6.0]), slots,
                      LayoutSettings(clip_start=1.0))  # clipping is ignored on loops
    assert len(spans(res, "corner")) == 4
    assert covered(spans(res, "default")) == pytest.approx(8.0 - 0.8)


def test_max_modules_truncates():
    res = layout_path(PathInfo(1000.0), {"default": slot("default", 0.1)}, LayoutSettings(max_modules=50))
    assert res.truncated and len(res.placements) == 50


def test_paddings_extend_footprint():
    s = slot("default", 1.0, pad_before=0.25, pad_after=0.25)
    res = layout_path(PathInfo(6.0), {"default": s}, LayoutSettings())
    ds = spans(res, "default")
    assert len(ds) == 4
    assert ds[0].x0 == pytest.approx(0.25)
    assert ds[0].k == pytest.approx(1.0)
    assert all(p.kind == SPAN for p in ds)
