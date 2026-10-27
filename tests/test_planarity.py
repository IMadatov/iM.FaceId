import numpy as np
import pytest

from app.pipeline.planarity import (
    max_planarity_residual,
    planarity_residual,
    view_change,
    view_change_spread,
)

from tests.synthetic_face import face_model as _face_model
from tests.synthetic_face import project as _project

# InsightFace kps order: 0=left eye, 1=right eye, 2=nose, 3=left mouth, 4=right mouth.
# The four outer points sit on the face plane; only the nose protrudes.


def test_flat_face_turning_has_near_zero_residual():
    """A photo/screen is planar, so one homography maps every point exactly."""
    flat = _face_model(nose_protrusion=0.0)
    residual = planarity_residual(_project(flat, 0.0), _project(flat, 25.0))
    assert residual < 0.01


def test_protruding_nose_produces_large_residual():
    """A real nose sits ~25mm off the face plane, so it breaks the homography."""
    real = _face_model(nose_protrusion=25.0)
    residual = planarity_residual(_project(real, 0.0), _project(real, 25.0))
    assert residual > 0.10


def test_residual_is_scale_invariant():
    """Normalising by interocular distance keeps the score independent of face size."""
    real = _face_model(nose_protrusion=25.0)
    small = planarity_residual(_project(real, 0.0), _project(real, 25.0))
    large = planarity_residual(
        _project(real, 0.0, f=1800.0), _project(real, 25.0, f=1800.0)
    )
    assert large == pytest.approx(small, rel=0.05)


def test_frontal_pair_without_rotation_cannot_separate_flat_from_real():
    """Guards the motion gate: no yaw means no parallax, even for a real face."""
    real = _face_model(nose_protrusion=25.0)
    residual = planarity_residual(_project(real, 0.0), _project(real, 0.0))
    assert residual < 0.01


def test_max_planarity_residual_finds_parallax_in_a_single_pair():
    """One strongly-rotated frame is enough to prove the subject is 3D."""
    real = _face_model(nose_protrusion=25.0)
    frames = [_project(real, 0.0), _project(real, 2.0), _project(real, 30.0)]
    assert max_planarity_residual(frames) > 0.10


def test_max_planarity_residual_stays_low_for_a_replayed_photo():
    """Every pair of a flat sequence is homography-explained, so the max is still ~0."""
    flat = _face_model(nose_protrusion=0.0)
    frames = [_project(flat, 0.0), _project(flat, 12.0), _project(flat, 30.0)]
    assert max_planarity_residual(frames) < 0.01


def test_max_planarity_residual_requires_two_frames():
    with pytest.raises(ValueError):
        max_planarity_residual([_project(_face_model(0.0), 0.0)])


def test_view_change_spread_is_blind_to_whether_the_face_is_3d():
    """The gate must not double as a 3D detector, or it rejects spoofs as 'no movement'."""
    yaws = [-20.0, 0.0, 20.0]
    real = [_project(_face_model(25.0), y) for y in yaws]
    flat = [_project(_face_model(0.0), y) for y in yaws]
    assert view_change_spread(real) == pytest.approx(view_change_spread(flat), rel=1e-6)


def test_view_change_spread_grows_with_rotation():
    real = _face_model(nose_protrusion=25.0)
    turned = view_change_spread([_project(real, -20.0), _project(real, 20.0)])
    still = view_change_spread([_project(real, 0.0), _project(real, 1.0)])
    assert turned > 0.05
    assert still < 0.01


def test_view_change_spread_is_independent_of_camera_distance():
    """The gate must read the turn, not how close the camera is.

    ``view_change`` gates ``planarity_residual``, and that residual barely moves
    with distance. A gate that shrinks as the subject steps back therefore
    rejects captures whose parallax evidence is perfectly good.
    """
    real = _face_model(25.0)
    near = view_change_spread(
        [_project(real, 0.0, dist=400.0), _project(real, 15.0, dist=400.0)]
    )
    far = view_change_spread(
        [_project(real, 0.0, dist=1500.0), _project(real, 15.0, dist=1500.0)]
    )
    assert far == pytest.approx(near, rel=0.05)


def test_view_change_spread_registers_a_left_right_alternation():
    """Two opposite turns are mirror images, so an aspect-only read sees nothing.

    Filming a liveness check as a left/right alternation never passes through
    frontal, which is why the spread has to fall back on the signed perspective
    cue for this shape of capture.
    """
    real = _face_model(25.0)
    frames = [_project(real, -25.0), _project(real, 25.0)]
    assert view_change(frames[0]) == pytest.approx(view_change(frames[1]), rel=1e-4)
    assert view_change_spread(frames) > 0.05


def test_view_change_spread_registers_a_turn_at_arms_length():
    """A 20-degree turn at 1m must read as far more motion than a 2-degree wobble."""
    real = _face_model(25.0)
    turned = view_change_spread(
        [_project(real, 0.0, dist=1000.0), _project(real, 20.0, dist=1000.0)]
    )
    wobble = view_change_spread(
        [_project(real, 0.0, dist=1000.0), _project(real, 2.0, dist=1000.0)]
    )
    assert turned > 0.05
    assert turned > 10 * wobble


def test_view_change_peaks_at_frontal():
    """Yaw in either direction foreshortens the face, so frontal is the extreme.

    The absolute value is a per-subject baseline (it encodes face proportions),
    so only its *spread* across frames is meaningful — this pins the direction
    of change rather than any particular value.
    """
    real = _face_model(25.0)
    frontal = view_change(_project(real, 0.0))
    assert frontal > view_change(_project(real, 20.0))
    assert frontal > view_change(_project(real, -20.0))
