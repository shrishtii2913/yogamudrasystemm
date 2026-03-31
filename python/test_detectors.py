"""
=============================================================================
test_detectors.py — Unit Tests for Mudra Detector Logic
=============================================================================
Tests each mudra detector function in isolation using synthetic landmark data.
No camera or MediaPipe installation required to run these tests.

Run:
    python3 test_detectors.py
=============================================================================
"""

import math
import unittest
from types import SimpleNamespace


# ── Replicate the geometry helpers from pose_detector.py ──────────────────────

def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def finger_extended(tip, mcp):
    return tip.y < mcp.y

def P(x, y, z=0.0):
    """Create a synthetic landmark."""
    return SimpleNamespace(x=x, y=y, z=z)

def make_flat_hand():
    """
    21-landmark list where all fingers are fully extended (tips higher than MCPs).
    Thumb extends to the left (tip.x < IP.x for right hand).
    """
    lm = [None] * 21
    # Wrist
    lm[0] = P(0.5, 0.9)

    # Thumb: 1=CMC 2=MCP 3=IP 4=TIP
    lm[1] = P(0.42, 0.85); lm[2] = P(0.36, 0.80)
    lm[3] = P(0.30, 0.76); lm[4] = P(0.24, 0.73)   # tip.x < IP.x ✓

    # Index: 5=MCP 6=PIP 7=DIP 8=TIP
    lm[5] = P(0.50, 0.75); lm[6] = P(0.50, 0.60)
    lm[7] = P(0.50, 0.48); lm[8] = P(0.50, 0.36)   # tip.y < mcp.y ✓

    # Middle: 9=MCP 10=PIP 11=DIP 12=TIP
    lm[9]  = P(0.55, 0.74); lm[10] = P(0.55, 0.58)
    lm[11] = P(0.55, 0.46); lm[12] = P(0.55, 0.33)

    # Ring: 13=MCP 14=PIP 15=DIP 16=TIP
    lm[13] = P(0.60, 0.75); lm[14] = P(0.60, 0.60)
    lm[15] = P(0.60, 0.48); lm[16] = P(0.60, 0.38)

    # Pinky: 17=MCP 18=PIP 19=DIP 20=TIP
    lm[17] = P(0.65, 0.77); lm[18] = P(0.65, 0.64)
    lm[19] = P(0.65, 0.53); lm[20] = P(0.65, 0.44)

    return lm

def make_fist():
    """
    21-landmark list where all fingers are curled (tips below MCPs).
    """
    lm = make_flat_hand()
    # Curl each fingertip below its MCP
    for tip_idx, mcp_idx in [(8,5),(12,9),(16,13),(20,17)]:
        lm[tip_idx] = P(lm[mcp_idx].x, lm[mcp_idx].y + 0.05)
    return lm


# ── Detector functions (copied verbatim from pose_detector.py) ────────────────

def detect_gyan_mudra(lm):
    d = dist(lm[4], lm[8])
    threshold = 0.06
    middle_ext = finger_extended(lm[12], lm[9])
    ring_ext   = finger_extended(lm[16], lm[13])
    pinky_ext  = finger_extended(lm[20], lm[17])
    if d < threshold and middle_ext and ring_ext and pinky_ext:
        return True, round(1.0 - (d / threshold), 3)
    return False, 0.0

def detect_abhaya_mudra(lm):
    tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
    mcps = [lm[2], lm[5], lm[9],  lm[13], lm[17]]
    all_extended = all(finger_extended(t, m) for t, m in zip(tips[1:], mcps[1:]))
    thumb_ext    = lm[4].x < lm[3].x
    if all_extended and thumb_ext:
        magnitudes = [abs(t.y - m.y) for t, m in zip(tips[1:], mcps[1:])]
        return True, round(min(sum(magnitudes) / len(magnitudes) * 5, 1.0), 3)
    return False, 0.0

def detect_dhyana_mudra(lm):
    tips = [lm[8], lm[12], lm[16], lm[20]]
    mcps = [lm[5], lm[9],  lm[13], lm[17]]
    all_curled = all(not finger_extended(t, m) for t, m in zip(tips, mcps))
    if all_curled:
        depths = [abs(t.y - m.y) for t, m in zip(tips, mcps)]
        return True, round(min(sum(depths) / len(depths) * 4, 1.0), 3)
    return False, 0.0

def detect_shuni_mudra(lm):
    d = dist(lm[4], lm[12])
    threshold = 0.06
    index_ext = finger_extended(lm[8],  lm[5])
    ring_ext  = finger_extended(lm[16], lm[13])
    pinky_ext = finger_extended(lm[20], lm[17])
    if d < threshold and index_ext and ring_ext and pinky_ext:
        return True, round(1.0 - (d / threshold), 3)
    return False, 0.0


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestGyanMudra(unittest.TestCase):

    def test_detected_when_thumb_index_close(self):
        lm = make_flat_hand()
        lm[8] = P(lm[4].x + 0.02, lm[4].y + 0.02)   # index tip near thumb tip
        matched, conf = detect_gyan_mudra(lm)
        self.assertTrue(matched, "Gyan Mudra should be detected")
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_not_detected_when_far_apart(self):
        lm = make_flat_hand()
        # index tip is at (0.5, 0.36) and thumb at (0.24, 0.73) — far apart
        matched, conf = detect_gyan_mudra(lm)
        self.assertFalse(matched, "Gyan Mudra should NOT be detected when fingers far apart")
        self.assertEqual(conf, 0.0)

    def test_confidence_is_higher_when_closer(self):
        lm_close = make_flat_hand()
        lm_far   = make_flat_hand()
        # Very close touch
        lm_close[8] = P(lm_close[4].x + 0.01, lm_close[4].y)
        # Near-threshold touch
        lm_far[8]   = P(lm_far[4].x + 0.045, lm_far[4].y)
        _, c_close = detect_gyan_mudra(lm_close)
        _, c_far   = detect_gyan_mudra(lm_far)
        self.assertGreater(c_close, c_far,
            "Confidence should be higher when fingertips are closer")


class TestAbhayaMudra(unittest.TestCase):

    def test_detected_with_open_palm(self):
        lm = make_flat_hand()
        matched, conf = detect_abhaya_mudra(lm)
        self.assertTrue(matched, "Abhaya Mudra should fire on open palm")
        self.assertGreater(conf, 0.0)

    def test_not_detected_with_fist(self):
        lm = make_fist()
        matched, _ = detect_abhaya_mudra(lm)
        self.assertFalse(matched, "Abhaya Mudra should NOT fire for a fist")


class TestDhyanaMudra(unittest.TestCase):

    def test_detected_with_fist(self):
        lm = make_fist()
        matched, conf = detect_dhyana_mudra(lm)
        self.assertTrue(matched, "Dhyana Mudra should fire for a closed fist")
        self.assertGreater(conf, 0.0)

    def test_not_detected_with_open_palm(self):
        lm = make_flat_hand()
        matched, _ = detect_dhyana_mudra(lm)
        self.assertFalse(matched, "Dhyana Mudra should NOT fire for an open palm")


class TestShuniMudra(unittest.TestCase):

    def test_detected_when_thumb_middle_close(self):
        lm = make_flat_hand()
        # Place middle tip near thumb tip
        lm[12] = P(lm[4].x + 0.02, lm[4].y + 0.02)
        matched, conf = detect_shuni_mudra(lm)
        self.assertTrue(matched, "Shuni Mudra should be detected")
        self.assertGreater(conf, 0.0)


class TestGeometryHelpers(unittest.TestCase):

    def test_dist_zero(self):
        p = P(0.5, 0.5)
        self.assertAlmostEqual(dist(p, p), 0.0)

    def test_dist_known(self):
        p1, p2 = P(0.0, 0.0), P(0.3, 0.4)
        self.assertAlmostEqual(dist(p1, p2), 0.5, places=5)

    def test_finger_extended_true(self):
        tip = P(0.5, 0.3); mcp = P(0.5, 0.7)
        self.assertTrue(finger_extended(tip, mcp))

    def test_finger_extended_false(self):
        tip = P(0.5, 0.8); mcp = P(0.5, 0.7)
        self.assertFalse(finger_extended(tip, mcp))


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Yoga Mudra — Detector Unit Tests")
    print("=" * 60)
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(__import__("__main__"))
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    print("=" * 60)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED ✓")
    else:
        print(f"  {len(result.failures)} failure(s), {len(result.errors)} error(s)")
    print("=" * 60)
