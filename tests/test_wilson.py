# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Tests for Wilson score interval implementation."""

import math

import numpy as np
import pytest

from reliable_cua.wilson import wilson_score_batch, wilson_score_interval


class TestWilsonScoreInterval:
    def test_known_value(self):
        """Verify against hand computation: R=10, k=7."""
        result = wilson_score_interval(7, 10)
        assert result.confidence_level == 0.95
        # Wilson center should be between p_hat=0.7 and 0.5
        assert 0.5 < result.point < 0.7
        assert result.lower < result.point < result.upper
        assert 0 <= result.lower
        assert result.upper <= 1

    def test_zero_successes(self):
        """k=0: lower should be 0, upper should be > 0."""
        result = wilson_score_interval(0, 3)
        assert result.lower == 0.0
        assert result.upper > 0.0
        # Wilson center is pulled away from 0
        assert result.point > 0.0

    def test_all_successes(self):
        """k=R: Wilson center should be < 1 (unlike Wald which gives 1.0)."""
        result = wilson_score_interval(3, 3)
        assert result.upper <= 1.0
        assert result.lower > 0.0
        # Wilson center is pulled away from 1 (shrinkage toward 0.5)
        assert result.point < 1.0

    def test_single_trial(self):
        """R=1: should produce valid interval."""
        result = wilson_score_interval(1, 1)
        assert 0 <= result.lower <= result.point <= result.upper <= 1

        result = wilson_score_interval(0, 1)
        assert 0 <= result.lower <= result.point <= result.upper <= 1

    def test_symmetry(self):
        """wilson(k, R) and wilson(R-k, R) should have mirrored intervals."""
        a = wilson_score_interval(3, 10)
        b = wilson_score_interval(7, 10)
        assert abs(a.point + b.point - 1.0) < 1e-10
        assert abs(a.lower + b.upper - 1.0) < 1e-10
        assert abs(a.upper + b.lower - 1.0) < 1e-10

    def test_large_r_approaches_wald(self):
        """At large R, Wilson should approximate Wald."""
        R = 10000
        k = 5000
        result = wilson_score_interval(k, R)
        # At R=10000, p_hat=0.5, Wald width ~ 2*1.96*sqrt(0.25/10000) ~ 0.0196
        assert abs(result.point - 0.5) < 0.01
        assert abs(result.width - 0.0196) < 0.005

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            wilson_score_interval(5, 0)
        with pytest.raises(ValueError):
            wilson_score_interval(-1, 5)
        with pytest.raises(ValueError):
            wilson_score_interval(6, 5)

    def test_different_confidence(self):
        ci_95 = wilson_score_interval(5, 10, 0.95)
        ci_99 = wilson_score_interval(5, 10, 0.99)
        assert ci_99.width > ci_95.width


class TestWilsonScoreBatch:
    def test_matches_scalar(self):
        """Batch should match scalar for each element."""
        successes = np.array([0, 3, 7, 10])
        trials = np.array([10, 10, 10, 10])
        centers, lowers, uppers = wilson_score_batch(successes, trials)

        for i in range(4):
            scalar = wilson_score_interval(int(successes[i]), int(trials[i]))
            assert abs(centers[i] - scalar.point) < 1e-10
            assert abs(lowers[i] - scalar.lower) < 1e-10
            assert abs(uppers[i] - scalar.upper) < 1e-10

    def test_output_bounds(self):
        centers, lowers, uppers = wilson_score_batch(
            np.array([0, 5, 10]), np.array([10, 10, 10])
        )
        assert np.all(lowers >= 0)
        assert np.all(uppers <= 1)
        assert np.all(lowers <= centers)
        assert np.all(centers <= uppers)
