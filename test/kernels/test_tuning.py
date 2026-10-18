"""The config timing switch: ``lycoris.kernels.set_tuning`` over
``LYCORIS_KERNEL_TUNE``, and what both tuners do with timing off and on.

CPU only: the timing itself is replaced by fixed per-config times.

Usage:
    python -m unittest test.kernels.test_tuning -v
"""

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from parameterized import parameterized

import lycoris.kernels
from lycoris.kernels import set_tuning, tuning_enabled
from lycoris.kernels.plans import tune
from lycoris.kernels.plans.cost import TilePlan
from lycoris.kernels.tuner import DeviceInfo, Tuner

ENV = "LYCORIS_KERNEL_TUNE"
MODES = [("set_tuning",), ("env",)]

SLOW = TilePlan(16, 64, 0, 4, 1, 1.0, "dram")
FAST = TilePlan(32, 64, 0, 4, 1, 2.0, "dram")
MS = {SLOW: 2.0, FAST: 1.0}


@contextlib.contextmanager
def _switch(mode: str, on: bool):
    """Timing on or off through set_tuning, or through the env var alone."""
    if mode == "set_tuning":
        prev = set_tuning(on)
        try:
            yield
        finally:
            set_tuning(prev)
    else:
        prev = set_tuning(None)
        try:
            with mock.patch.dict(os.environ, {ENV: "on" if on else "off"}):
                yield
        finally:
            set_tuning(prev)


class SetTuning(unittest.TestCase):
    def setUp(self):
        self.prev = set_tuning(None)

    def tearDown(self):
        set_tuning(self.prev)

    def test_exported(self):
        self.assertIs(lycoris.kernels.set_tuning, tune.set_tuning)
        self.assertIs(lycoris.kernels.tuning_enabled, tune.tuning_enabled)
        self.assertIn("set_tuning", lycoris.kernels.__all__)

    def test_env_is_the_initial_value(self):
        for value, want in (
            ("on", True),
            ("1", True),
            ("off", False),
            ("OFF", False),
            ("0", False),
        ):
            with mock.patch.dict(os.environ, {ENV: value}):
                self.assertIs(tuning_enabled(), want, value)
        with mock.patch.dict(os.environ):
            os.environ.pop(ENV, None)
            self.assertTrue(tuning_enabled())

    def test_overrides_env_and_restores(self):
        with mock.patch.dict(os.environ, {ENV: "on"}):
            prev = set_tuning(False)
            self.assertIsNone(prev)
            self.assertFalse(tuning_enabled())
            inner = set_tuning(True)
            self.assertIs(inner, False)
            os.environ[ENV] = "off"
            self.assertTrue(tuning_enabled())
            self.assertIs(set_tuning(inner), True)
            self.assertFalse(tuning_enabled())
            self.assertIs(set_tuning(prev), False)
            # None follows the env var again, read on each lookup.
            self.assertFalse(tuning_enabled())
            os.environ[ENV] = "on"
            self.assertTrue(tuning_enabled())

    def test_rejects_non_bool(self):
        set_tuning(False)
        for bad in ("off", "on", 0, 1):
            with self.assertRaises(TypeError):
                set_tuning(bad)
            self.assertFalse(tuning_enabled())


class PlansTuned(unittest.TestCase):
    """plans.tune.tuned: untimed first picks stay out of the table and are
    timed once timing is on."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.addCleanup(self.dir.cleanup)
        stack.enter_context(
            mock.patch.dict(os.environ, {"LYCORIS_KERNEL_CACHE_DIR": self.dir.name})
        )
        stack.enter_context(mock.patch.dict(tune._CACHE, clear=True))
        stack.enter_context(mock.patch.dict(tune._UNTIMED, clear=True))
        stack.enter_context(mock.patch.object(tune, "_LOADED", False))
        stack.enter_context(
            mock.patch.object(
                tune, "resolve_device", return_value=SimpleNamespace(name="dev")
            )
        )
        self.timed = []
        stack.enter_context(mock.patch.object(tune, "_time", side_effect=self._time))
        self.listed = 0

    def _time(self, plan):
        self.timed.append(plan)
        return MS[plan]

    def _shortlist(self):
        self.listed += 1
        return [SLOW, FAST]

    def _tuned(self, key):
        return tune.tuned("k", key, self._shortlist, lambda plan: plan)

    def _stored(self) -> list:
        path = Path(self.dir.name) / "tuning.json"
        if not path.exists():
            return []
        return [tuple(row["key"]) for row in json.loads(path.read_text("utf-8"))]

    @parameterized.expand(MODES)
    def test_off_takes_the_first_pick_untimed(self, mode):
        with _switch(mode, False):
            self.assertEqual(self._tuned((1,)), SLOW)
            self.assertEqual(self._tuned((1,)), SLOW)
        self.assertEqual(self.timed, [])
        self.assertEqual(self.listed, 1)
        self.assertEqual(self._stored(), [])

    @parameterized.expand(MODES)
    def test_on_times_a_key_picked_untimed(self, mode):
        with _switch(mode, False):
            self.assertEqual(self._tuned((1,)), SLOW)
        with _switch(mode, True):
            self.assertEqual(self._tuned((1,)), FAST)
        self.assertEqual(self.timed, [SLOW, FAST])
        self.assertEqual(self._stored(), [("dev", "k", 1)])
        # Timed and stored: off again reads the timed pick.
        with _switch(mode, False):
            self.assertEqual(self._tuned((1,)), FAST)

    @parameterized.expand(MODES)
    def test_untimed_picks_are_never_stored(self, mode):
        with _switch(mode, False):
            self._tuned((1,))
        with _switch(mode, True):
            self._tuned((2,))
        self.assertEqual(self._stored(), [("dev", "k", 2)])


class TunerSelect(unittest.TestCase):
    """The model-ranked tuner: the model pick with timing off, timed and stored
    once timing is on."""

    DEV = DeviceInfo(
        kind="test",
        arch="test",
        name="test-device",
        index=0,
        units=1,
        capability=None,
        l2_bytes=1,
        smem=1,
        bf16=False,
        mma=False,
    )
    RANKED = [{"block": 16, "num_warps": 4}, {"block": 32, "num_warps": 4}]
    KEY = ("op", ("float16",), (64,))

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        env = mock.patch.dict(os.environ, {"LYCORIS_KERNEL_CACHE_DIR": self.dir.name})
        env.start()
        self.addCleanup(env.stop)
        self.tuner = Tuner()
        self.measured = []

    def _measure(self, configs):
        self.measured.append(configs)
        return [1.0 if c["block"] == 32 else 2.0 for c in configs]

    def _select(self):
        return self.tuner.select(
            self.DEV, self.KEY, lambda: list(self.RANKED), self._measure
        )

    def _stored(self) -> list:
        path = Path(self.dir.name) / "kernels" / "test-test-device.json"
        if not path.exists():
            return []
        return [row["config"] for row in json.loads(path.read_text("utf-8"))["entries"]]

    @parameterized.expand(MODES)
    def test_off_then_on(self, mode):
        with _switch(mode, False):
            self.assertEqual(self._select(), self.RANKED[0])
            self.assertEqual(self._select(), self.RANKED[0])
        self.assertEqual(self.measured, [])
        self.assertEqual(self._stored(), [])
        with _switch(mode, True):
            self.assertEqual(self._select(), self.RANKED[1])
        self.assertEqual(len(self.measured), 1)
        self.assertEqual(self._stored(), [self.RANKED[1]])
        with _switch(mode, False):
            self.assertEqual(self._select(), self.RANKED[1])
        self.assertEqual(len(self.measured), 1)


if __name__ == "__main__":
    unittest.main()
