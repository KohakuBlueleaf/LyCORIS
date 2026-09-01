import unittest
import logging
import coverage

cov = coverage.Coverage()
cov.start()

from lycoris.logging import logger

logger.setLevel(logging.ERROR)

from test.module import LycorisModuleTests
from test.wrapper import LycorisWrapperTests
from test.functional import LycorisFunctionalTests
from test.kohya import LycorisKohyaWrapperTests
from test.precision_merge_test import MergePrecisionTests
from test.kernels.test_ops import OpsVsFp64
from test.kernels.test_autograd import AutogradParity, SafeFallback

# The kernel suites skip themselves without CUDA.
TESTS = [
    LycorisModuleTests,
    LycorisFunctionalTests,
    LycorisWrapperTests,
    LycorisKohyaWrapperTests,
    MergePrecisionTests,
    OpsVsFp64,
    AutogradParity,
    SafeFallback,
]


if __name__ == "__main__":
    test_loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=0)
    for test in TESTS:
        suite = test_loader.loadTestsFromTestCase(test)
        result = runner.run(suite)

    cov.stop()
    cov.save()
    cov.report()
    cov.html_report(directory="coverage_report")
