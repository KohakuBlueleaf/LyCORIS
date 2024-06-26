import unittest
import coverage

cov = coverage.Coverage()
cov.start()

from test.module import LycorisModuleTests


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(LycorisModuleTests)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    cov.stop()
    cov.save()
    cov.report()
    cov.html_report(directory="coverage_report")
