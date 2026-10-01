"""Run physical regression checks; legacy source is archived."""

import unittest
from test_simulation import PhysicsTests

if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PhysicsTests)
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
