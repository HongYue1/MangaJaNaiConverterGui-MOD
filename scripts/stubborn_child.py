"""A worker that ignores 'cancel' - stands in for one stuck in a forward pass.

It never reads stdin and never exits on its own inside the test window, so the
only thing that can stop it is the GUI's kill backstop.
"""

import sys
import time

sys.stdout.write('{"type": "log", "level": "info", "message": "stubborn worker up"}\n')
sys.stdout.flush()
time.sleep(120)
