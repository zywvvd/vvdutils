from .lib.loader import try_to_import

import os
from pathlib import Path

try_to_import('pytest', 'pip install pytest to run mtutils_test')
import pytest

def mtutils_test():
    mtutils_dir = Path(__file__).parent
    os.chdir(mtutils_dir)
    pytest.main(['tests'])