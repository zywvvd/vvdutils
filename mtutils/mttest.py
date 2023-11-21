import pytest
import os
from pathlib import Path

def mtutils_test():
    mtutils_dir = Path(__file__).parent
    os.chdir(mtutils_dir)
    pytest.main(['tests'])