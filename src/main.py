import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
