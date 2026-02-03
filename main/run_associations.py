import os
import sys

# ======================================================
# PATH SETUP
# ======================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.association_rules.train_fpgrowth_spark import train_fpgrowth_spark

# ======================================================
# RUN ASSOCIATION RULES
# ======================================================
train_fpgrowth_spark()