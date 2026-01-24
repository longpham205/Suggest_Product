import os
import sys

# ======================================================
# PATH SETUP
# ======================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.clustering.behavior.behavior_train import train_behavior_clustering
from src.clustering.preference.preference_train import train_preference_clustering
from src.clustering.lifecycle.assign_lifecycle import assign_lifecycle

# ======================================================
# RUN BEHAVIOR
# ======================================================
train_behavior_clustering()

# ======================================================
# RUN CLUSTERING
# ======================================================
train_preference_clustering()

# ======================================================
# RUN LIFECYCLE
# ======================================================
assign_lifecycle()

print("\nCLUSTERING DONE")