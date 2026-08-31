# **************************************************************************
# *
# * Authors: Joaquin Algorta (joaquin.algorta@cnb.csic.es)
# *
# * CNB - CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
# Plugin version (independent of the wrapped GNINA version)
ALPHA_VERSION = '0.1'

# Wrapped GNINA release
GNINA_DEFAULT_VERSION = '1.3.2'

GNINA_DIC = {'name': 'gnina', 'version': GNINA_DEFAULT_VERSION, 'home': 'GNINA_HOME'}

# ---------------------------------------------------------------------------
# Scipion environment variable names
# ---------------------------------------------------------------------------
GNINA_HOME            = 'GNINA_HOME'
GNINA_ACTIVATION_CMD  = 'GNINA_ACTIVATION_CMD'   # optional; empty = no activation
GNINA_BINARY_NAME     = 'gnina'

# ---------------------------------------------------------------------------
# CNN scoring modes
# ---------------------------------------------------------------------------
CNN_SCORING_RESCORE     = 0
CNN_SCORING_REFINEMENT  = 1
CNN_SCORING_METRORESCORE = 2
CNN_SCORING_METROREFINE = 3
CNN_SCORING_ALL         = 4
CNN_SCORING_NONE        = 5

CNN_SCORING_CHOICES = [
    'rescore',
    'refinement',
    'metrorescore',
    'metrorefine',
    'all',
    'none',
]

# ---------------------------------------------------------------------------
# Built-in CNN models / ensembles (--cnn).

CNN_MODEL_DEFAULT = 0
CNN_MODEL_SENTINEL = 'gnina default ensemble'

# New entries must be APPENDED: the index is what gets stored in saved
# workflows, so reordering would silently change the model of existing runs.
CNN_MODEL_CHOICES = [
    CNN_MODEL_SENTINEL,
    # Single models
    'fast',
    'default2017',
    'default1.0',
    'crossdock_default2018',
    'redock_default2018',
    'general_default2018',
    'dense',
    # Ensembles: '<prefix>_ensemble' averages every model with that prefix.
    # More robust than the matching single model, proportionally slower.
    'all_default_to_default_1_3_ensemble',
    'crossdock_default2018_ensemble',
    'redock_default2018_ensemble',
    'general_default2018_ensemble',
    'dense_ensemble',
]

# ---------------------------------------------------------------------------
# Empirical scoring functions (--scoring).

SCORING_DEFAULT  = 0
SCORING_VINA     = 1
SCORING_VINARDO  = 2
SCORING_AD4      = 3
SCORING_DKOES    = 4

SCORING_CHOICES = [
    'default',
    'vina',
    'vinardo',
    'ad4_scoring',
    'dkoes_scoring',
]

# Name of the per-pose flexible-receptor file written by gnina (--out_flex)
GNINA_FLEX_PDBQT = 'flex_receptor.pdbqt'

# ---------------------------------------------------------------------------
# Rescoring score modes — operate on already-docked poses (no global search).

SCORE_ONLY     = 0   # --score_only : evaluate the pose as-is, no movement
SCORE_LOCAL    = 1   # --local_only : local optimisation around the pose
SCORE_MINIMIZE = 2   # --minimize   : energy minimisation of the pose

SCORE_MODE_CHOICES = [
    'Score only',
    'Local optimization',
    'Energy minimization',
]

# ---------------------------------------------------------------------------
# Output / pose sort order (--pose_sort_order)
# ---------------------------------------------------------------------------
SORT_CNN_SCORE    = 0
SORT_CNN_AFFINITY = 1
SORT_ENERGY       = 2

SORT_CHOICES = [
    'CNNscore',
    'CNNaffinity',
    'Energy',
]

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
GNINA_OUTPUT_SDF = 'docked.sdf'