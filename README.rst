=====================
GNINA plugin
=====================

This is a **Scipion** plugin that offers molecular docking with
`GNINA <https://github.com/gnina/gnina>`_, a fork of smina/AutoDock Vina that
integrates convolutional neural network (CNN) scoring of the docked poses.

The plugin provides two protocols:

- **GNINA docking**: full global docking search of a set of small molecules on a
  whole protein (automatic bounding box) or on a set of structural ROIs
  (binding pockets). Produces a docked ``SetOfSmallMolecules`` whose poses carry
  the minimized affinity (``_energy``), ``_cnnScore`` and ``_cnnAffinity``.
- **GNINA rescoring**: re-evaluation of *already-docked* poses, either scoring
  them as they are, locally optimising them, or energy-minimising them inside
  the pocket.

Results integrate with the rest of the scipion-chem ecosystem: because the
output is a standard ``SetOfSmallMolecules``, it can be inspected with the
scipion-chem small-molecules viewer (PyMOL / ChimeraX pose display and score
tables) and fed into consensus docking, filtering or pose-quality protocols.

==========================
Install this plugin
==========================

You will need to first install
`Scipion3 <https://scipion-em.github.io/docs/release-3.0.0/docs/scipion-modes/how-to-install.html>`_ and
`Scipion-chem <https://github.com/scipion-chem/scipion-chem>`_ to run these protocols.

1. **Install the plugin in Scipion**

- **Stable version**

    Through the plugin manager GUI by launching Scipion and following
    **Configuration** >> **Plugins**

    or

.. code-block::

    scipion3 installp -p scipion-chem-gnina

- **Developer's version**

    1. **Download repository**:

    .. code-block::

        git clone https://github.com/scipion-chem/scipion-chem-gnina.git

    2. **Switch to the desired branch** (master or devel):

    Scipion-chem-gnina is constantly under development and including new features.
    If you want a relatively older and more stable version, use the master branch (default).
    If you want the latest changes and developments, use the devel branch.

    .. code-block::

        cd scipion-chem-gnina
        git checkout devel

    3. **Install**:

    .. code-block::

        scipion3 installp -p path_to_scipion-chem-gnina --devel

2. **Install the GNINA binary**

The GNINA executable and a minimal conda environment providing ``cudnn 9`` (used
for GPU CNN scoring) are installed automatically by Scipion:

.. code-block::

    scipion3 installb gnina

GNINA is distributed as a self-contained static binary, so no compilation is
required. A CUDA-capable GPU is strongly recommended: CNN scoring on CPU is
considerably slower. Docking still works without a GPU by unchecking
"Use GPU for execution" in the protocol form.

==========================
Running the tests
==========================

.. code-block::

    scipion3 test gnina.tests.test_gnina.TestGninaDocking
    scipion3 test gnina.tests.test_gnina.TestGninaScore

The tests download the ``model_building_tutorial`` and ``smallMolecules``
datasets automatically.

==========================
References
==========================

* McNutt A.T. et al. *GNINA 1.0: molecular docking with deep learning.*
  Journal of Cheminformatics 13, 43 (2021). https://doi.org/10.1186/s13321-021-00522-2
* McNutt A.T. et al. *GNINA 1.3: the next increment in molecular docking with
  deep learning.* Journal of Cheminformatics 17, 28 (2025).
  https://doi.org/10.1186/s13321-025-00973-x
