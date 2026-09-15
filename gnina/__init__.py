# **************************************************************************
# *
# * Authors: Joaquin Algorta (joaquin.algorta@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

download_url = 'https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2'
import os
import subprocess

import pwchem

import pyworkflow.utils as pwutils
from .bibtex import _bibtexStr

from .constants import (ALPHA_VERSION, GNINA_ACTIVATION_CMD, GNINA_BINARY_NAME,
                        GNINA_DIC, GNINA_HOME)

_logo = 'icon.png'
_references = ['McNutt2021', 'McNutt2025']
__version__ = ALPHA_VERSION

# Shell fragment that locates Open Babel's data directory and leaves it in
# $BABEL_DIR (empty if there is none). The released gnina binary is statically
# linked against Open Babel but ships none of its data files, so the force
# field it needs for --covalent_optimize_lig, UFF.prm, is looked up at run time
# under $BABEL_DATADIR. With that unset the binary reports "Cannot open UFF.prm"
# on stderr and then carries on without optimising anything, which is how
# covalent poses come out strained and score positive affinities.
#
# The conda env of the plugin is searched first, then its sibling envs: the
# pwchem env always carries Open Babel, and UFF.prm is a plain parameter table
# that does not change between the 3.x releases.
#
# The directory is exported rather than passed as an assignment prefix on the
# gnina command: bash decides whether a word is an assignment before expanding
# it, so a prefix built by expansion is run as a command name instead.
BABEL_DATADIR_LOOKUP = (
    'BABEL_DIR=$(ls -d "$CONDA_PREFIX"/share/openbabel/*/ 2>/dev/null | tail -1); '
    '[ -n "$BABEL_DIR" ] || '
    'BABEL_DIR=$(ls -d "$CONDA_PREFIX"/../*/share/openbabel/*/ 2>/dev/null | tail -1); '
    '[ -n "$BABEL_DIR" ] && export BABEL_DATADIR="$BABEL_DIR"; ')


class Plugin(pwchem.Plugin):
    """Plugin to integrate GNINA molecular docking into Scipion."""

    _homeVar = GNINA_HOME
    _pathVars = [GNINA_HOME]
    _babelDataDir = None

    @classmethod
    def defineBinaries(cls, env):
        cls.addGninaPackage(env)

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(GNINA_DIC['home'], cls.getEnvName(GNINA_DIC))

    @classmethod
    def addGninaPackage(cls, env, default=True):
        """Install gnina binary + minimal conda env with cudnn=9
        """
        from scipion.install.funcs import InstallHelper

        installer = InstallHelper(GNINA_DIC['name'],
                                  packageHome=cls.getVar(GNINA_DIC['home']),
                                  packageVersion=GNINA_DIC['version'])

        gninaEnvName = cls.getEnvName(GNINA_DIC)
        installer.addCommand(
            f'conda create -n {gninaEnvName} cudnn=9 cuda-libraries=12 -c nvidia -y',
            'GNINA_ENV_CREATED'
        )

        # Open Babel is installed for its data files, not for its library: the
        # gnina binary carries its own copy of the code but none of the .prm
        # tables, and without UFF.prm --covalent_optimize_lig does nothing.
        # Kept as a command of its own so an existing installation can pick it
        # up, and so the conda-forge channel cannot disturb the env creation.
        installer.addCommand(
            f'conda install -n {gninaEnvName} openbabel -c conda-forge -y',
            'GNINA_BABEL_DATA'
        )

        installer.addCommand(
            f'wget -O {GNINA_BINARY_NAME} {download_url} && '
            f'chmod +x {GNINA_BINARY_NAME}',
            'GNINA_BINARY_READY'
        )

        installer.addPackage(env, dependencies=['wget', 'conda'], default=default)

    ######################## UTILS #########################

    @classmethod
    def getGninaHome(cls, *paths):
        """Return path inside the gnina home directory."""
        return os.path.join(cls.getVar(GNINA_HOME), *paths)

    @classmethod
    def getGninaBinary(cls):
        """Return the full path to the gnina executable."""
        return cls.getGninaHome(GNINA_BINARY_NAME)

    @classmethod
    def getGninaEnvActivation(cls):
        """Return any activation string needed before calling gnina.
        """
        return cls.getVar(GNINA_ACTIVATION_CMD) if cls.getVar(GNINA_ACTIVATION_CMD) else ''

    @classmethod
    def runGnina(cls, protocol, args, cwd=None, popen=False, gpuId=None):
        """Run a gnina command inside a protocol step.

        The conda env is activated first so that $CONDA_PREFIX is set, then
        LD_LIBRARY_PATH is prepended with $CONDA_PREFIX/lib to make cudnn9
        (and any other conda-managed libs) visible to the static binary.

        The GPU is selected with CUDA_VISIBLE_DEVICES, not with gnina's
        --device: the Torch backend of gnina 1.3.2 ignores that flag and says
        so ("Torch backend ignores device argument"), which silently sent every
        job to the first visible card. Note the chosen GPU is renumbered to
        index 0 inside the process, so --device must not be passed as well.

        :param protocol: calling Scipion protocol object
        :param args:     command-line argument string (without 'gnina' prefix)
        :param cwd:      working directory (default: protocol._getExtraPath())
        :param popen:    if True use subprocess.check_call instead of runJob
        :param gpuId:    CUDA device to expose; None leaves every card visible
        """
        # 'is not None': GPU 0 is a valid id and must not be treated as unset.
        gpuStr = f'CUDA_VISIBLE_DEVICES={gpuId} ' if gpuId is not None else ''
        fullProgram = (
            f'{cls.getEnvActivationCommand(GNINA_DIC)} && '
            f'{BABEL_DATADIR_LOOKUP}'
            f'{gpuStr}'
            f'LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH '
            f'{cls.getGninaBinary()}')
        if not popen:
            protocol.runJob(fullProgram, args, env=cls.getEnviron(), cwd=cwd,
                            numberOfThreads=1)
        else:
            subprocess.check_call(f'{fullProgram} {args}', cwd=cwd, shell=True,
                                  executable='/bin/bash')

    @classmethod
    def getBabelDataDir(cls):
        """Open Babel data directory holding UFF.prm, or '' if there is none.

        Resolved the same way as in runGnina, but from Python and checked for
        the file itself, so a protocol can refuse to start an optimisation that
        would silently do nothing. Cached: it costs a conda activation (~2 s).
        """
        if cls._babelDataDir is None:
            cmd = (f'{cls.getEnvActivationCommand(GNINA_DIC)} && '
                   f'{BABEL_DATADIR_LOOKUP} echo "GNINA_BABEL_DIR=$BABEL_DIR"')
            babelDir = ''
            try:
                proc = subprocess.run(cmd, shell=True, executable='/bin/bash',
                                      capture_output=True, text=True, timeout=120)
                for line in proc.stdout.splitlines():
                    # Marked so that whatever conda prints on activation cannot
                    # be mistaken for the answer.
                    if line.startswith('GNINA_BABEL_DIR='):
                        babelDir = line.split('=', 1)[1].strip()
            except (OSError, subprocess.SubprocessError):
                babelDir = ''

            if babelDir and not os.path.isfile(os.path.join(babelDir, 'UFF.prm')):
                babelDir = ''
            cls._babelDataDir = babelDir
        return cls._babelDataDir

    @classmethod
    def getEnviron(cls):
        """Return an environment dict for running gnina subprocesses."""
        environ = pwutils.Environ(os.environ)
        return environ