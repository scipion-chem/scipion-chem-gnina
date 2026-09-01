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

from .constants import *

_logo = 'icon.png'
_references = ['McNutt2021', 'McNutt2025']
__version__ = ALPHA_VERSION


class Plugin(pwchem.Plugin):
    """Plugin to integrate GNINA molecular docking into Scipion."""

    _homeVar = GNINA_HOME
    _pathVars = [GNINA_HOME]

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
    def runGnina(cls, protocol, args, cwd=None, popen=False):
        """Run a gnina command inside a protocol step.

        The conda env is activated first so that $CONDA_PREFIX is set, then
        LD_LIBRARY_PATH is prepended with $CONDA_PREFIX/lib to make cudnn9
        (and any other conda-managed libs) visible to the static binary.

        :param protocol: calling Scipion protocol object
        :param args:     command-line argument string (without 'gnina' prefix)
        :param cwd:      working directory (default: protocol._getExtraPath())
        :param popen:    if True use subprocess.check_call instead of runJob
        """
        fullProgram = (
            f'{cls.getEnvActivationCommand(GNINA_DIC)} && '
            f'LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH '
            f'{cls.getGninaBinary()}'
        )
        if not popen:
            protocol.runJob(fullProgram, args, env=cls.getEnviron(), cwd=cwd,
                            numberOfThreads=1)
        else:
            subprocess.check_call(f'{fullProgram} {args}', cwd=cwd, shell=True,
                                  executable='/bin/bash')

    @classmethod
    def getEnviron(cls):
        """Return an environment dict for running gnina subprocesses."""
        environ = pwutils.Environ(os.environ)
        return environ