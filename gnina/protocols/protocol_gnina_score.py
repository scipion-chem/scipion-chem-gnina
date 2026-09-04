# **************************************************************************
# *
# * Authors: Joaquin Algorta (joaquin.algorta@cnb.csic.es)
# *
# * Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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

import os, glob, json

from pyworkflow.protocol.params import (
    PointerParam, EnumParam, IntParam, FloatParam, BooleanParam,
    StringParam, LEVEL_ADVANCED, STEPS_PARALLEL, USE_GPU, GPU_LIST,
)
from pyworkflow.utils.path import makePath
import pyworkflow.object as pwobj

from pwchem.objects import SetOfSmallMolecules
from pwchem.utils import getBaseName, makeSubsets, convertToSdf

from .. import Plugin
from .protocol_gnina import ProtGninaDocking
from ..constants import (
    CNN_SCORING_CHOICES, CNN_SCORING_RESCORE,
    CNN_MODEL_CHOICES, CNN_MODEL_DEFAULT, CNN_MODEL_SENTINEL,
    SCORING_CHOICES, SCORING_DEFAULT,
    SCORE_MODE_CHOICES, SCORE_ONLY, SCORE_LOCAL, SCORE_MINIMIZE,
)


class ProtGninaScore(ProtGninaDocking):
    """Rescore already-docked poses with GNINA.

    Unlike the docking protocol, this one does **not** perform a global search:
    it takes an existing (docked) ``SetOfSmallMolecules`` and re-evaluates every
    pose *in place* with GNINA's empirical + CNN scoring. Three modes are exposed:

    - *Score only* (``--score_only``): score the pose as-is, no movement.
    - *Local optimization* (``--local_only``): local search around the pose.
    - *Energy minimization* (``--minimize``): minimise the pose in the pocket.

    **One gnina call per subset (like the docking protocol).** The input poses
    are split into one subset per worker thread and each subset is written to a
    single multi-ligand SDF that gnina scores in one process (with ``--cpu`` =
    number of threads). Every pose is scored — whether or not several poses
    belong to the same molecule — because each block is reduced to its clean
    molblock (data tags are stripped; see :meth:`_poseBody`), which keeps gnina's
    multi-ligand parser from stopping after the first entry.

    The output is a copy of the input set whose molecules carry the refreshed
    GNINA scores (``_gninaEnergy`` = affinity, ``_gninaCnnScore``,
    ``_gninaCnnAffinity``). In *Score only* mode the original pose files are kept
    untouched; the optimisation / minimisation modes do move the atoms, so for
    those the refined pose is saved and becomes the new pose file.

    Receptor conversion is reused from :class:`ProtGninaDocking`; the receptor is
    taken from the input set's protein file.

    References:
      McNutt et al., J. Cheminformatics 2021 (GNINA 1.0)
      McNutt et al., J. Cheminformatics 2025 (GNINA 1.3)
    """

    _label = 'GNINA rescoring'
    stepsExecutionMode = STEPS_PARALLEL

    # ------------------------------------------------------------------ #
    #  Form definition                                                     #
    # ------------------------------------------------------------------ #
    def _defineParams(self, form):
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution: ",
                       help="GNINA can use CUDA GPUs to accelerate CNN scoring. "
                            "If disabled, gnina runs on CPU (much slower for CNN modes).")
        form.addHidden(GPU_LIST, StringParam, default='0', label="Choose GPU IDs",
                       help="Comma-separated list of GPU device indices that can be used.")

        # ---- Input ----------------------------------------------------- #
        form.addSection(label='Input')
        inputGroup = form.addGroup('Input specifications')
        inputGroup.addParam('inputSmallMolecules', PointerParam, pointerClass='SetOfSmallMolecules',
                            label='Docked small molecules: ', allowsNull=False,
                            help='Set of already-docked small molecules (with poses) to rescore. '
                                 'Every pose in the set is scored.')

        # ---- Rescoring ------------------------------------------------- #
        form.addSection(label='Rescoring')
        form.addParam('scoreMode', EnumParam, choices=SCORE_MODE_CHOICES, default=SCORE_ONLY,
                      label='Rescoring mode: ',
                      help='- *Score only*: score the input pose without moving it. '
                           'The original pose is kept; only the scores are added.\n'
                           '- *Local optimization*: local search around the input pose.\n'
                           '- *Energy minimization*: minimise the input pose.\n'
                           'The last two modify the atoms, so the refined structure is stored as the '
                           'new pose.')
        form.addParam('autoboxAdd', FloatParam, label='Autobox buffer (Å): ',
                      default=4.0, expertLevel=LEVEL_ADVANCED,
                      help='Buffer space added on every side of the box auto-generated around the '
                           'poses. Only relevant for the optimisation/minimisation modes.')

        cnnGroup = form.addGroup('CNN scoring')
        cnnGroup.addParam('cnnScoring', EnumParam, choices=CNN_SCORING_CHOICES, default=CNN_SCORING_RESCORE,
                          label='CNN scoring mode: ',
                          help='Controls how the CNN is applied:\n\n'
                               '- *rescore* (default): CNN re-ranks the pose. Fastest CNN mode.\n'
                               '- *refinement*: CNN refines the pose (~10x slower).\n'
                               '- *metrorescore* / *metrorefine*: Metropolis variants.\n'
                               '- *all*: CNN used throughout.\n'
                               '- *none*: purely empirical scoring (fastest, no CNN).')
        cnnGroup.addParam('cnnModel', EnumParam, choices=CNN_MODEL_CHOICES, default=CNN_MODEL_DEFAULT,
                          label='CNN model: ', expertLevel=LEVEL_ADVANCED,
                          help='Built-in CNN model used for scoring (gnina --cnn).\n\n'
                               '- *gnina default ensemble* (recommended): does not pass --cnn, so gnina '
                               'uses its own default model ensemble. This ensemble has no name accepted '
                               'by --cnn, so this entry is the only way to select it.\n'
                               '- *fast* / *dense*: a small, quick network vs. a heavier DenseNet one.\n'
                               '- *default2017* / *default1.0*: the models earlier gnina versions used.\n'
                               '- *crossdock_/redock_/general_default2018*: the 2018 family, differing in '
                               'the poses they were trained on (cross-docked, re-docked, general set).\n'
                               '- *..._ensemble*: averages every model sharing that prefix. More robust '
                               'than the matching single model, and proportionally slower.\n\n'
                               'Note the choice changes the scores appreciably, so keep it fixed when '
                               'comparing runs.')
        cnnGroup.addParam('cnnRotation', IntParam, default=0, label='CNN rotations: ',
                          expertLevel=LEVEL_ADVANCED,
                          help='Evaluate this many random rotations of each pose with the CNN and average '
                               '(max 24). 0 disables (--cnn_rotation).')

        form.addParam('scoring', EnumParam, choices=SCORING_CHOICES, default=SCORING_DEFAULT,
                      label='Empirical scoring function: ', expertLevel=LEVEL_ADVANCED,
                      help='Empirical scoring function used for the non-CNN stage of the pipeline.')
        form.addParam('addH', BooleanParam, default=False, label='Add hydrogens to ligands: ',
                      expertLevel=LEVEL_ADVANCED,
                      help='Let gnina automatically add hydrogens to the ligands (on by default).')
        form.addParam('seed', IntParam, default=42, label='Random seed: ', expertLevel=LEVEL_ADVANCED,
                      help='Set to a positive integer for reproducible runs. Set to 0 for a random seed.')

        form.addParallelSection(threads=4, mpi=1)

    # ------------------------------------------------------------------ #
    #  Steps insertion                                                     #
    # ------------------------------------------------------------------ #
    def _insertAllSteps(self):
        nt = self.numberOfThreads.get()
        subsets = makeSubsets(self.inputSmallMolecules.get(), max(nt - 1, 1), cloneItem=True)
        needsGPU = bool(getattr(self, USE_GPU).get())

        cRStep = self._insertFunctionStep(self.convertReceptorStep, prerequisites=[], needsGPU=False)

        sSteps = []
        for it, molSet in enumerate(subsets):
            sId = self._insertFunctionStep(self.scoreStep, molSet, it,
                                           prerequisites=[cRStep], needsGPU=needsGPU)
            sSteps.append(sId)

        self._insertFunctionStep(self.createOutputStep, prerequisites=sSteps, needsGPU=False)

    # ------------------------------------------------------------------ #
    #  Step functions                                                      #
    # ------------------------------------------------------------------ #
    def scoreStep(self, molSet, it):
        """Score one subset of poses in a single gnina call.
        Results are written to a per-subset JSON file.
        """
        recFile = self.getReceptorPDBQT()
        minimizing = self.scoreMode.get() != SCORE_ONLY

        runDir = os.path.abspath(self._getExtraPath(f'subset_{it}'))
        makePath(runDir)
        inSdf = os.path.join(runDir, 'ligands.sdf')
        logFile = os.path.join(runDir, 'gnina.log')
        outSdf = os.path.join(runDir, 'scored.sdf')

        # Build a single multi-ligand SDF.
        with open(inSdf, 'w') as fout:
            for mol in molSet:
                poseFile = os.path.abspath(mol.getPoseFile())
                poseKey = getBaseName(poseFile)
                body = self._poseBody(poseFile)
                fout.write(f'{poseKey}\n{body}\n$$$$\n')

        args = self._buildScoreArgs(recFile, inSdf, logFile, outSdf)
        Plugin.runGnina(self, args, cwd=runDir, gpuId=self.getGpuId())

        outPoseDir = self._getPath('outputLigands')
        if minimizing:
            makePath(outPoseDir)

        scores = {}
        for title, block in self._iterSdfEntries(outSdf):
            sd = {
                'energy': self._tagFromBlock(block, 'minimizedAffinity'),
                'cnnScore': self._tagFromBlock(block, 'CNNscore'),
                'cnnAffinity': self._tagFromBlock(block, 'CNNaffinity'),
            }
            # Only the optimisation/minimisation modes move the atoms, so only
            # then is a new pose file written; score-only keeps the input pose.
            if minimizing:
                posePath = os.path.join(outPoseDir, f'resc_{title}.sdf')
                with open(posePath, 'w') as pf:
                    pf.write(block + '\n$$$$\n')
                sd['poseFile'] = os.path.relpath(posePath)
            scores[title] = sd

        with open(self._getScoresFile(it), 'w') as fh:
            json.dump(scores, fh)

    def createOutputStep(self):
        """Copy the input set and annotate every pose with its GNINA scores."""
        inMols = self.inputSmallMolecules.get()
        minimizing = self.scoreMode.get() != SCORE_ONLY
        recFile = self.getReceptorPDBQT()

        # Merge the per-batch score files: pose base name -> score dict.
        scores = {}
        for scoresFile in glob.glob(self._getExtraPath('scores_*.json')):
            with open(scoresFile) as fh:
                scores.update(json.load(fh))

        newMols = SetOfSmallMolecules.createCopy(inMols, self._getPath(), copyInfo=True)
        for mol in inMols:
            newMol = mol.clone()
            poseKey = getBaseName(mol.getPoseFile())
            sd = scores.get(poseKey)
            if sd is None:
                print(f"Warning: no GNINA score for pose '{poseKey}'; keeping it unscored.")
                sd = {}
            # Set the three attributes
            newMol._gninaEnergy = pwobj.Float(sd.get('energy'))
            newMol._gninaCnnScore = pwobj.Float(sd.get('cnnScore'))
            newMol._gninaCnnAffinity = pwobj.Float(sd.get('cnnAffinity'))
            if minimizing and sd.get('poseFile'):
                newMol.setPoseFile(sd['poseFile'])
            # Record the receptor the pose was (re)scored against.
            newMol.setProteinFile(os.path.relpath(recFile))
            newMols.append(newMol)

        newMols.updateMolClass()
        self._defineOutputs(outputSmallMolecules=newMols)
        self._defineSourceRelation(self.inputSmallMolecules, newMols)

    # ------------------------------------------------------------------ #
    #  Argument building                                                   #
    # ------------------------------------------------------------------ #
    def _buildScoreArgs(self, recFile, ligFile, logFile, outFile):
        """Assemble the gnina rescoring command line for a group of poses."""
        args = f'-r "{recFile}" -l "{ligFile}" -o "{outFile}" --log "{logFile}"'

        # Search box auto-generated around the (multi-ligand) input.
        args += f' --autobox_ligand "{ligFile}" --autobox_add {self.autoboxAdd.get()}'

        # Rescoring mode (operates on the provided poses, no global search).
        mode = self.scoreMode.get()
        if mode == SCORE_ONLY:
            args += ' --score_only'
        elif mode == SCORE_LOCAL:
            args += ' --local_only'
        elif mode == SCORE_MINIMIZE:
            args += ' --minimize'

        # CNN scoring + model. The first CNN_MODEL choice is a sentinel: gnina's
        # default ensemble has no --cnn name, so it is reached by omitting the flag.
        args += f' --cnn_scoring {CNN_SCORING_CHOICES[self.cnnScoring.get()]}'
        cnnModel = CNN_MODEL_CHOICES[self.cnnModel.get()]
        if cnnModel != CNN_MODEL_SENTINEL:
            args += f' --cnn {cnnModel}'
        if self.cnnRotation.get() > 0:
            args += f' --cnn_rotation {self.cnnRotation.get()}'

        # Empirical scoring ('default' is itself a valid gnina scoring function)
        args += f' --scoring {SCORING_CHOICES[self.scoring.get()]}'

        if not self.addH.get():
            args += ' --addH 0'

        seed = self.seed.get()
        if seed > 0:
            args += f' --seed {seed}'

        # GPU / CPU. The device is chosen through CUDA_VISIBLE_DEVICES in
        # Plugin.runGnina, not with --device: gnina 1.3.2's Torch backend
        # ignores that flag and warns about it, so passing it selected nothing.
        if not getattr(self, USE_GPU).get():
            args += ' --no_gpu'

        # Let gnina parallelise scoring of the group's ligands across CPU threads.
        args += f' --cpu {self.numberOfThreads.get()}'
        return args

    # ------------------------------------------------------------------ #
    #  SDF helpers                                                          #
    # ------------------------------------------------------------------ #
    def _poseBody(self, poseFile):
        """Return the molblock of a pose (atoms/bonds up to 'M  END'), WITHOUT
        any SDF data tags.

        Docked pose SDFs often carry leftover data tags (e.g. '> <CNNscore>' from
        a previous docking). Those tags break gnina's multi-ligand SDF parser,
        which then scores only the first molecule of the file, so they must be
        stripped. Non-SDF poses (e.g. AutoDock/Vina .pdbqt) are converted to SDF
        in Tmp first.
        """
        ext = os.path.splitext(poseFile)[1].lower()
        sdf = poseFile if ext == '.sdf' else os.path.abspath(convertToSdf(self, poseFile))
        with open(sdf) as fh:
            block = next((b for b in fh.read().split('$$$$') if b.strip()), '')
        return self._cleanBlock(block)

    @staticmethod
    def _iterSdfEntries(sdfFile):
        """Yield (title, block) for each molecule entry of an SDF file."""
        if not os.path.exists(sdfFile):
            return
        with open(sdfFile) as fh:
            content = fh.read()
        for block in content.split('$$$$'):
            block = block.strip('\n')
            if not block.strip():
                continue
            title = block.splitlines()[0].strip()
            yield title, block

    @staticmethod
    def _tagFromBlock(block, tag):
        """Return the float value of an SDF data tag (> <tag>) within a block."""
        lines = block.splitlines()
        marker = f'> <{tag}>'
        for i, line in enumerate(lines):
            if line.strip() == marker and i + 1 < len(lines):
                try:
                    return float(lines[i + 1].strip())
                except ValueError:
                    return None
        return None

    def _getScoresFile(self, it):
        return self._getExtraPath(f'scores_{it}.json')

    # ------------------------------------------------------------------ #
    #  Receptor source (override: read it from the docked input set)       #
    # ------------------------------------------------------------------ #
    def getOriginalReceptorFile(self, getLink=True):
        recLink = self.getReceptorLink()
        if recLink is None or not getLink:
            recFile = self.inputSmallMolecules.get().getProteinFile()
            if not recFile:
                print('No protein file found in the input docked set')
                return None

            if getLink:
                recDir = self._getExtraPath('originalReceptor')
                if not os.path.exists(recDir):
                    os.mkdir(recDir)
                recLink = os.path.join(recDir, os.path.basename(recFile))
                if not os.path.exists(recLink):
                    os.link(recFile, recLink)
            else:
                recLink = recFile
        return recLink

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #
    def _validate(self):
        errors = []
        if not os.path.isfile(Plugin.getGninaBinary()):
            errors.append('gnina binary not found at: %s\n'
                          'Please install it with "scipion3 installb gnina".'
                          % Plugin.getGninaBinary())

        molSet = self.inputSmallMolecules.get()
        if molSet is not None:
            if not molSet.isDocked():
                errors.append('The input %s is not docked yet; this protocol rescores '
                              'already-docked poses.' % molSet)
            if not molSet.getProteinFile():
                errors.append('The input set has no associated receptor (protein file).')
        return errors

    # ------------------------------------------------------------------ #
    #  Summary / methods / citations                                       #
    # ------------------------------------------------------------------ #
    def _summary(self):
        summary = []
        if self.inputSmallMolecules.get():
            summary.append(f'Input poses: {self.inputSmallMolecules.get().getSize()}')
        summary.append(f'Rescoring mode: {SCORE_MODE_CHOICES[self.scoreMode.get()]}')
        summary.append(f'CNN scoring: {CNN_SCORING_CHOICES[self.cnnScoring.get()]}')
        if self.hasAttribute('outputSmallMolecules'):
            summary.append(f'Rescored poses: {self.outputSmallMolecules.getSize()}')
        return summary

    def _methods(self):
        return [
            'Docking poses were rescored with GNINA [McNutt2021, McNutt2025] using '
            f'the "{SCORE_MODE_CHOICES[self.scoreMode.get()]}" mode and CNN scoring '
            f'"{CNN_SCORING_CHOICES[self.cnnScoring.get()]}".',
        ]

    def _citations(self):
        return ['McNutt2021', 'McNutt2025']
