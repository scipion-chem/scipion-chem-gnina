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

import os, glob

from pyworkflow.protocol.params import (
    PointerParam, EnumParam, IntParam, FloatParam, BooleanParam,
    StringParam, LEVEL_ADVANCED, STEPS_PARALLEL, USE_GPU, GPU_LIST,
)
from pyworkflow.utils.path import makePath
import pyworkflow.object as pwobj

from pwem.protocols import EMProtocol
from pwchem.objects import SetOfSmallMolecules, SmallMolecule
from pwchem.utils import getBaseName, getBaseFileName, makeSubsets, runOpenBabel, convertToSdf

from .. import Plugin
from ..constants import (
    CNN_SCORING_CHOICES, CNN_SCORING_RESCORE,
    CNN_MODEL_CHOICES, CNN_MODEL_DEFAULT,
    SCORING_CHOICES, SCORING_DEFAULT,
    SORT_CHOICES,
    GNINA_OUTPUT_SDF,
)

FROM_PROTEIN = 0
FROM_POCKET = 1
PDBext, CIFext, PDBQText = '.pdb', '.cif', '.pdbqt'


class ProtGninaDocking(EMProtocol):
    """Perform molecular docking using GNINA.

    GNINA extends smina/AutoDock Vina with integrated CNN-based scoring.
    It accepts a rigid receptor (PDB/PDBQT), one or more ligands (SDF/MOL2/PDBQT),
    and a search-space definition (whole protein autobox or a structural ROI).

    For high-throughput screening, increase the number of threads (ligands are
    split into one subset per thread) and, on CUDA hardware, keep the GPU enabled
    for fast CNN scoring.

    This protocol always performs a full global docking search. To (re)score,
    minimise or locally optimise *already-docked* poses, use the dedicated
    "GNINA rescoring" protocol (ProtGninaScore) instead.

    References:
      McNutt et al., J. Cheminformatics 2021 (GNINA 1.0)
      McNutt et al., J. Cheminformatics 2025 (GNINA 1.3)
    """

    _label = 'GNINA docking'
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
        inputGroup.addParam('fromReceptor', EnumParam, label='Dock on : ', default=FROM_POCKET,
                            choices=['Whole protein', 'SetOfStructROIs'], display=EnumParam.DISPLAY_HLIST,
                            help='Whether to dock on a whole protein surface or on specific regions (pockets).')
        inputGroup.addParam('inputAtomStruct', PointerParam, pointerClass='AtomStruct',
                            label='Receptor structure: ', condition=f'fromReceptor == {FROM_PROTEIN}',
                            help='Protein structure to use as receptor. The whole protein bounding box '
                                 '(plus a buffer) is used as search space.')
        inputGroup.addParam('inputStructROIs', PointerParam, pointerClass="SetOfStructROIs",
                            label='Input pockets: ', condition=f'fromReceptor == {FROM_POCKET}',
                            help="The protein structural ROIs (pockets) to dock in. Docking is run "
                                 "independently for each pocket.")
        inputGroup.addParam('inputSmallMolecules', PointerParam, pointerClass='SetOfSmallMolecules',
                            label='Ligand set: ', allowsNull=False,
                            help='Set of small molecules to dock. Ligands are split into one multi-ligand '
                                 'SDF per thread and passed to gnina for efficiency.')
        inputGroup.addParam('pocketRadiusN', FloatParam, label='Grid radius vs ROI radius: ',
                            default=1.2, allowsNull=False, condition=f'fromReceptor == {FROM_POCKET}',
                            help='The size of each StructROI multiplied by this factor is used as the '
                                 'search-box size.')
        inputGroup.addParam('autoboxAdd', FloatParam, label='Autobox buffer (Å): ',
                            default=4.0, condition=f'fromReceptor == {FROM_PROTEIN}',
                            help='Buffer space added on every side of the auto-generated box that wraps '
                                 'the whole receptor (gnina --autobox_add).')

        # ---- Docking & Scoring ----------------------------------------- #
        form.addSection(label='Docking & Scoring')
        form.addParam('exhaustiveness', IntParam, default=8, label='Exhaustiveness: ',
                      help='Exhaustiveness of the global Monte-Carlo search (roughly proportional to time).')
        form.addParam('numPoses', IntParam, default=9, label='Number of binding modes: ',
                      help='Maximum number of docking poses to generate per ligand (--num_modes).')
        form.addParam('minRmsdFilter', FloatParam, default=1.0, label='Min. RMSD filter (Å): ',
                      expertLevel=LEVEL_ADVANCED,
                      help='Minimum RMSD between output poses to prune near-duplicate conformations.')

        cnnGroup = form.addGroup('CNN scoring')
        cnnGroup.addParam('cnnScoring', EnumParam, choices=CNN_SCORING_CHOICES, default=CNN_SCORING_RESCORE,
                          label='CNN scoring mode: ',
                          help='Controls when the CNN is applied:\n\n'
                               '- *rescore* (default): CNN re-ranks final poses. Fastest CNN mode.\n'
                               '- *refinement*: CNN refines poses (~10x slower).\n'
                               '- *metrorescore* / *metrorefine*: Metropolis variants.\n'
                               '- *all*: CNN used throughout.\n'
                               '- *none*: purely empirical scoring (fastest, no CNN).')
        cnnGroup.addParam('cnnModel', EnumParam, choices=CNN_MODEL_CHOICES, default=CNN_MODEL_DEFAULT,
                          label='CNN model: ', expertLevel=LEVEL_ADVANCED,
                          help='Built-in CNN model to use. "default" lets gnina use its default model '
                               'ensemble (recommended). Selecting a single model (e.g. crossdock_default2018) '
                               'is faster but may be less robust.')
        cnnGroup.addParam('cnnRotation', IntParam, default=0, label='CNN rotations: ',
                          expertLevel=LEVEL_ADVANCED,
                          help='Evaluate this many random rotations of each pose with the CNN and average '
                               '(max 24). 0 disables (--cnn_rotation).')

        form.addParam('scoring', EnumParam, choices=SCORING_CHOICES, default=SCORING_DEFAULT,
                      label='Empirical scoring function: ', expertLevel=LEVEL_ADVANCED,
                      help='Empirical scoring function used for the non-CNN stage of the pipeline.')
        form.addParam('poseSortOrder', EnumParam, choices=SORT_CHOICES, default=0,
                      label='Sort poses by: ', expertLevel=LEVEL_ADVANCED,
                      help='Criterion used by gnina to rank the output poses (--pose_sort_order).')
        form.addParam('addH', BooleanParam, default=True, label='Add hydrogens to ligands: ',
                      expertLevel=LEVEL_ADVANCED,
                      help='Let gnina automatically add hydrogens to the ligands (on by default).')
        form.addParam('seed', IntParam, default=42, label='Random seed: ', expertLevel=LEVEL_ADVANCED,
                      help='Set to a positive integer for reproducible runs. Set to 0 for a random seed.')

        # ---- Flexible residues ----------------------------------------- #
        flexGroup = form.addGroup('Flexible residues')
        flexGroup.addParam('doFlexRes', BooleanParam, label='Add flexible residues: ', default=False,
                           help='Treat some receptor residues as flexible during docking.')
        flexGroup.addParam('flexChain', StringParam, label='Residue chain: ', condition='doFlexRes',
                           help='Specify the protein chain (use the wizard).')
        flexGroup.addParam('flexRes', StringParam, label='Flexible residues: ', condition='doFlexRes',
                           help='Comma-separated list of chain:resid (e.g. A:42,A:43) or use the wizard.')

        form.addParallelSection(threads=4, mpi=1)

    # ------------------------------------------------------------------ #
    #  Steps insertion                                                     #
    # ------------------------------------------------------------------ #
    def _insertAllSteps(self):
        nt = self.numberOfThreads.get()
        subsets = makeSubsets(self.inputSmallMolecules.get(), max(nt - 1, 1), cloneItem=True)
        needsGPU = bool(getattr(self, USE_GPU).get())

        cRStep = self._insertFunctionStep(self.convertReceptorStep, prerequisites=[], needsGPU=False)

        cSteps = []
        for it, molSet in enumerate(subsets):
            cId = self._insertFunctionStep(self.convertLigandsStep, molSet, it,
                                           prerequisites=[], needsGPU=False)
            cSteps.append(cId)

        dockSteps = []
        if self.fromReceptor.get() == FROM_PROTEIN:
            for it in range(len(subsets)):
                dId = self._insertFunctionStep(self.dockingStep, it, None,
                                               prerequisites=[cRStep, cSteps[it]], needsGPU=needsGPU)
                dockSteps.append(dId)
        else:
            for pocket in self.inputStructROIs.get():
                for it in range(len(subsets)):
                    dId = self._insertFunctionStep(self.dockingStep, it, pocket.clone(),
                                                   prerequisites=[cRStep, cSteps[it]], needsGPU=needsGPU)
                    dockSteps.append(dId)

        self._insertFunctionStep(self.createOutputStep, prerequisites=dockSteps, needsGPU=False)

    # ------------------------------------------------------------------ #
    #  Step functions                                                      #
    # ------------------------------------------------------------------ #
    def convertReceptorStep(self):
        receptorFile = self.getOriginalReceptorFile()
        self.other2pdbqt(receptorFile, self.getReceptorPDBQT())

    def convertLigandsStep(self, molSet, it):
        """Merge a ligand subset into a single multi-ligand SDF for this thread.

        Each molecule block is retitled with the ligand base name so the poses in
        the gnina output SDF can be matched back to the input molecules.
        """
        outSdf = self._getSubsetLigandFile(it)
        with open(outSdf, 'w') as fout:
            for mol in molSet:
                molFile = mol.getFileName()
                molName = getBaseName(molFile)
                sdfFile = convertToSdf(self, molFile)
                with open(sdfFile) as fin:
                    blocks = [b for b in fin.read().split('$$$$') if b.strip()]
                for block in blocks:
                    block = block.lstrip('\n')
                    rest = block.split('\n', 1)
                    body = rest[1] if len(rest) > 1 else ''
                    fout.write(f'{molName}\n{body.rstrip()}\n$$$$\n')

    def dockingStep(self, subsetId, pocket=None):
        """Run gnina for one (ligand subset, pocket) target pair."""
        recFile = self.getReceptorPDBQT()
        ligFile = os.path.abspath(self._getSubsetLigandFile(subsetId))

        outDir = self._getRunDir(pocket, subsetId)
        makePath(outDir)
        outFile = os.path.join(outDir, GNINA_OUTPUT_SDF)
        logFile = os.path.join(outDir, 'gnina.log')

        args = self._buildArgs(recFile, ligFile, outFile, logFile, pocket)
        Plugin.runGnina(self, args, cwd=outDir)

    def createOutputStep(self):
        """Collect all docked SDF files and build a SetOfSmallMolecules output."""
        outDir = self._getPath('outputLigands')
        makePath(outDir)

        recFile = self.getReceptorPDBQT()
        outputSet = SetOfSmallMolecules().create(outputPath=self._getPath())

        # Lookup: ligand base name -> source SmallMolecule
        inputMolsDic = {getBaseName(mol.getFileName()): mol.clone()
                        for mol in self.inputSmallMolecules.get()}

        sdfFiles = sorted(glob.glob(self._getExtraPath('*', 'subset_*', GNINA_OUTPUT_SDF)))
        for sdfFile in sdfFiles:
            pocketId = self._pocketIdFromPath(sdfFile)
            gridId = pocketId if pocketId is not None else 1
            # Final pose name follows the suite convention g<gridId>_<molName>_<poseId>;
            # the subset (thread) layout is internal and must not leak into the name.
            prefix = f'g{gridId}_'

            for poseData in self.splitGninaSDF(sdfFile, outDir, prefix=prefix):
                srcMol = inputMolsDic.get(poseData['molName'])
                if srcMol is None:
                    print(f"Warning: docked molecule '{poseData['molName']}' not found "
                          f"among input ligands; skipping pose.")
                    continue

                newMol = SmallMolecule()
                newMol.copy(srcMol, copyId=False)
                newMol.setPoseFile(os.path.relpath(poseData['poseFile']))
                newMol.setPoseId(poseData['mode'])
                newMol.setGridId(gridId)
                newMol.setMolClass('Gnina')
                newMol.setDockId(self.getObjId())

                energy = self.getTagValueFromSdf(poseData['poseFile'], 'minimizedAffinity')
                cnnScore = self.getTagValueFromSdf(poseData['poseFile'], 'CNNscore')
                cnnAff = self.getTagValueFromSdf(poseData['poseFile'], 'CNNaffinity')
                if energy is not None:
                    newMol._energy = pwobj.Float(energy)
                if cnnScore is not None:
                    newMol._cnnScore = pwobj.Float(cnnScore)
                if cnnAff is not None:
                    newMol._cnnAffinity = pwobj.Float(cnnAff)

                outputSet.append(newMol)

        outputSet.updateMolClass()
        outputSet.setProteinFile(os.path.relpath(recFile))
        outputSet.setDocked(True)
        self._defineOutputs(outputSmallMolecules=outputSet)
        self._defineSourceRelation(self.inputSmallMolecules, outputSet)

    # ------------------------------------------------------------------ #
    #  Argument building                                                   #
    # ------------------------------------------------------------------ #
    def _buildArgs(self, recFile, ligFile, outFile, logFile, pocket=None):
        """Assemble the gnina command-line argument string."""
        args = f'-r "{recFile}" -l "{ligFile}" -o "{outFile}" --log "{logFile}"'

        # Search space
        if self.fromReceptor.get() == FROM_PROTEIN:
            args += f' --autobox_ligand "{recFile}" --autobox_add {self.autoboxAdd.get()}'
        elif pocket is not None:
            minMaxCoords = pocket.getLimits()
            xCenter, yCenter, zCenter = pocket.calculateMassCenter()
            diams = [(mm[1] - mm[0]) * self.pocketRadiusN.get() for mm in minMaxCoords]
            args += f' --center_x {xCenter} --center_y {yCenter} --center_z {zCenter}'
            args += f' --size_x {diams[0]} --size_y {diams[1]} --size_z {diams[2]}'

        # CNN scoring + model
        args += f' --cnn_scoring {CNN_SCORING_CHOICES[self.cnnScoring.get()]}'
        cnnModel = CNN_MODEL_CHOICES[self.cnnModel.get()]
        if cnnModel != 'default':
            args += f' --cnn {cnnModel}'
        if self.cnnRotation.get() > 0:
            args += f' --cnn_rotation {self.cnnRotation.get()}'

        # Empirical scoring (only if non-default)
        scoringFn = SCORING_CHOICES[self.scoring.get()]
        if scoringFn != 'default':
            args += f' --scoring {scoringFn}'

        # Search parameters
        args += f' --exhaustiveness {self.exhaustiveness.get()}'
        args += f' --num_modes {self.numPoses.get()}'
        args += f' --min_rmsd_filter {self.minRmsdFilter.get()}'

        args += f' --pose_sort_order {SORT_CHOICES[self.poseSortOrder.get()]}'

        if not self.addH.get():
            args += ' --addH 0'

        seed = self.seed.get()
        if seed > 0:
            args += f' --seed {seed}'

        # Flexible residues
        if self.doFlexRes.get():
            flexRes = self.flexRes.get().strip() if self.flexRes.get() else ''
            if flexRes:
                args += f' --flexres {flexRes}'

        # GPU / CPU
        gpuList = self._getGPUIds()
        if getattr(self, USE_GPU).get() and gpuList:
            args += f' --device {gpuList[0]}'
        else:
            args += ' --no_gpu'

        args += f' --cpu {self.exhaustiveness.get()}'
        return args

    # ------------------------------------------------------------------ #
    #  Output parsing                                                      #
    # ------------------------------------------------------------------ #
    def splitGninaSDF(self, sdfFile, outDir, prefix=''):
        """Split a multi-ligand output SDF into one file per pose.

        Returns a list of dicts with molName / mode (pose index per molecule) /
        poseFile (absolute path).
        """
        with open(sdfFile) as fh:
            content = fh.read()

        rawBlocks = [b.strip() for b in content.split('$$$$') if b.strip()]
        poses, molCounts = [], {}
        for globalIdx, block in enumerate(rawBlocks):
            lines = block.splitlines()
            molName = lines[0].strip() if lines else f'mol_{globalIdx}'

            molCounts[molName] = molCounts.get(molName, 0) + 1
            mode = molCounts[molName]

            poseFile = os.path.join(outDir, f'{prefix}{molName}_{mode}.sdf')
            with open(poseFile, 'w') as fh:
                fh.write(block + '\n$$$$\n')

            poses.append({'molName': molName, 'mode': mode,
                          'poseFile': os.path.abspath(poseFile)})
        return poses

    @staticmethod
    def getTagValueFromSdf(sdfFile, tag):
        """Return the float value of an SDF data tag (> <tag>) from a pose file."""
        marker = f'> <{tag}>'
        with open(sdfFile) as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines):
            if line.strip() == marker and i + 1 < len(lines):
                try:
                    return float(lines[i + 1].strip())
                except ValueError:
                    return None
        return None

    def _pocketIdFromPath(self, sdfFile):
        for part in sdfFile.split(os.sep):
            if part.startswith('pocket_'):
                return int(part.split('_')[1])
        return None

    # ------------------------------------------------------------------ #
    #  Path helpers                                                        #
    # ------------------------------------------------------------------ #
    def _getSubsetLigandFile(self, subsetId):
        return self._getExtraPath(f'ligands_subset_{subsetId}.sdf')

    def _getRunDir(self, pocket, subsetId):
        if pocket is None:
            return os.path.abspath(self._getExtraPath('whole_receptor', f'subset_{subsetId}'))
        return os.path.abspath(self._getExtraPath(f'pocket_{pocket.getObjId()}', f'subset_{subsetId}'))

    # ------------------------------------------------------------------ #
    #  Receptor helpers                                                    #
    # ------------------------------------------------------------------ #
    def getOriginalReceptorFile(self, getLink=True):
        recLink = self.getReceptorLink()
        if recLink is None or not getLink:
            if hasattr(self, 'inputAtomStruct') and self.inputAtomStruct.get() and \
                    (not hasattr(self, 'fromReceptor') or self.fromReceptor.get() == FROM_PROTEIN):
                recFile = self.inputAtomStruct.get().getFileName()
            elif hasattr(self, 'inputStructROIs') and self.inputStructROIs.get() and \
                    (not hasattr(self, 'fromReceptor') or self.fromReceptor.get() == FROM_POCKET):
                recFile = self.inputStructROIs.get().getProteinFile()
            else:
                print('No original receptor file found')
                return None

            if getLink:
                recDir = self._getExtraPath('originalReceptor')
                if not os.path.exists(recDir):
                    os.mkdir(recDir)
                recLink = os.path.join(recDir, getBaseFileName(recFile))
                if not os.path.exists(recLink):
                    os.link(recFile, recLink)
            else:
                recLink = recFile
        return recLink

    def getReceptorLink(self):
        recDir = self._getExtraPath('originalReceptor')
        if os.path.exists(recDir):
            for file in os.listdir(recDir):
                return os.path.join(recDir, file)
        return None

    def getReceptorName(self):
        return getBaseName(self.getOriginalReceptorFile())

    def getReceptorPDBQT(self):
        return os.path.abspath(self._getExtraPath(f'{self.getReceptorName()}.pdbqt'))

    def other2pdbqt(self, otherFile, pdbqtFile):
        """Convert pdb (or other receptor formats) to pdbqt using openbabel."""
        inExt = os.path.splitext(os.path.basename(otherFile))[1]
        inFormat = inExt[1:] if inExt in [PDBext, '.mol2', '.sdf', '.mol', CIFext] else 'pdb'

        args = ' -i{} {} -opdbqt -O {} -xr'.format(inFormat, os.path.abspath(otherFile),
                                                   os.path.abspath(pdbqtFile))
        runOpenBabel(self, args=args, popen=True)
        self._cleanReceptorPDBQT(pdbqtFile)
        return os.path.abspath(pdbqtFile)

    @staticmethod
    def _cleanReceptorPDBQT(pdbqtFile):
        """Drop atom records OpenBabel could not assign an AutoDock type to.

        Metals such as the heme iron are written by OpenBabel with an empty
        type column, which makes gnina's strict PDBQT parser abort with
        'ATOM syntax incorrect'. Such atoms are removed so the receptor parses.
        """
        kept = []
        for line in open(pdbqtFile):
            if line.startswith(('ATOM', 'HETATM')) and not line[77:].strip():
                continue
            kept.append(line)
        with open(pdbqtFile, 'w') as fh:
            fh.writelines(kept)

    def _getGPUIds(self):
        gpus = []
        for gp in getattr(self, GPU_LIST).get().split(','):
            if gp.strip():
                gpus.append(str(int(gp.strip())))
        return gpus

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #
    def _validate(self):
        errors = []
        if not os.path.isfile(Plugin.getGninaBinary()):
            errors.append('gnina binary not found at: %s\n'
                          'Please install it with "scipion3 installb gnina".'
                          % Plugin.getGninaBinary())

        if self.fromReceptor.get() == FROM_POCKET and not self.inputStructROIs.get():
            errors.append('Pocket mode requires a SetOfStructROIs input.')
        if self.fromReceptor.get() == FROM_PROTEIN and not self.inputAtomStruct.get():
            errors.append('Whole-protein mode requires an AtomStruct receptor.')

        if self.numPoses.get() < 1:
            errors.append('Number of binding modes must be >= 1.')
        if self.exhaustiveness.get() < 1:
            errors.append('Exhaustiveness must be >= 1.')

        if self.doFlexRes.get() and not (self.flexRes.get() and self.flexRes.get().strip()):
            errors.append('Flexible docking is enabled but no flexible residues were defined '
                          '(use the wizard or type a chain:resid list).')
        return errors

    # ------------------------------------------------------------------ #
    #  Summary / methods / citations                                       #
    # ------------------------------------------------------------------ #
    def _summary(self):
        summary = []
        if self.fromReceptor.get() == FROM_PROTEIN and self.inputAtomStruct.get():
            summary.append(f'Receptor: {os.path.basename(self.inputAtomStruct.get().getFileName())}')
        if self.fromReceptor.get() == FROM_POCKET and self.inputStructROIs.get():
            summary.append(f'Pockets: {self.inputStructROIs.get().getSize()} ROI(s)')
        if self.inputSmallMolecules.get():
            summary.append(f'Ligands: {self.inputSmallMolecules.get().getSize()} molecule(s)')
        summary.append(f'CNN scoring: {CNN_SCORING_CHOICES[self.cnnScoring.get()]}')
        summary.append(f'Exhaustiveness: {self.exhaustiveness.get()} | Modes: {self.numPoses.get()}')
        if self.hasAttribute('outputSmallMolecules'):
            summary.append(f'Output poses: {self.outputSmallMolecules.getSize()}')
        return summary

    def _methods(self):
        return [
            'Molecular docking was performed with GNINA [McNutt2021, McNutt2025].',
            f'CNN scoring mode "{CNN_SCORING_CHOICES[self.cnnScoring.get()]}" was used '
            f'with exhaustiveness {self.exhaustiveness.get()} and up to '
            f'{self.numPoses.get()} binding modes per ligand.',
        ]

    def _citations(self):
        return ['McNutt2021', 'McNutt2025']
