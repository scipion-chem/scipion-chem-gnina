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

import os, shutil, glob, subprocess

from pyworkflow.protocol.params import (
    PointerParam, EnumParam, IntParam, FloatParam, BooleanParam,
    StringParam, LEVEL_ADVANCED, USE_GPU, GPU_LIST,
)
from pyworkflow.utils.path import makePath
import pyworkflow.object as pwobj

from pwem.protocols import EMProtocol
from pwchem.objects import SetOfSmallMolecules, SmallMolecule, SetOfStructROIs
from pwchem.utils import getBaseName, makeSubsets, performBatchThreading, pdbFromASFile, getBaseFileName, runOpenBabel, convertToSdf
# from pwchem.constants import MGL_DIC, OPENBABEL_DIC, RDKIT_DIC

from .. import Plugin
from ..constants import (
    BOX_MODE_AUTOBOX, BOX_MODE_MANUAL, BOX_MODE_ROI,
    CNN_SCORING_CHOICES, CNN_SCORING_RESCORE,
    SCORING_CHOICES, SCORING_DEFAULT,
)

FROM_POCKET = 1
FROM_PROTEIN = 0
PDBext, CIFext, PDBQText = '.pdb', '.cif', '.pdbqt'


class ProtGninaDocking(EMProtocol):
    """Perform molecular docking using GNINA.

    GNINA extends smina/AutoDock Vina with integrated CNN-based scoring.
    It accepts a rigid receptor (PDB/PDBQT), one or more ligands (SDF/MOL2/PDBQT),
    and a search-space definition (ROI, autobox or explicit centre + size).

    For high-throughput screening, provide the receptor as PDBQT to skip
    re-protonation, and set exhaustiveness equal to the number of CPU threads.

    References:
      McNutt et al., J. Cheminformatics 2021 (GNINA 1.0)
      McNutt et al., J. Cheminformatics 2025 (GNINA 1.3)
    """

    _label = 'GNINA docking'

    def _defineParams(self, form):
        # ---- Input ----------------------------------------------------- #
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution: ",
                       help="This protocol has both CPU and GPU implementation.\
                                                 Select the one you want to use.")

        form.addHidden(GPU_LIST, StringParam, default='0', label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        form.addSection(label='Input')

        # form.addHidden(USE_GPU, BooleanParam, default=True,
        #                label='Use GPU for execution',
        #                help='GNINA can use CUDA GPUs to accelerate CNN scoring.')
        # form.addHidden(GPU_LIST, StringParam, default='0',
        #                label='Choose GPU IDs',
        #                help='Comma-separated list of GPU device indices to use.')
        inputGroup = form.addGroup('Input specifications')
        inputGroup.addParam('fromReceptor', EnumParam, label='Dock on : ', default=1,
                            choices=['Whole protein', 'SetOfStructROIs'], display=EnumParam.DISPLAY_HLIST,
                            help='Whether to dock on a whole protein surface or on specific regions')
        inputGroup.addParam('inputAtomStruct', PointerParam,
                            pointerClass='AtomStruct',
                            label='Receptor structure', condition=f'fromReceptor == {FROM_PROTEIN}',
                            help='Protein structure to use as receptor.')
        inputGroup.addParam('inputStructROIs', PointerParam, pointerClass="SetOfStructROIs",
                            label='Input pockets: ', condition=f'fromReceptor == {FROM_POCKET}',
                            help="The protein structural ROIs to dock in")
        inputGroup.addParam('inputSmallMolecules', PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      label='Ligand set',
                      help='Set of small molecules to dock. All ligands are merged '
                           'into a single multi-ligand SDF per thread and passed to '
                           'gnina in one run for efficiency.')
        inputGroup.addParam('pocketRadiusN', FloatParam, label='Grid radius: ',
                            default=1.2, allowsNull=False, condition=f'fromReceptor == {FROM_POCKET}',
                            help='The radius * n of each StructROI will be used as grid radius')

        # ---- Docking & Scoring ----------------------------------------- #
        form.addSection(label='Docking & Scoring')
        form.addParam('exhaustiveness', IntParam, default=8,
                      label='Exhaustiveness',
                      help='Exhaustiveness of the global Monte Carlo search. '
                           'For best performance set this equal to the number of '
                           'CPU threads.')
        form.addParam('numPoses', IntParam, default=10,
                      label='Number of binding modes',
                      help='Maximum number of docking poses to generate per ligand.')
        form.addParam('cnnScoring', EnumParam,
                      choices=CNN_SCORING_CHOICES,
                      default=CNN_SCORING_RESCORE,
                      label='CNN scoring mode',
                      help='Controls when the CNN is applied:\n\n'
                           '- *rescore* (default): CNN re-ranks final poses. Fastest.\n'
                           '- *refinement*: CNN refines poses (~10x slower).\n'
                           '- *metrorescore* / *metrorefine*: Metropolis variants.\n'
                           '- *none*: purely empirical scoring (fastest, no CNN).')
        form.addParam('scoring', EnumParam,
                      choices=SCORING_CHOICES,
                      default=SCORING_DEFAULT,
                      label='Empirical scoring function',
                      expertLevel=LEVEL_ADVANCED,
                      help='Empirical scoring function used when CNN is disabled '
                           'or for the non-CNN stage of the pipeline.')
        form.addParam('minRmsdFilter', FloatParam, default=1.0,
                      label='Min. RMSD filter (Å)',
                      expertLevel=LEVEL_ADVANCED,
                      help='Minimum RMSD between output poses to prune near-duplicate '
                           'conformations.')

        form.addParam('seed', IntParam, default=42,
                      label='Random seed',
                      expertLevel=LEVEL_ADVANCED,
                      help='Set to a positive integer for fully reproducible runs. Set to'
                            ' 0 if a random seed is wanted.')

        flexGroup = form.addGroup('Flexible residues')
        flexGroup.addParam('doFlexRes', BooleanParam, label='Add flexible residues: ', default=False,
                       help='Whether to add residues of the receptor which will be treated as flexible.')
        flexGroup.addParam('flexChain', StringParam, label='Residue chain: ', condition='doFlexRes',
                       help='Specify the protein chain.')
        flexGroup.addParam('flexRes', StringParam, label='Add defined residues', condition='doFlexRes',
                       help='Here you can define the flexible residues in a comma '
                            'separated list of chain:resid or using the wizard.')

        form.addParallelSection(threads=4, mpi=1)

    def _insertAllSteps(self):
        inMols  = self.inputSmallMolecules.get()
        nt      = self.numberOfThreads.get()
        gpuList = self._getGPUIds()
        subsets = makeSubsets(inMols, max(nt - 1, 1), cloneItem=True)
        print(subsets)

        self._insertFunctionStep(self.convertReceptorStep, needsGPU=False)

        self._insertFunctionStep(self.convertLigandsStep, nt, needsGPU=False)

        if self.fromReceptor.get() == FROM_PROTEIN:
            for it, _ in enumerate(subsets):
                self._insertFunctionStep(self.dockingStep,gpuList, it)
        else:
            for pocket in self.inputStructROIs.get():
                for it, _ in enumerate(subsets):
                    self._insertFunctionStep(self.dockingStep, gpuList, pocket.clone(), it)

        self._insertFunctionStep(
            self.createOutputStep, needsGPU=False)

    # ------------------------------------------------------------------ #
    #  Step functions                                                      #
    # ------------------------------------------------------------------ #

    def convertReceptorStep(self):
        receptorFile = self.getOriginalReceptorFile()
        self.other2pdbqt(receptorFile, self.getReceptorPDBQT())
          # gridId = self._insertFunctionStep(self.generateGridsStep, pocket.clone(), prerequisites=gridReqs, needsGPU=False)

    def convertLigandsStep(self, nt):
        """Merge a ligand subset into a single multi-ligand SDF for this thread."""
        inMols  = self.inputSmallMolecules.get()
        molsName = inMols.getName()
        subsets = makeSubsets(inMols, max(nt - 1, 1), cloneItem=True)
        for it, molSet in enumerate(subsets):
            outSdf = self._getSubsetLigandFile(it)
            with open(outSdf, 'w') as fout:
                for mol in inMols:
                    molFile = mol.getFileName()
                    print(molFile)
                    sdfFile = convertToSdf(self, molFile)
                    with open(sdfFile) as fin:
                        fout.write(fin.read())
                        print(outSdf)

    def dockingStep(self, gpuList, pocket=None, subsetId=0):
        """Run gnina for one distinct (ligand subset, pocket) target pair."""
        recFile = os.path.abspath(self.getReceptorPDBQT())
        ligFile = os.path.abspath(self._getSubsetLigandFile(subsetId))

        # Isolate directory context by both pocket ID and subset ID index
        outDir = os.path.abspath(self._getRunDir(pocket, subsetId))
        makePath(outDir)

        outFile = os.path.join(outDir, 'docked.sdf')
        logFile = os.path.join(outDir, 'gnina.log')
        flexPdbFile = os.path.join(outDir, 'outputr_eceptor.pdb') if self.doFlexRes.get() else None

        args = self._buildArgs(recFile, ligFile, outFile, logFile, gpuList, pocket)
        Plugin.runGnina(self, args, cwd=outDir)


    def createOutputStep(self):
        """Collect all docked SDF files and build a SetOfSmallMolecules output."""
        outDir = self._getPath('outputLigands')
        makePath(outDir)

        recFile   = self.getReceptorPDBQT()
        outputSet = SetOfSmallMolecules().create(outputPath=outDir)
        outputSet.copyInfo(self.inputSmallMolecules.get())
        outputSet.setProteinFile(recFile)
        outputSet.setDocked(True)

        # Parse every docked SDF into a nested lookup dict
        poseMap = {}

        if self.fromReceptor.get() == FROM_PROTEIN:
            searchPattern = self._getExtraPath('whole_receptor', 'subset_*', 'docked.sdf')
            for sdfFile in sorted(glob.glob(searchPattern)):
                self._parsePosesIntoMap(sdfFile, outDir, poseMap, pocketId=None)
        else:
            searchPattern = self._getExtraPath('pocket_*', 'subset_*', 'docked.sdf')
            for sdfFile in sorted(glob.glob(searchPattern)):
                pathParts   = sdfFile.split(os.sep)
                pocketDir   = [p for p in pathParts if p.startswith('pocket_')][-1]
                pocketId    = int(pocketDir.split('_')[1])
                self._parsePosesIntoMap(sdfFile, outDir, poseMap, pocketId=pocketId)

                # 2. For each input molecule, look up its poses and build output objects
                for smallMol in self.inputSmallMolecules.get():
                    molName = getBaseName(smallMol.getFileName())
                    for (pocketId, mn), poses in poseMap.items():
                        if mn != molName:
                            continue
                        for poseData in poses:
                            newMol = SmallMolecule()
                            newMol.copy(smallMol, copyId=False)
                            newMol.setPoseFile(os.path.relpath(poseData['poseFile']))
                            newMol.setPoseId(poseData['mode'])
                            newMol.setDockId(self.getObjId())
                            newMol.setMolClass('Gnina')
                            if pocketId is not None:
                                newMol.gridId.set(pocketId)
                            if poseData['minimizedAffinity'] is not None:
                                newMol._GninaEnergy = pwobj.Float(poseData['minimizedAffinity'])
                            if poseData['CNNscore'] is not None:
                                newMol._GninaCnnScore = pwobj.Float(poseData['CNNscore'])
                            if poseData['CNNaffinity'] is not None:
                                newMol._GninaCnnAffinity = pwobj.Float(poseData['CNNaffinity'])
                            outputSet.append(newMol)

                outputSet.saveGroupIndexes()
                self._defineOutputs(outputSmallMolecules=outputSet)
                self._defineSourceRelation(self.inputSmallMolecules, outputSet)

        outputSet.saveGroupIndexes()
        self._defineOutputs(outputSmallMolecules=outputSet)
        self._defineSourceRelation(self.inputSmallMolecules, outputSet)

    def _parsePosesIntoMap(self, sdfFile, outDir, poseMap, pocketId=None):
        """Split one docked SDF and accumulate pose dicts into poseMap."""
        subsetDir    = next(p for p in sdfFile.split(os.sep) if p.startswith('subset_'))
        prefix       = f"{'p' + str(pocketId) if pocketId is not None else 'wr'}_{subsetDir}_"

        for poseData in self.splitGninaSDF(sdfFile, outDir, prefix=prefix):
            poseData['minimizedAffinity'] = self.getTagValueFromSdf(poseData['poseFile'], 'minimizedAffinity')
            poseData['CNNscore']          = self.getTagValueFromSdf(poseData['poseFile'], 'CNNscore')
            poseData['CNNaffinity']       = self.getTagValueFromSdf(poseData['poseFile'], 'CNNaffinity')
            poseMap.setdefault((pocketId, poseData['molName']), []).append(poseData)

    # ------------------------------------------------------------------ #
    #  Argument building                                                   #
    # ------------------------------------------------------------------ #

    def _buildArgs(self, recFile, ligFile, outFile, logFile, gpuList, pocket=None):
        """Assemble the gnina command-line argument string."""
        args = f'-r "{recFile}" -l "{ligFile}" -o "{outFile}" --log "{logFile}"'

        if self.fromReceptor.get() == FROM_PROTEIN:
            args += f' --autobox_ligand "{recFile}" '
        elif pocket != None:
            minMaxCoords = pocket.getLimits()
            xCenter, yCenter, zCenter = pocket.calculateMassCenter()
            diams = [(minMax[1] - minMax[0]) * self.pocketRadiusN.get() for minMax in minMaxCoords]
            args += f' --center_x {xCenter} --center_y {yCenter} --center_z {zCenter}'
            args += f' --size_x {diams[0]} --size_y {diams[1]} --size_z {diams[2]}'

        # CNN scoring
        args += f' --cnn_scoring {CNN_SCORING_CHOICES[self.cnnScoring.get()]}'


        # Empirical scoring (only if non-default)
        scoringFn = SCORING_CHOICES[self.scoring.get()]
        if scoringFn != 'default':
            args += f' --scoring {scoringFn}'

        # Docking parameters
        args += f' --exhaustiveness {self.exhaustiveness.get()}'
        args += f' --num_modes {self.numPoses.get()}'
        args += f' --min_rmsd_filter {self.minRmsdFilter.get()}'

        seed = self.seed.get()
        if seed > 0:
            args += f' --seed {seed}'

        # Flexible residues
        flexRes = self.flexRes.get().strip()
        if flexRes:
            args += f' --flexres {flexRes}'

        # GPU / CPU
        if getattr(self, USE_GPU).get() and gpuList:
            args += f' --device {gpuList[0]}'
        else:
            args += ' --no_gpu'

        # CPU threads = exhaustiveness for optimal parallelism
        args += f' --cpu {self.exhaustiveness.get()}'

        return args

    # ------------------------------------------------------------------ #
    #  Output parsing                                                      #
    # ------------------------------------------------------------------ #

    def _getSubsetLigandFile(self, subsetId):
        return self._getExtraPath(f'ligands_subset_{subsetId}.sdf')

    def _getRunDir(self, pocket, subsetId):
        if pocket is None:
            return self._getExtraPath('whole_receptor', f'subset_{subsetId}')
        return self._getExtraPath(f'pocket_{self._getPocketId(pocket)}', f'subset_{subsetId}')

    def _parseDockingOutput(self, sdfFile, outputSet, pocketId=None):
        """Read a gnina output SDF and append each pose as a SmallMolecule."""
        with open(sdfFile) as f:
            content = f.read()

        blocks = [b.strip() for b in content.split('$$$$') if b.strip()]
        for posId, block in enumerate(blocks, start=1):
            poseFile = self._getExtraPath(
                f'pose_p{pocketId or "wr"}_{os.path.basename(sdfFile)[:-4]}_{posId}.sdf')
            with open(poseFile, 'w') as pf:
                pf.write(block + '\n$$$$\n')

            newMol = SmallMolecule()
            newMol.setFileName(os.path.relpath(poseFile))
            newMol.setPoseId(posId)
            newMol.setDockId(self.getObjId())

            if pocketId is not None:
                newMol.gridId.set(pocketId)

            cnnScore    = self._extractSDTag(block, 'CNNscore')
            cnnAffinity = self._extractSDTag(block, 'CNNaffinity')
            minAffinity = self._extractSDTag(block, 'minimizedAffinity')
            if cnnScore    is not None: newMol._cnnScore    = pwobj.Float(cnnScore)
            if cnnAffinity is not None: newMol._cnnAffinity = pwobj.Float(cnnAffinity)
            if minAffinity is not None: newMol._energy      = pwobj.Float(minAffinity)

            outputSet.append(newMol)

    @staticmethod
    def _extractSDTag(molBlock, tag):
        """Return the float value of an SDF data tag, or None if absent."""
        lines = molBlock.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == f'> <{tag}>' and i + 1 < len(lines):
                try:
                    return float(lines[i + 1].strip())
                except ValueError:
                    return None
        return None

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def getOriginalReceptorFile(self, getLink=True):
        recLink = self.getReceptorLink()
        if recLink is None or not getLink:
          if hasattr(self, 'inputAtomStruct') and \
                  (not hasattr(self, 'fromReceptor') or self.fromReceptor.get() == 0):
              recFile = self.inputAtomStruct.get().getFileName()
          elif hasattr(self, 'inputStructROIs') and \
                  (not hasattr(self, 'fromReceptor') or self.fromReceptor.get() == 1):
              recFile = self.inputStructROIs.get().getProteinFile()
          else:
              print('No original receptor file found')
              return None

          # Return a link so we avoid recursive threads into DB
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

    def getReceptorPDB(self):
        return os.path.abspath(self._getExtraPath(f'{self.getReceptorName()}.pdb'))

    def getReceptorName(self):
        fnReceptor = self.getOriginalReceptorFile()
        return getBaseName(fnReceptor)

    def getReceptorDir(self):
        fnReceptor = self.getOriginalReceptorFile()
        return os.path.dirname(fnReceptor)

    def getReceptorPDBQT(self):
        return os.path.abspath(self._getExtraPath(f'{self.getReceptorName()}.pdbqt'))

    def getReceptorLink(self):
      recDir = self._getExtraPath('originalReceptor')
      if os.path.exists(recDir):
        for file in os.listdir(recDir):
          return os.path.join(recDir, file)
      return None

    def _getReceptorFile(self):
        ext = os.path.splitext(self.inputAtomStruct.get().getFileName())[1].lower()
        return self._getExtraPath(f'receptor{ext}')

    def _getMergedLigandFile(self, name):
        return self._getExtraPath(f'ligands_{name}.sdf')

    def _getPocketId(self, pocket):
        return pocket.getObjId() if pocket is not None else None

    def _getPocketDir(self, pocket):
        if pocket is None:
            return self._getExtraPath('whole_receptor')
        return self._getExtraPath(f'pocket_{self._getPocketId(pocket)}')

    def other2pdbqt(self, otherFile, pdbqtFile):
        '''Convert pdb or others to pdbqt using openbabel (better for AtomStruct)'''
        inExt = os.path.splitext(os.path.basename(otherFile))[1]

        if inExt not in ['.pdb', '.mol2', '.sdf', '.mol', '.cif']:
            inFormat = 'pdb'
        else:
            inFormat = inExt[1:]

        args = ' -i{} {} -opdbqt -O {}'.format(inFormat, os.path.abspath(otherFile), os.path.abspath(pdbqtFile))
        runOpenBabel(self, args=args, popen=True)

        return os.path.abspath(pdbqtFile)

    def splitGninaSDF(self, sdfFile, outDir, prefix=''):
        """Split a multi-ligand output SDF into isolated subset molecule files safely."""
        with open(sdfFile) as fh:
            content = fh.read()

        rawBlocks = [b.strip() for b in content.split('$$$$') if b.strip()]
        poses = []
        molCounts = {}

        for globalIdx, block in enumerate(rawBlocks):
            lines = block.splitlines()
            molName = lines[0].strip() if lines else f'mol_{globalIdx}'

            molCounts[molName] = molCounts.get(molName, 0) + 1
            mode = molCounts[molName]

            fname = f'{prefix}{molName}_{mode}.sdf'
            poseFile = os.path.join(outDir, fname)
            with open(poseFile, 'w') as fh:
                fh.write(block + '\n$$$$\n')

            poses.append({
                'molName': molName,
                'molIdx': globalIdx,
                'mode': mode,
                'poseFile': os.path.abspath(poseFile),
            })
        return poses

    def _getGPUIds(self):
        gpus = []
        for gp in getattr(self, GPU_LIST).get().split(','):
            gpus.append(str(int(gp.strip())))
        return gpus

    def getSdfFile(self):
        searchPattern = self._getExtraPath('*.sdf')

        # glob.glob returns a list of all files that match the pattern
        matches = glob.glob(searchPattern)

        if matches:
            return matches[0]  # Return the first matching file it found
        else:
            return None

    def getDockedDir(self):
        path = self._getExtraPath('docked_poses')
        os.makedirs(path, exist_ok=True)
        return path

    def getLogFile(self):
        return os.path.abspath(self._getPath('log.log'))

    def getTagValueFromSdf(sefl, sdfFile, tag):
        """Return the float value of an SDF data tag from a pose file.
        """
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
    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    # def validate(self):
    #     errors = []
    #     if not os.path.isfile(Plugin.getGninaBinary()):
    #         errors.append(
    #             'gnina binary not found at: %s\n'
    #             'Please install via "scipion installp -p scipion-chem-gnina --devel".'
    #             % Plugin.getGninaBinary()
    #         )
    #     if self.fromReceptor.get() == FROM_PROTEIN:
    #         if self.boxMode.get() == BOX_MODE_AUTOBOX and not self.autoboxLigand.get():
    #             errors.append('Autobox mode requires a reference ligand/structure.')
    #     else:
    #         if not self.inputStructROIs.get():
    #             errors.append('Structural ROI mode requires a SetOfStructROIs input.')
    #     if self.numPoses.get() < 1:
    #         errors.append('Number of binding modes must be >= 1.')
    #     if self.exhaustiveness.get() < 1:
    #         errors.append('Exhaustiveness must be >= 1.')
    #     return errors

    # ------------------------------------------------------------------ #
    #  Summary / methods                                                   #
    # ------------------------------------------------------------------ #

    def _summary(self):
        summary = []
        if self.inputAtomStruct.get():
            summary.append(f'Receptor: {os.path.basename(self.inputAtomStruct.get().getFileName())}')
        if self.inputSmallMolecules.get():
            summary.append(f'Ligands: {self.inputSmallMolecules.get().getSize()} molecule(s)')
        if self.fromReceptor.get() == FROM_POCKET and self.inputStructROIs.get():
            summary.append(f'Pockets: {self.inputStructROIs.get().getSize()} ROI(s)')
        summary.append(f'CNN scoring: {CNN_SCORING_CHOICES[self.cnnScoring.get()]}')
        summary.append(f'Exhaustiveness: {self.exhaustiveness.get()}')
        summary.append(f'Modes: {self.numPoses.get()}')
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