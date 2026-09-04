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

import os, glob, math

from pyworkflow.protocol.params import (
    PointerParam, EnumParam, IntParam, FloatParam, BooleanParam,
    StringParam, LEVEL_ADVANCED, STEPS_PARALLEL, USE_GPU, GPU_LIST,
)
from pyworkflow.utils.path import makePath
import pyworkflow.object as pwobj

from pwchem.objects import SetOfSmallMolecules, SmallMolecule
from pwchem.utils import getBaseName, parseAtomStruct, runOpenBabel

from .. import Plugin
from ..constants import *
from .protocol_gnina import ProtGninaDocking, FROM_PROTEIN, FROM_POCKET, CIFext

# Per-pose receptor+ligand complexes, with the covalent bond as a CONECT record
GNINA_COMPLEX_DIR = 'outputComplexes'
# A covalent C-S bond is ~1.8 A; the next ligand atom out is ~2.8 A, so this
# cutoff identifies the bonded atom unambiguously.
COVALENT_BOND_CUTOFF = 2.2


class ProtGninaCovalentDocking(ProtGninaDocking):
    """Perform covalent molecular docking using GNINA.

    The ligand is bonded to a receptor atom and only its own torsions are
    searched around that anchor, instead of the free 6D + torsional search
    the plain docking protocol performs.

    Use it for electrophilic warheads that react with a nucleophilic residue
    (cysteine thiol, serine hydroxyl, lysine amine...). Supply the *unreacted*
    ligand: gnina forms the bond itself, so a ligand whose warhead is already
    saturated (e.g. one extracted from a covalent complex in the PDB) has
    nothing left to react and is skipped.

    CNN scoring is not used: gnina reports it as not calibrated for covalent
    complexes, so poses are ranked by the empirical energy instead.

    The bond only exists in gnina's --out_flex output, never in the ligand
    SDF, so each output pose carries its own receptor (built from that output)
    with the covalently modified residue, reachable through getProteinFile().

    References:
      McNutt et al., J. Cheminformatics 2021 (GNINA 1.0)
      McNutt et al., J. Cheminformatics 2025 (GNINA 1.3)
    """

    _label = 'GNINA covalent docking'
    stepsExecutionMode = STEPS_PARALLEL

    # ------------------------------------------------------------------ #
    #  Form definition                                                     #
    # ------------------------------------------------------------------ #
    def _defineParams(self, form):
        self._defineGpuParams(
            form, gpuHelp="GNINA can use CUDA GPUs to accelerate execution. If disabled, "
                          "gnina runs on CPU.")
        self._defineInputParams(
            form, fromReceptorDefault=FROM_PROTEIN,
            fromReceptorHelp='Whole protein autoboxes the receptor. Note the covalent atom must '
                             'fall inside the search box, so with ROIs pick the pocket that '
                             'contains the reacting residue.',
            ligandHelp='Ligands must carry an unreacted warhead matching the SMARTS below.')

        # ---- Covalent bond --------------------------------------------- #
        form.addSection(label='Covalent bond')
        form.addParam('covalentRecAtom', StringParam, label='Receptor atom: ', default='',
                      help='Receptor atom the ligand binds to, either as\n\n'
                           '- *chain:resnum:atom_name*, e.g. A:13:SG for the sulfur of Cys13 in chain A\n'
                           '- *x,y,z* Cartesian coordinates, e.g. 12.34,45.67,7.89\n\n'
                           'Give the numbering of the *input* structure.')
        form.addParam('covalentLigPattern', StringParam, label='Ligand warhead (SMARTS): ', default='',
                      help='SMARTS matching the ligand warhead.\n\n'
                           'The *first* atom of the match is the one bonded to the receptor, so write '
                           'the reacting atom first. Examples:\n\n'
                           + '\n'.join(f'- {name}: {smarts}'
                                       for name, smarts in COVALENT_WARHEAD_EXAMPLES))
        form.addParam('covalentOptimizeLig', BooleanParam, label='Optimize covalent complex: ',
                      default=True,
                      help='Relax the ligand with UFF once bonded (--covalent_optimize_lig), which '
                           'changes its bond angles and lengths.\n\n'
                           'Recommended: the bond is formed geometrically, so without this step the '
                           'junction stays strained and poses score badly (positive affinities).\n\n'
                           'Needs Open Babel force-field data. If the log shows "Cannot open UFF.prm" '
                           'the optimisation silently did nothing: point BABEL_DATADIR at an Open '
                           'Babel data directory before re-running.')
        form.addParam('covalentBondOrder', EnumParam, choices=COVALENT_BOND_ORDER_CHOICES,
                      label='Covalent bond order: ', default=0, expertLevel=LEVEL_ADVANCED,
                      help='Bond order of the new receptor-ligand bond.\n\n'
                           '- *1*: nearly every real warhead. Thiol-Michael addition, SN2 '
                           '(chloroacetamide), epoxide opening, boronic esters.\n'
                           '- *2*: Schiff base / imine formation, e.g. an aldehyde reacting with '
                           'a lysine amine.\n'
                           '- *3*: no realistic protein-ligand chemistry; offered for completeness.')
        form.addParam('covalentLigPosition', StringParam, label='Initial warhead position: ',
                      default='', expertLevel=LEVEL_ADVANCED,
                      help='Optional *x,y,z* starting placement for the bonding ligand atom '
                           "(--covalent_lig_atom_position). Left empty, gnina places the ligand with "
                           "Open Babel's GetNewBondVector.")
        form.addParam('covalentFixLigPosition', BooleanParam, label='Fix warhead at that position: ',
                      condition='covalentLigPosition != ""', default=False,
                      expertLevel=LEVEL_ADVANCED,
                      help='Keep the bonding ligand atom pinned at the position above for the whole '
                           'run (--covalent_fix_lig_atom_position) instead of using it only as the '
                           'initial structure.\n\n'
                           'Use it when the attachment geometry is known (crystallography, QM) and '
                           'must be respected; otherwise leave it off so the search can relax the '
                           'junction.')

        # ---- Search & Scoring ------------------------------------------ #
        form.addSection(label='Search & Scoring')
        form.addParam('exhaustiveness', IntParam, default=8, label='Exhaustiveness: ',
                      help='Exhaustiveness of the Monte-Carlo search (roughly proportional to time).')
        form.addParam('numPoses', IntParam, default=9, label='Number of binding modes: ',
                      help='Maximum number of docking poses to generate per ligand (--num_modes).')
        form.addParam('minRmsdFilter', FloatParam, default=1.0, label='Min. RMSD filter (Å): ',
                      expertLevel=LEVEL_ADVANCED,
                      help='Minimum RMSD between output poses to prune near-duplicate conformations. '
                           'Covalent poses cluster more tightly than free ones, since one end of the '
                           'ligand is pinned.')
        form.addParam('scoring', EnumParam, choices=SCORING_CHOICES, default=SCORING_DEFAULT,
                      label='Empirical scoring function: ', expertLevel=LEVEL_ADVANCED,
                      help='Empirical scoring function used to rank the poses.\n\n'
                           'Note none of these model the covalent bond itself: they score the '
                           'non-covalent contacts of the rest of the ligand around a fixed anchor, '
                           'so the value is not a covalent binding affinity.')
        form.addParam('addH', BooleanParam, default=False, label='Add hydrogens to ligands: ',
                      expertLevel=LEVEL_ADVANCED,
                      help='Let gnina automatically add hydrogens to the ligands.')
        form.addParam('seed', IntParam, default=42, label='Random seed: ', expertLevel=LEVEL_ADVANCED,
                      help='Set to a positive integer for reproducible runs. Set to 0 for a random seed.')

        form.addParallelSection(threads=4, mpi=1)

    # ------------------------------------------------------------------ #
    #  Warhead pre-filter                                                  #
    # ------------------------------------------------------------------ #
    def convertLigandsStep(self, molSet, it):
        """Convert this subset, then drop the molecules with no warhead"""
        super().convertLigandsStep(molSet, it)
        self._filterWarheadSubset(it)

    def dockingStep(self, subsetId, pocket=None):
        """Dock this subset, unless the filter left nothing in it"""
        if not self._sdfTitles(self._getSubsetLigandFile(subsetId)):
            print(f'Ligand subset {subsetId} holds no molecule carrying the warhead; '
                  f'not calling gnina for it.')
            return
        super().dockingStep(subsetId, pocket)

    def _filterWarheadSubset(self, subsetId):
        """Keep in the subset SDF only what matches the warhead SMARTS"""
        ligFile = os.path.abspath(self._getSubsetLigandFile(subsetId))
        before = self._sdfTitles(ligFile)
        if not before:
            return

        smarts = self.covalentLigPattern.get().strip()
        filteredFile = f'{os.path.splitext(ligFile)[0]}_warhead.sdf'
        runOpenBabel(self, args=f'{ligFile} -osdf -O {filteredFile} -s "{smarts}"', popen=True)
        # Nothing matching still writes a file, an empty one; no file means Open
        # Babel did not run, so dock the subset and let gnina do the skipping.
        if not os.path.exists(filteredFile):
            print(f'Warning: Open Babel produced no output for subset {subsetId}; '
                  f'docking it unfiltered.')
            return

        after = self._sdfTitles(filteredFile)
        os.replace(filteredFile, ligFile)
        if len(after) < len(before):
            print(f'Subset {subsetId}: {len(after)} of {len(before)} molecule(s) match {smarts}.')

    def getWarheadNames(self):
        """(docked, not docked) molecule names.

        Read back from the subset files: what survived the filter is precisely
        what is left in them, and the input set holds the rest. A subset that
        could not be filtered still holds everything, and counting those as
        docked is right - they were.
        """
        kept = []
        for subsetId in range(max(self.numberOfThreads.get() - 1, 1)):
            kept += self._sdfTitles(self._getSubsetLigandFile(subsetId))

        keptNames = set(kept)
        dropped = [getBaseName(mol.getFileName()) for mol in self.inputSmallMolecules.get()
                   if getBaseName(mol.getFileName()) not in keptNames]
        return kept, dropped

    def getGninaSkippedNames(self):
        """Molecules gnina itself reported as not matching the SMARTS.

        Should always be empty, since the filter removed them first. Anything
        here is the filter and gnina disagreeing, which must stay visible.
        """
        # The warning goes to the console, not into --log, hence the stdout too.
        logFiles = glob.glob(self._getExtraPath('*', 'subset_*', 'gnina.log'))
        try:
            logFiles += [path for path in self.getLogPaths() if isinstance(path, str)]
        except Exception:
            pass

        skipped = set()
        for logFile in logFiles:
            if not os.path.isfile(logFile):
                continue
            with open(logFile, errors='ignore') as fIn:
                for line in fIn:
                    if 'did not match covalent_lig_atom_pattern' in line:
                        # 'WARNING: Ligand <name> did not match ...'
                        molName = line.split('Ligand', 1)[-1].split('did not match')[0].strip()
                        if molName:
                            skipped.add(molName)
        return sorted(skipped)

    @staticmethod
    def _sdfTitles(sdfFile):
        """Names of the molecules in a multi-molecule SDF, in file order"""
        if not os.path.exists(sdfFile):
            return []

        titles = []
        with open(sdfFile) as fIn:
            for block in fIn.read().split('$$$$'):
                block = block.lstrip('\n')
                if block.strip():
                    titles.append(block.split('\n', 1)[0].strip())
        return titles

    # ------------------------------------------------------------------ #
    #  Output                                                              #
    # ------------------------------------------------------------------ #
    def createOutputStep(self):
        """Collect the docked poses, adding the covalent complex column.

        A copy of the docking protocol's version rather than a hook into it, so
        nothing covalent leaks there. Keep the two in step.
        """
        outDir = self._getPath('outputLigands')
        makePath(outDir)

        recFile = self.getReceptorPDBQT()
        outputSet = SetOfSmallMolecules().create(outputPath=self._getPath())

        inputMolsDic = {getBaseName(mol.getFileName()): mol.clone()
                        for mol in self.inputSmallMolecules.get()}

        kept, dropped = self.getWarheadNames()
        if dropped and not kept:
            # Say so, rather than define an empty output that reads as a
            # docking which found nothing.
            raise Exception(
                f'None of the {len(dropped)} input molecules matches the warhead SMARTS '
                f'{self.covalentLigPattern.get()}, so there was nothing to dock.\n\n'
                f'A ligand taken from a covalent complex is already in its reacted form and no '
                f'longer carries the warhead: dock the free compound instead. The wizard on the '
                f'SMARTS field checks a pattern against the input molecules before a run.')

        recDir = self._getPath('outputReceptors')
        sdfFiles = sorted(glob.glob(self._getExtraPath('*', 'subset_*', GNINA_OUTPUT_SDF)))
        for sdfFile in sdfFiles:
            pocketId = self._pocketIdFromPath(sdfFile)
            gridId = pocketId if pocketId is not None else 1
            prefix = f'g{gridId}_'

            poses = self.splitGninaSDF(sdfFile, outDir, prefix=prefix)

            # Per-pose receptors, and the complexes along the way.
            poseRecFiles = self._buildFlexReceptors(sdfFile, poses, recFile, recDir)

            for poseIdx, poseData in enumerate(poses):
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
                newMol.setProteinFile(os.path.relpath(
                    poseRecFiles[poseIdx] if poseRecFiles else recFile))

                # Set even when gnina reported no tag: see below.
                newMol._gninaEnergy = pwobj.Float(
                    self.getTagValueFromSdf(poseData['poseFile'], 'minimizedAffinity'))
                newMol._gninaCnnScore = pwobj.Float(
                    self.getTagValueFromSdf(poseData['poseFile'], 'CNNscore'))
                newMol._gninaCnnAffinity = pwobj.Float(
                    self.getTagValueFromSdf(poseData['poseFile'], 'CNNaffinity'))

                # Receptor + pose in one PDB, the bond as a CONECT record. Set
                # on every pose even when missing: a Set fixes its columns from
                # the first item, so a later item without it aborts the insert.
                complexFile = self._complexFilePath(poseData['poseFile'])
                newMol.covalentPoseFile = pwobj.String(
                    os.path.relpath(complexFile) if os.path.exists(complexFile) else None)

                outputSet.append(newMol)

        outputSet.updateMolClass()
        outputSet.setProteinFile(os.path.relpath(recFile))
        outputSet.setDocked(True)
        self._defineOutputs(outputSmallMolecules=outputSet)
        self._defineSourceRelation(self.inputSmallMolecules, outputSet)

        # Not an output set: counted in the summary, named in a file of their
        # own. A screening library leaves thousands behind, far too many to log.
        if dropped:
            with open(self._getNotDockedFile(), 'w') as fOut:
                fOut.write('\n'.join(dropped) + '\n')
            print(f'{len(dropped)} molecule(s) were not docked, having no warhead matching '
                  f'{self.covalentLigPattern.get()}; named in '
                  f'{os.path.basename(self._getNotDockedFile())}')

    def _getNotDockedFile(self):
        return self._getExtraPath('notDocked.txt')

    # ------------------------------------------------------------------ #
    #  Argument building                                                   #
    # ------------------------------------------------------------------ #
    def _buildArgs(self, recFile, ligFile, outFile, logFile, pocket=None):
        """Assemble the gnina command line for a covalent run."""
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

        # Covalent bond
        args += f' --covalent_rec_atom {self.getCovalentRecAtom()}'
        # Quoted: a SMARTS carries characters the shell would eat.
        args += f' --covalent_lig_atom_pattern "{self.covalentLigPattern.get().strip()}"'
        if self.covalentOptimizeLig.get():
            args += ' --covalent_optimize_lig'
        # EnumParam stores the index, not the order itself.
        bondOrder = COVALENT_BOND_ORDER_CHOICES[self.covalentBondOrder.get()]
        if bondOrder != '1':
            args += f' --covalent_bond_order {bondOrder}'
        ligPos = self.covalentLigPosition.get()
        if ligPos and ligPos.strip():
            args += f' --covalent_lig_atom_position {ligPos.strip()}'
            if self.covalentFixLigPosition.get():
                args += ' --covalent_fix_lig_atom_position'

        # The CNN is not calibrated for covalent docking, so it is off and the
        # ranking falls back on the empirical energy.
        args += f' --cnn_scoring {COVALENT_CNN_SCORING}'
        args += f' --pose_sort_order {COVALENT_SORT_ORDER}'
        args += f' --scoring {SCORING_CHOICES[self.scoring.get()]}'

        # Search parameters
        args += f' --exhaustiveness {self.exhaustiveness.get()}'
        args += f' --num_modes {self.numPoses.get()}'
        args += f' --min_rmsd_filter {self.minRmsdFilter.get()}'

        if not self.addH.get():
            args += ' --addH 0'

        seed = self.seed.get()
        if seed > 0:
            args += f' --seed {seed}'

        # The only place the bond is written: --full_flex_output merges residue
        # and ligand into one connected fragment per pose.
        args += ' --full_flex_output'
        args += f' --out_flex "{os.path.join(os.path.dirname(outFile), GNINA_FLEX_PDBQT)}"'

        # The device is chosen with CUDA_VISIBLE_DEVICES in Plugin.runGnina:
        # gnina 1.3.2's Torch backend ignores --device.
        if not getattr(self, USE_GPU).get():
            args += ' --no_gpu'

        args += f' --cpu {self.exhaustiveness.get()}'
        return args

    # ------------------------------------------------------------------ #
    #  Output parsing                                                      #
    # ------------------------------------------------------------------ #
    def splitGninaSDF(self, sdfFile, outDir, prefix=''):
        """Split a multi-ligand covalent output SDF into one file per pose.

        As the docking protocol's, but the name is stripped of the underscore
        gnina prefixes: it names the complex '<receptor>_<ligand>' and the
        receptor part is empty. Unstripped, no pose matches its input ligand and
        the whole output set comes out empty.

        Returns dicts of molName / mode (pose index per molecule) / poseFile.
        """
        with open(sdfFile) as fh:
            content = fh.read()

        rawBlocks = [b.strip() for b in content.split('$$$$') if b.strip()]
        poses, molCounts = [], {}
        for globalIdx, block in enumerate(rawBlocks):
            lines = block.splitlines()
            molName = lines[0].strip() if lines else f'mol_{globalIdx}'
            molName = molName.lstrip('_')

            molCounts[molName] = molCounts.get(molName, 0) + 1
            mode = molCounts[molName]

            poseFile = os.path.join(outDir, f'{prefix}{molName}_{mode}.sdf')
            with open(poseFile, 'w') as fh:
                fh.write(block + '\n$$$$\n')

            poses.append({'molName': molName, 'mode': mode,
                          'poseFile': os.path.abspath(poseFile)})
        return poses

    # ------------------------------------------------------------------ #
    #  Per-pose receptors                                                  #
    # ------------------------------------------------------------------ #
    def _buildFlexReceptors(self, sdfFile, poses, rigidRecFile, recDir):
        """Build one receptor file per pose from the covalent --out_flex output.

        Not parsable like the flexible-residue output: --full_flex_output writes
        ligand and bonded residue as one fragment per MODEL, ligand atoms first,
        and labels the receptor side 'UNK 1' with no chain or residue number. So
        atoms are matched to the rigid receptor by element and position, and the
        split point is the ligand's atom count taken from the pose SDF (both use
        the same united-atom convention).

        Only the receptor side is written: the ligand stays in its pose file, so
        the two meet at the bond with no atom duplicated.

        Returns receptor paths aligned with `poses`, or None if the output does
        not match them.
        """
        flexFile = os.path.join(os.path.dirname(sdfFile), GNINA_FLEX_PDBQT)
        if not os.path.exists(flexFile):
            print(f'Warning: no covalent receptor output at {flexFile}; '
                  f'poses will reference the rigid receptor.')
            return None

        # MODEL blocks in file order, which is the pose order of the SDF.
        groups, curId, curAtoms = [], None, []
        for line in open(flexFile):
            if line.startswith('MODEL'):
                modelId = line.split()[1] if len(line.split()) > 1 else ''
                if curId is not None and modelId != curId:
                    groups.append(curAtoms)
                    curAtoms = []
                curId = modelId
            elif line.startswith(('ATOM', 'HETATM')):
                curAtoms.append(line)
        if curAtoms:
            groups.append(curAtoms)

        if len(groups) != len(poses):
            print(f'Warning: {len(groups)} covalent-receptor group(s) for '
                  f'{len(poses)} pose(s) in {flexFile}; poses will reference the '
                  f'rigid receptor.')
            return None

        rigidLines = list(open(rigidRecFile))
        makePath(recDir)
        recFiles = []
        for poseData, atomLines in zip(poses, groups):
            # Split by position, not by count: gnina writes the fragment in its
            # own order, which interleaves the residue among the ligand atoms
            # rather than appending it. Taking the last atoms instead put ligand
            # atoms in the receptor, duplicating them at the same coordinates.
            ligAtoms, _ = self._readSdfMol(poseData['poseFile'])
            recSide = [line for line in atomLines if not self._isLigandAtom(line, ligAtoms)]
            if not ligAtoms or not recSide or len(recSide) == len(atomLines):
                print(f'Warning: cannot split the covalent fragment of '
                      f'{poseData["poseFile"]} into ligand and receptor '
                      f'({len(atomLines) - len(recSide)} of {len(atomLines)} fragment atoms '
                      f'matched the pose); poses will reference the rigid receptor.')
                return None

            # Same element within 0.6 A. gnina held the residue fixed in every
            # run tested, so this is usually a no-op, but survives one that moves it.
            moved, extra, taken = {}, [], set()
            for line in recSide:
                elem, xyz = self._pdbqtElement(line), self._pdbqtCoords(line)
                if xyz is None:
                    continue
                best, bestD2 = None, 0.36
                for idx, rLine in enumerate(rigidLines):
                    if idx in taken or not rLine.startswith(('ATOM', 'HETATM')):
                        continue
                    if self._pdbqtElement(rLine) != elem:
                        continue
                    rXyz = self._pdbqtCoords(rLine)
                    if rXyz is None:
                        continue
                    d2 = sum((a - b) ** 2 for a, b in zip(xyz, rXyz))
                    if d2 < bestD2:
                        best, bestD2 = idx, d2
                if best is not None:
                    moved[best] = line[30:54]
                    taken.add(best)
                else:
                    extra.append(line)

            outRec = os.path.join(recDir, f'{getBaseName(poseData["poseFile"])}_rec.pdbqt')
            kept, lastSerial = [], 0
            for idx, line in enumerate(rigidLines):
                if line.startswith(('ATOM', 'HETATM')):
                    if idx in moved:
                        line = line[:30] + moved[idx] + line[54:]
                    try:
                        lastSerial = max(lastSerial, int(line[6:11]))
                    except ValueError:
                        pass
                kept.append(line)

            # Appended rather than dropped: gnina protonates the residue, so it
            # can carry polar hydrogens the '-xr' receptor lacks.
            appended = []
            for line in extra:
                lastSerial += 1
                line = f'{line[:6]}{lastSerial:>5}{line[11:]}'
                # gnina labels them 'UNK 1' with no chain, which would make a
                # residue of their own and break the chain. Take the identity of
                # the closest receptor atom: a polar hydrogen is never far from
                # the atom it protonates.
                host = self._closestRigidAtom(line, rigidLines)
                if host is not None:
                    line = f'{line[:17]}{host[17:27]}{line[27:]}'
                appended.append(line)

            with open(outRec, 'w') as fOut:
                fOut.writelines(kept + appended)
            recFiles.append(os.path.abspath(outRec))

            self._writeCovalentComplex(poseData, kept + appended)
        return recFiles

    def _closestRigidAtom(self, line, rigidLines, maxDist=2.0):
        """Rigid-receptor atom line closest to `line`, or None beyond maxDist."""
        xyz = self._pdbqtCoords(line)
        if xyz is None:
            return None

        best, bestD2 = None, maxDist ** 2
        for rLine in rigidLines:
            if not rLine.startswith(('ATOM', 'HETATM')):
                continue
            rXyz = self._pdbqtCoords(rLine)
            if rXyz is None:
                continue
            d2 = sum((a - b) ** 2 for a, b in zip(xyz, rXyz))
            if d2 < bestD2:
                best, bestD2 = rLine, d2
        return best

    # ------------------------------------------------------------------ #
    #  Covalent complex files                                              #
    # ------------------------------------------------------------------ #
    def _writeCovalentComplex(self, poseData, recLines):
        """Write receptor + pose as one PDB whose CONECT record is the bond.

        Neither input file carries it: the pose SDF has no link to the receptor,
        and the receptor has no ligand. Viewers honour CONECT, so the bond then
        needs no viewer code.

        Returns the path written, or None if the bond could not be resolved.
        """
        ligAtoms, ligBonds = self._readSdfMol(poseData['poseFile'])
        if not ligAtoms:
            print(f'Warning: no atoms read from {poseData["poseFile"]}; '
                  f'no covalent complex written.')
            return None

        recAtoms = [ln for ln in recLines if ln.startswith(('ATOM', 'HETATM'))]
        recIdx = self._findRecAtomIndex(recAtoms)
        if recIdx is None:
            print(f'Warning: receptor atom "{self.covalentRecAtom.get()}" not found in the '
                  f'per-pose receptor; no covalent complex written.')
            return None
        recXyz = self._pdbqtCoords(recAtoms[recIdx])

        complexDir = self._getPath(GNINA_COMPLEX_DIR)
        makePath(complexDir)
        outFile = self._complexFilePath(poseData['poseFile'])

        # Renumbered from 1: a PDBQT restarts serials after every TER, so the
        # original ones cannot address an atom uniquely in a CONECT.
        lines, serial, recSerial = [], 0, None
        for idx, line in enumerate(recAtoms):
            serial += 1
            if idx == recIdx:
                recSerial = serial
            lines.append(self._pdbAtomLine(line, serial))
        lines.append('TER\n')

        # The ligand as its own HETATM residue, so it collides with no receptor
        # chain. The nearest of its atoms to the receptor atom is the bonded one.
        ligSerial, bestD2, elemCounts = None, COVALENT_BOND_CUTOFF ** 2, {}
        ligSerials = []
        for x, y, z, element in ligAtoms:
            serial += 1
            ligSerials.append(serial)
            elemCounts[element] = elemCounts.get(element, 0) + 1
            name = f'{element}{elemCounts[element]}'
            if len(element) == 1:
                name = f' {name}'
            lines.append(f'HETATM{serial:>5} {name[:4]:<4} LIG Z   1    '
                         f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          '
                         f'{element:>2}\n')
            if recXyz is not None:
                d2 = (x - recXyz[0]) ** 2 + (y - recXyz[1]) ** 2 + (z - recXyz[2]) ** 2
                if d2 < bestD2:
                    ligSerial, bestD2 = serial, d2
        lines.append('TER\n')

        # The ligand's own bonds go in too: a reader that finds any CONECT for a
        # HETATM residue takes them as its complete connectivity, so a file with
        # only the covalent link draws loose atoms. Receptor bonds are left out,
        # as in a deposited entry: they come from the reader's own templates.
        neighbours = {}
        for first, second in ligBonds:
            firstSerial, secondSerial = ligSerials[first - 1], ligSerials[second - 1]
            neighbours.setdefault(firstSerial, []).append(secondSerial)
            neighbours.setdefault(secondSerial, []).append(firstSerial)

        if ligSerial is None:
            # Report the geometry rather than invent a bond, and write the file
            # anyway so it can be looked at.
            print(f'Warning: no ligand atom within {COVALENT_BOND_CUTOFF} A of '
                  f'{self.covalentRecAtom.get()} for {getBaseName(poseData["poseFile"])}; '
                  f'the complex is written without a bond record.')
        else:
            # Both directions: some readers only follow the first field.
            neighbours.setdefault(recSerial, []).append(ligSerial)
            neighbours.setdefault(ligSerial, []).append(recSerial)

        lines += self._conectLines(neighbours)
        lines.append('END\n')

        with open(outFile, 'w') as fOut:
            fOut.writelines(lines)
        return os.path.abspath(outFile)

    def _complexFilePath(self, poseFile):
        """Where the covalent complex of a pose lives (whether or not written)."""
        return self._getPath(GNINA_COMPLEX_DIR,
                             f'{getBaseName(poseFile)}_cov.pdb')

    def getComplexFile(self, mol):
        """Path of the covalent complex PDB of a pose, or None if not written."""
        complexFile = self._complexFilePath(mol.getPoseFile())
        return complexFile if os.path.exists(complexFile) else None

    def getCovalentRecAtom(self):
        """The receptor atom address in the numbering gnina actually sees.

        Open Babel renumbers residues from 1 when it writes the PDBQT the
        protocol docks against, so a structure whose numbering does not already
        start at 1 comes out shifted: 6OIM opens with an expression-tag Gly
        numbered 0, which turns its Cys 12 into Cys 13, and gnina then stops
        with 'Could not find receptor atom'.

        The form value therefore names the atom as it is numbered in the input
        structure - what the user reads in their own file, and what the wizard
        offers - and it is translated here by matching coordinates, which is
        exact: the conversion renumbers residues but moves no atom.
        """
        recAtom = (self.covalentRecAtom.get() or '').strip()
        if getattr(self, '_recAtomCache', None) is not None:
            return self._recAtomCache

        resolved = recAtom
        # Coordinates need no translation, and neither does a receptor that was
        # already given as a PDBQT (no conversion took place).
        if recAtom.count(':') == 2:
            xyz = self._inputAtomCoords(recAtom)
            if xyz is None:
                print(f'Warning: receptor atom {recAtom} was not found in the input structure, so '
                      f'it cannot be checked against the converted receptor; passing it to gnina '
                      f'unchanged.')
            else:
                match = self._pdbqtAtomAt(self.getReceptorPDBQT(), xyz)
                if match is None:
                    print(f'Warning: no atom of the converted receptor sits at the position of '
                          f'{recAtom}; passing it to gnina unchanged.')
                elif match != recAtom:
                    print(f'Receptor atom {recAtom} of the input structure is {match} in the '
                          f'converted receptor (Open Babel renumbers residues from 1); '
                          f'{match} is what gnina is given.')
                    resolved = match

        self._recAtomCache = resolved
        return resolved

    def _inputAtomCoords(self, recAtom):
        """Coordinates of chain:resnum:atom_name in the *input* receptor file"""
        chain, resNum, atomName = [s.strip() for s in recAtom.split(':')]
        recFile = self.getOriginalReceptorFile()
        if recFile is None:
            return None

        if recFile.endswith(CIFext):
            structure = parseAtomStruct(os.path.abspath(recFile))
            if structure is None:
                return None
            for model in structure:
                for ch in model:
                    if chain and ch.get_id() != chain:
                        continue
                    for res in ch:
                        if str(res.get_id()[1]) != resNum:
                            continue
                        for atom in res:
                            if atom.get_id() == atomName:
                                return tuple(float(c) for c in atom.get_coord())
            return None

        # PDB and PDBQT share the ATOM record columns
        for line in open(os.path.abspath(recFile)):
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            name, lineChain, lineRes = self._pdbqtAtomKey(line)
            if name == atomName and lineRes == resNum and (lineChain == chain or not chain):
                return self._pdbqtCoords(line)
        return None

    @classmethod
    def _pdbqtAtomAt(cls, pdbqtFile, xyz, tol=0.05):
        """chain:resnum:atom_name of the atom of `pdbqtFile` sitting at `xyz`"""
        if not os.path.exists(pdbqtFile):
            return None

        best, bestD2 = None, tol ** 2
        for line in open(pdbqtFile):
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            lineXyz = cls._pdbqtCoords(line)
            if lineXyz is None:
                continue
            d2 = sum((a - b) ** 2 for a, b in zip(xyz, lineXyz))
            if d2 < bestD2:
                name, chain, resNum = cls._pdbqtAtomKey(line)
                best, bestD2 = f'{chain}:{resNum}:{name}', d2
        return best

    def _findRecAtomIndex(self, recAtoms):
        """Index of the covalent receptor atom among `recAtoms`.

        Accepts both forms the form allows: chain:resnum:atom_name is matched by
        name, and x,y,z by taking the nearest atom to that point.
        """
        spec = self.getCovalentRecAtom()
        if spec.count(':') == 2:
            chain, resNum, atomName = [s.strip() for s in spec.split(':')]
            for idx, line in enumerate(recAtoms):
                name, lineChain, lineRes = self._pdbqtAtomKey(line)
                if name == atomName and lineRes == resNum and (lineChain == chain or not chain):
                    return idx
            return None

        if spec.count(',') == 2:
            try:
                target = tuple(float(v) for v in spec.split(','))
            except ValueError:
                return None
            best, bestD2 = None, 1.0
            for idx, line in enumerate(recAtoms):
                xyz = self._pdbqtCoords(line)
                if xyz is None:
                    continue
                d2 = sum((a - b) ** 2 for a, b in zip(target, xyz))
                if d2 < bestD2:
                    best, bestD2 = idx, d2
            return best
        return None

    @classmethod
    def _pdbAtomLine(cls, pdbqtLine, serial):
        """Rewrite a PDBQT atom line as PDB with a new serial.

        PDBQT is PDB up to column 66; the AutoDock type that follows is dropped
        and the element column filled in from it, since that is what viewers
        read to colour and size the atom.
        """
        body = pdbqtLine[:66].ljust(66)
        element = cls._pdbqtElement(pdbqtLine) or ''
        return f'{body[:6]}{serial:>5}{body[11:66]}          {element:>2}\n'

    @staticmethod
    def _readSdfMol(sdfFile):
        """Return (atoms, bonds) from the molblock of an SDF pose.

        atoms is [(x, y, z, element)]; bonds is [(i, j)] with 1-based indices
        into atoms. Bond order is dropped: PDB encodes it by repeating the
        partner serial, which buys nothing here (PyMOL only draws orders with
        valence display on) and cannot express an aromatic bond anyway.
        """
        with open(sdfFile) as fh:
            lines = fh.read().splitlines()
        if len(lines) < 4:
            return [], []
        try:
            nAtoms, nBonds = int(lines[3][:3]), int(lines[3][3:6])
        except ValueError:
            return [], []

        atoms = []
        for line in lines[4:4 + nAtoms]:
            # V2000 is fixed-width (%10.4f x3 then the symbol), which is what
            # gnina writes; the split() fallback covers writers that pad
            # differently rather than silently dropping the atom.
            try:
                x, y, z = float(line[0:10]), float(line[10:20]), float(line[20:30])
                element = line[31:34].strip()
            except ValueError:
                fields = line.split()
                if len(fields) < 4:
                    continue
                try:
                    x, y, z = (float(v) for v in fields[:3])
                except ValueError:
                    continue
                element = fields[3]
            atoms.append((x, y, z, element))

        bonds = []
        for line in lines[4 + nAtoms:4 + nAtoms + nBonds]:
            try:
                first, second = int(line[0:3]), int(line[3:6])
            except ValueError:
                fields = line.split()
                if len(fields) < 2:
                    continue
                try:
                    first, second = int(fields[0]), int(fields[1])
                except ValueError:
                    continue
            if 1 <= first <= len(atoms) and 1 <= second <= len(atoms):
                bonds.append((first, second))
        return atoms, bonds

    @staticmethod
    def _conectLines(neighbours):
        """CONECT records for {serial: [partner serials]}, 4 partners per line."""
        lines = []
        for serial in sorted(neighbours):
            partners = neighbours[serial]
            for start in range(0, len(partners), 4):
                chunk = ''.join(f'{p:>5}' for p in partners[start:start + 4])
                lines.append(f'CONECT{serial:>5}{chunk}\n')
        return lines

    @classmethod
    def _isLigandAtom(cls, line, ligAtoms):
        """True when this fragment atom belongs to the pose rather than the receptor"""
        xyz = cls._pdbqtCoords(line)
        if xyz is None or not ligAtoms:
            return False

        nearest = min(math.dist(xyz, lig[:3]) for lig in ligAtoms)
        # The flex output also carries the ligand's polar hydrogens, which the
        # united-atom pose SDF has none of, so they match no position: a
        # hydrogen within bonding distance of the pose is the pose's.
        return nearest < 0.1 or (nearest < 1.3 and cls._pdbqtElement(line) == 'H')

    @staticmethod
    def _pdbqtElement(line):
        """Element of a PDBQT atom, from its AutoDock type column.

        Types are element-derived ('C', 'OA', 'NA', 'HD', 'S'...) except 'A',
        which is an aromatic carbon.
        """
        adType = line[77:].strip()
        if not adType:
            return None
        return 'C' if adType[0] == 'A' else adType[0]

    @staticmethod
    def _pdbqtCoords(line):
        try:
            return float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            return None

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

        recAtom = (self.covalentRecAtom.get() or '').strip()
        if not recAtom:
            errors.append('A receptor atom is required '
                          '(chain:resnum:atom_name, e.g. A:13:SG, or x,y,z).')
        elif '"' in recAtom or "'" in recAtom or ' ' in recAtom:
            errors.append(f'The receptor atom "{recAtom}" contains quotes or spaces. Give it as '
                          f'chain:resnum:atom_name (e.g. A:13:SG) or x,y,z, with nothing else.')
        elif not (recAtom.count(':') == 2 or recAtom.count(',') == 2):
            errors.append(f'Could not read the receptor atom "{recAtom}": use '
                          f'chain:resnum:atom_name (e.g. A:13:SG) or x,y,z coordinates.')

        smarts = (self.covalentLigPattern.get() or '').strip()
        if not smarts:
            errors.append('A SMARTS pattern for the ligand warhead is required '
                          '(e.g. [CH2]=[CH]C(=O) for an acrylamide). The first atom of the '
                          'match is the one bonded to the receptor.')
        else:
            # Cheap sanity checks: a real SMARTS carries no quotes or whitespace
            # and has balanced brackets. Both have cost a failed run before.
            if '"' in smarts or "'" in smarts or ' ' in smarts:
                errors.append(f'The SMARTS "{smarts}" contains quotes or spaces. Give the pattern '
                              f'alone, e.g. [CH2]=[CH]C(=O).')
            if smarts.count('[') != smarts.count(']') or smarts.count('(') != smarts.count(')'):
                errors.append(f'The SMARTS "{smarts}" has unbalanced brackets '
                              f'({smarts.count("[")} "[" vs {smarts.count("]")} "]", '
                              f'{smarts.count("(")} "(" vs {smarts.count(")")} ")").')

        ligPos = (self.covalentLigPosition.get() or '').strip()
        if ligPos and ligPos.count(',') != 2:
            errors.append(f'Could not read the initial warhead position "{ligPos}": '
                          f'use x,y,z coordinates.')
        if self.covalentFixLigPosition.get() and not ligPos:
            errors.append('Fixing the warhead position requires an initial warhead position.')
        return errors

    def _warnings(self):
        warnings = []
        if not self.covalentOptimizeLig.get():
            warnings.append('Covalent docking without UFF optimisation of the complex: the bond is '
                            'formed geometrically, so the junction stays strained and poses usually '
                            'score much worse (positive affinities are a symptom).')
        return warnings

    # ------------------------------------------------------------------ #
    #  Summary / methods                                                   #
    # ------------------------------------------------------------------ #
    def _summary(self):
        summary = []
        if self.fromReceptor.get() == FROM_PROTEIN and self.inputAtomStruct.get():
            summary.append(f'Receptor: {os.path.basename(self.inputAtomStruct.get().getFileName())}')
        if self.fromReceptor.get() == FROM_POCKET and self.inputStructROIs.get():
            summary.append(f'Pockets: {self.inputStructROIs.get().getSize()} ROI(s)')
        if self.inputSmallMolecules.get():
            summary.append(f'Ligands: {self.inputSmallMolecules.get().getSize()} molecule(s)')
        summary.append(f'Covalent bond: {self.covalentRecAtom.get()} '
                       f'<- {self.covalentLigPattern.get()}')

        kept, dropped = self.getWarheadNames()
        if kept or dropped:
            summary.append(f'Warhead: {len(kept)} of {len(kept) + len(dropped)} molecule(s) match '
                           f'the SMARTS; {len(dropped)} not docked')

        # gnina's own verdict, so a disagreement with the filter shows up here
        # instead of passing unnoticed.
        skipped = self.getGninaSkippedNames()
        if skipped:
            summary.append(f'gnina skipped {len(skipped)} molecule(s) as not matching the SMARTS')
        summary.append('CNN scoring: none (not calibrated for covalent docking)')
        summary.append(f'Exhaustiveness: {self.exhaustiveness.get()} | Modes: {self.numPoses.get()}')
        if self.hasAttribute('outputSmallMolecules'):
            summary.append(f'Output poses: {self.outputSmallMolecules.getSize()}')
        return summary

    def _methods(self):
        return [
            'Covalent molecular docking was performed with GNINA [McNutt2021, McNutt2025].',
            f'The ligand warhead matching "{self.covalentLigPattern.get()}" was bonded to receptor '
            f'atom {self.covalentRecAtom.get()}, with exhaustiveness {self.exhaustiveness.get()} '
            f'and up to {self.numPoses.get()} binding modes per ligand. CNN scoring was disabled, '
            f'as it is not calibrated for covalent complexes, and poses were ranked by the '
            f'"{SCORING_CHOICES[self.scoring.get()]}" empirical scoring function.',
        ]
