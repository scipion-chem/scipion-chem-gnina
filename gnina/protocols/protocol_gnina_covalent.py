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
    EnumParam, IntParam, FloatParam, BooleanParam, StringParam,
    LEVEL_ADVANCED, STEPS_PARALLEL,
)
from pyworkflow.utils.path import makePath
import pyworkflow.object as pwobj

from pwchem.objects import SetOfSmallMolecules
from pwchem.utils import getBaseName, parseAtomStruct, runOpenBabel

from ..constants import *
from .protocol_gnina import ProtGninaDocking, FROM_PROTEIN, CIFext

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
    ligand: gnina forms the bond itself. A ligand whose warhead is already
    saturated and has nothing left to react and is skipped.

    The bond don't exists the ligand SDF but the bonded Protein-Ligand structure
    is reachable in the column covalentPoseFile of the SetOfSmallMolecules output object.

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
                      help='Relax the ligand with UFF once bonded (--covalent_optimize_lig).\n\n'
                           'Recommended: the bond is formed geometrically, so without this step the '
                           'junction stays strained and poses score badly (positive affinities).')
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
                      help='Optional *x,y,z* starting placement for the bonding ligand atom. '
                           "If left empty, gnina places the ligand with Open Babel's GetNewBondVector.")
        form.addParam('covalentFixLigPosition', BooleanParam, label='Fix warhead at that position: ',
                      condition='covalentLigPosition != ""', default=False,
                      expertLevel=LEVEL_ADVANCED,
                      help='Keep the bonding ligand atom pinned at the position above for the whole '
                           'run instead of using it only as the initial structure.\n\n'
                           'Use it when the attachment geometry is known (crystallography, QM) and '
                           'must be respected; otherwise leave it empty.')

        # ---- Search & Scoring ------------------------------------------ #
        form.addSection(label='Search & Scoring')
        form.addParam('exhaustiveness', IntParam, default=8, label='Exhaustiveness: ',
                      help='Exhaustiveness of the Monte-Carlo search (roughly proportional to time).')
        form.addParam('numPoses', IntParam, default=9, label='Number of binding modes: ',
                      help='Maximum number of docking poses to generate per ligand.')
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
        """(docked, not docked) molecule names."""
        kept = []
        for subsetId in range(max(self.numberOfThreads.get() - 1, 1)):
            kept += self._sdfTitles(self._getSubsetLigandFile(subsetId))

        keptNames = set(kept)
        dropped = [getBaseName(mol.getFileName()) for mol in self.inputSmallMolecules.get()
                   if getBaseName(mol.getFileName()) not in keptNames]
        return kept, dropped

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

    ########  Output  ########
    def createOutputStep(self):
        """Collect the docked poses, adding the covalent complex column."""
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
                f'{self.covalentLigPattern.get()}, so there was nothing to dock.')

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

                newMol = self._makePoseMol(
                    srcMol, poseData, gridId,
                    poseRecFiles[poseIdx] if poseRecFiles else recFile)

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

        if dropped:
            with open(self._getNotDockedFile(), 'w') as fOut:
                fOut.write('\n'.join(dropped) + '\n')
            print(f'{len(dropped)} molecule(s) were not docked, having no warhead matching '
                  f'{self.covalentLigPattern.get()}; named in '
                  f'{os.path.basename(self._getNotDockedFile())}')

    def _getNotDockedFile(self):
        return self._getExtraPath('notDocked.txt')

    # ------------------------------------------------------------------ #
    #  UTILS                                                             #
    # ------------------------------------------------------------------ #
    def _buildArgs(self, recFile, ligFile, outFile, logFile, pocket=None):
        """Assemble the gnina command line for a covalent run."""
        args = f'-r "{recFile}" -l "{ligFile}" -o "{outFile}" --log "{logFile}"'

        args += self._buildSearchSpaceArgs(recFile, pocket)

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

        args += self._buildDeviceArgs()
        return args


    @staticmethod
    def _poseMolName(molName):
        """gnina names a covalent complex '<receptor>_<ligand>', and the
        receptor part is empty. Unstripped, no pose matches its input ligand
        and the output set comes out empty."""
        return molName.lstrip('_')

    # ------------------------------------------------------------------ #
    #  Per-pose receptors                                                  #
    # ------------------------------------------------------------------ #
    def _buildFlexReceptors(self, sdfFile, poses, rigidRecFile, recDir):
        """Build one receptor file per pose from the covalent --out_flex output.

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

        groups = self._readModelGroups(flexFile)

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

            lastSerial = max((self._pdbqtSerial(l) for l in rigidLines), default=0)
            appended = []
            for line in recSide:
                xyz = self._pdbqtCoords(line)
                if xyz is None or self._nearestAtom(xyz, rigidLines, 0.05)[1] is not None:
                    continue
                if self._pdbqtElement(line) != 'H':
                    print(f'Warning: {line[12:16].strip()} of '
                          f'{getBaseName(poseData["poseFile"])} is not at its rigid-receptor '
                          f'position; gnina moved the residue and the per-pose receptor will '
                          f'not show it.')

                lastSerial += 1
                line = f'{line[:6]}{lastSerial:>5}{line[11:]}'
                # gnina labels them 'UNK 1' with no chain, which would make a
                # residue of their own and break the chain. Take the identity of
                # the nearest receptor atom: a polar hydrogen is never far from
                # the atom it protonates.
                host = self._nearestAtom(xyz, rigidLines, 2.0)[1]
                if host is not None:
                    line = f'{line[:17]}{host[17:27]}{line[27:]}'
                appended.append(line)

            recAtomLines = list(rigidLines) + appended
            outRec = os.path.join(recDir, f'{getBaseName(poseData["poseFile"])}_rec.pdbqt')
            with open(outRec, 'w') as fOut:
                fOut.writelines(recAtomLines)
            recFiles.append(os.path.abspath(outRec))

            self._writeCovalentComplex(poseData, recAtomLines)
        return recFiles

    @classmethod
    def _nearestAtom(cls, xyz, lines, maxDist):
        """(index, line) of the atom of `lines` nearest to `xyz`, (None, None)
        when the nearest one is further than maxDist"""
        best, bestD2 = (None, None), maxDist ** 2
        for idx, line in enumerate(lines):
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            lineXyz = cls._pdbqtCoords(line)
            if lineXyz is None:
                continue
            d2 = sum((a - b) ** 2 for a, b in zip(xyz, lineXyz))
            if d2 < bestD2:
                best, bestD2 = (idx, line), d2
        return best

    @staticmethod
    def _pdbqtSerial(line):
        """Atom serial of a PDBQT line, 0 for anything else"""
        try:
            return int(line[6:11])
        except ValueError:
            return 0

    # ------------------------------------------------------------------ #
    #  Covalent complex files                                              #
    # ------------------------------------------------------------------ #
    def _writeCovalentComplex(self, poseData, recLines):
        """Write receptor + pose as one PDB whose CONECT record is the bond.

        Neither input file carries it: the pose SDF has no link to the receptor,
        and the receptor has no ligand. Viewers honour CONECT, so the bond needs
        no viewer code. Returns the path written, or None.
        """
        ligAtoms, ligBonds = self._readSdfMol(poseData['poseFile'])
        recAtoms = [ln for ln in recLines if ln.startswith(('ATOM', 'HETATM'))]
        recIdx = self._findRecAtomIndex(recAtoms)
        recXyz = self._pdbqtCoords(recAtoms[recIdx]) if recIdx is not None else None
        if not ligAtoms:
            print(f'Warning: no atoms read from {getBaseName(poseData["poseFile"])}; '
                  f'no complex written.')
            return None
        if recXyz is None:
            print(f'Warning: receptor atom {self.getCovalentRecAtom()} not found in the '
                  f'per-pose receptor; no complex written.')
            return None

        makePath(self._getPath(GNINA_COMPLEX_DIR))
        outFile = self._complexFilePath(poseData['poseFile'])

        # Serials renumbered from 1: a PDBQT restarts them after every TER, so
        # the original ones cannot address an atom uniquely in a CONECT.
        lines, serial, recSerial = [], 0, None
        for idx, line in enumerate(recAtoms):
            serial += 1
            if idx == recIdx:
                recSerial = serial
            lines.append(self._pdbAtomLine(line, serial))
        lines.append('TER\n')

        # The ligand as its own HETATM residue, so it collides with no receptor
        # chain. The nearest of its atoms to the receptor atom is the bonded one.
        ligSerial, bestD2, elemCounts, ligSerials = None, COVALENT_BOND_CUTOFF ** 2, {}, []
        for x, y, z, element in ligAtoms:
            serial += 1
            ligSerials.append(serial)
            elemCounts[element] = elemCounts.get(element, 0) + 1
            # Two-letter elements start in column 13, one-letter ones in 14.
            name = f'{element}{elemCounts[element]}' if len(element) > 1 \
                else f' {element}{elemCounts[element]}'
            lines.append(f'HETATM{serial:>5} {name[:4]:<4} LIG Z   1    '
                         f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          '
                         f'{element:>2}\n')
            d2 = sum((a - b) ** 2 for a, b in zip((x, y, z), recXyz))
            if d2 < bestD2:
                ligSerial, bestD2 = serial, d2
        lines.append('TER\n')

        # The ligand's own bonds go in too: a reader that finds any CONECT for a
        # HETATM residue takes them as its complete connectivity, so a file with
        # only the covalent link draws loose atoms. Receptor bonds are left out,
        # as in a deposited entry: they come from the reader's own templates.
        neighbours = {}
        for first, second in ligBonds:
            neighbours.setdefault(ligSerials[first - 1], []).append(ligSerials[second - 1])
            neighbours.setdefault(ligSerials[second - 1], []).append(ligSerials[first - 1])

        if ligSerial is None:
            # Report the geometry rather than invent a bond; the file is still
            # written so it can be looked at.
            print(f'Warning: no ligand atom within {COVALENT_BOND_CUTOFF} A of '
                  f'{self.getCovalentRecAtom()} for {getBaseName(poseData["poseFile"])}; '
                  f'the complex has no bond record.')
        else:
            # Both directions: some readers only follow the first field.
            neighbours.setdefault(recSerial, []).append(ligSerial)
            neighbours.setdefault(ligSerial, []).append(recSerial)

        lines += self._conectLines(neighbours) + ['END\n']
        with open(outFile, 'w') as fOut:
            fOut.writelines(lines)
        return os.path.abspath(outFile)

    def _complexFilePath(self, poseFile):
        """Path of a pose's covalent complex, written or not"""
        return self._getPath(GNINA_COMPLEX_DIR, f'{getBaseName(poseFile)}_cov.pdb')

    def getCovalentRecAtom(self):
        """The receptor atom address in the numbering gnina actually sees.

        The form value uses the input structure's numbering, but Open Babel
        renumbers residues from 1 when it writes the PDBQT that is docked
        against: 6OIM starts at Gly 0, so its Cys 12 becomes Cys 13 and gnina
        stops with 'Could not find receptor atom'. Translated by coordinates,
        which is exact - the conversion renumbers but moves no atom.
        """
        if getattr(self, '_recAtomCache', None) is not None:
            return self._recAtomCache

        # The x,y,z form needs no translation.
        recAtom = resolved = (self.covalentRecAtom.get() or '').strip()
        if recAtom.count(':') == 2:
            xyz = self._inputAtomCoords(recAtom)
            match = self._pdbqtAtomAt(self.getReceptorPDBQT(), xyz) if xyz else None
            if match is None:
                print(f'Warning: {recAtom} could not be located in the converted receptor; '
                      f'passing it to gnina unchanged.')
            elif match != recAtom:
                print(f'Receptor atom {recAtom} is {match} in the converted receptor '
                      f'(Open Babel renumbers residues from 1); gnina is given {match}.')
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

        line = cls._nearestAtom(xyz, list(open(pdbqtFile)), tol)[1]
        if line is None:
            return None

        name, chain, resNum = cls._pdbqtAtomKey(line)
        return f'{chain}:{resNum}:{name}'

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
            return self._nearestAtom(target, recAtoms, 1.0)[0]
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
            try:
                atoms.append((float(line[0:10]), float(line[10:20]), float(line[20:30]),
                              line[31:34].strip()))
            except ValueError:
                # A partial atom list would desynchronise the bond indices, so
                # give up on the molecule and let the caller report it.
                return [], []

        bonds = []
        for line in lines[4 + nAtoms:4 + nAtoms + nBonds]:
            try:
                first, second = int(line[0:3]), int(line[3:6])
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
        errors = self._validateCommon()

        recAtom = (self.covalentRecAtom.get() or '').strip()
        if not recAtom:
            errors.append('A receptor atom is required '
                          '(chain:resnum:atom_name, e.g. A:13:SG, or x,y,z coordinates).')
        elif not (recAtom.count(':') == 2 or recAtom.count(',') == 2):
            errors.append(f'Could not read the receptor atom "{recAtom}": use '
                          f'chain:resnum:atom_name (e.g. A:13:SG) or x,y,z coordinates.')

        smarts = (self.covalentLigPattern.get() or '').strip()
        if not smarts:
            errors.append('A SMARTS pattern for the ligand warhead is required '
                          '(e.g. [CH2]=[CH]C(=O) for an acrylamide). The first atom of the '
                          'match is the one bonded to the receptor.')

        ligPos = (self.covalentLigPosition.get() or '').strip()
        if ligPos and ligPos.count(',') != 2:
            errors.append(f'Could not read the initial warhead position "{ligPos}": '
                          f'use x,y,z coordinates.')
        if self.covalentFixLigPosition.get() and not ligPos:
            errors.append('Fixing the warhead position requires an initial warhead position.')
        return errors

    # ------------------------------------------------------------------ #
    #  Summary / methods                                                   #
    # ------------------------------------------------------------------ #
    def _summary(self):
        summary = self._summaryInputs()
        summary.append(f'Covalent bond: {self.covalentRecAtom.get()} '
                       f'<- {self.covalentLigPattern.get()}')

        kept, dropped = self.getWarheadNames()
        if kept or dropped:
            summary.append(f'Warhead: {len(kept)} of {len(kept) + len(dropped)} molecule(s) match '
                           f'the SMARTS; {len(dropped)} not docked')
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
