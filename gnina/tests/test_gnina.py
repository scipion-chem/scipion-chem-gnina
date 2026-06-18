# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors: Joaquin Algorta (joaquin.algorta@cnb.csic.es)
# *
# * Biocomputing Unit, CNB-CSIC
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

# Scipion core imports
from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pwem.protocols import ProtImportPdb

# Scipion chem imports
from pwchem.protocols import (ProtChemImportSmallMolecules, ProtChemPrepareReceptor,
                              ProtDefineStructROIs)
from pwchem.utils import assertHandle

# Plugin imports
from ..protocols import ProtGninaDocking, ProtGninaScore


class TestGninaBase(BaseTest):
    """Shared setup for the GNINA tests: import the receptor + ligands and run a
    target preparation, so docking/rescoring always start from a prepared
    receptor (cleaned, waters removed, HEM kept for the pocket definition)."""

    @classmethod
    def setUpClass(cls):
        cls.ds = DataSet.getDataSet('model_building_tutorial')
        cls.dsLig = DataSet.getDataSet('smallMolecules')

        setupTestProject(cls)
        cls._runImportPDB()
        cls._runImportSmallMols()
        cls._waitOutput(cls.protImportPDB, 'outputPdb', sleepTime=5)
        cls._waitOutput(cls.protImportSmallMols, 'outputSmallMolecules', sleepTime=5)
        cls._runPrepareReceptor()
        cls._waitOutput(cls.protPrepRec, 'outputStructure', sleepTime=5)

    # ------------------------------------------------------------------ #
    #  Common inputs                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def _runImportPDB(cls):
        cls.protImportPDB = cls.newProtocol(
            ProtImportPdb, inputPdbData=1,
            pdbFile=cls.ds.getFile('PDBx_mmCIF/5ni1.pdb'))
        cls.proj.launchProtocol(cls.protImportPDB, wait=False)

    @classmethod
    def _runImportSmallMols(cls):
        # A small set of distinct ligands, shared by all tests, so several
        # molecules exercise the multi-ligand handling.
        cls.protImportSmallMols = cls.newProtocol(
            ProtChemImportSmallMolecules,
            filesPath=cls.dsLig.getFile('mol2'),
            filesPattern='*.mol2')
        cls.proj.launchProtocol(cls.protImportSmallMols, wait=False)

    @classmethod
    def _runPrepareReceptor(cls):
        # Target preparation: remove waters and HETATM but keep the heme (HEM)
        # so it can still be used to define the binding-site ROI.
        cls.protPrepRec = cls.newProtocol(
            ProtChemPrepareReceptor,
            inputAtomStruct=cls.protImportPDB.outputPdb,
            waters=True, HETATM=True, het2keep='HEM')
        cls.proj.launchProtocol(cls.protPrepRec, wait=False)

    @classmethod
    def _runDefineROIs(cls):
        # Define a structural ROI around the heme (HEM) site of 5ni1 and REMOVE
        # the HEM HETATM from the structure used for docking (remMol / remove=True),
        # so ligands dock into the freed heme pocket. The "N) " prefix is the
        # format the ROI-definition wizard stores in 'inROIs'.
        cls.protDefROIs = cls.newProtocol(
            ProtDefineStructROIs,
            inputAtomStruct=cls.protPrepRec.outputStructure,
            origin=2,  # Ligand
            extLig=False, molName='HEM', remMol=True,
            surfaceCoords=True,
            inROIs='1) Ligand: {"molName": "HEM", "remove": "True"}')
        cls.proj.launchProtocol(cls.protDefROIs, wait=True)
        return cls.protDefROIs

    # ------------------------------------------------------------------ #
    #  GNINA runs (helpers reused by the test classes)                     #
    # ------------------------------------------------------------------ #
    def _runGninaWholeProtein(self):
        protGnina = self.newProtocol(
            ProtGninaDocking,
            fromReceptor=0,
            inputAtomStruct=self.protPrepRec.outputStructure,
            inputSmallMolecules=self.protImportSmallMols.outputSmallMolecules,
            exhaustiveness=4, numPoses=3,
            cnnScoring=0,  # rescore
            numberOfThreads=2)
        self.proj.launchProtocol(protGnina, wait=False)
        return protGnina

    def _runGninaPockets(self, roisProt):
        protGnina = self.newProtocol(
            ProtGninaDocking,
            fromReceptor=1,
            inputStructROIs=roisProt.outputStructROIs,
            inputSmallMolecules=self.protImportSmallMols.outputSmallMolecules,
            pocketRadiusN=1.5, exhaustiveness=4, numPoses=3,
            cnnScoring=0,  # rescore
            numberOfThreads=2)
        self.proj.launchProtocol(protGnina, wait=False)
        return protGnina

    def _runGninaRescore(self, dockProt, scoreMode=0):
        # Rescore the poses produced by a previous GNINA docking run.
        protScore = self.newProtocol(
            ProtGninaScore,
            inputSmallMolecules=dockProt.outputSmallMolecules,
            scoreMode=scoreMode,  # 0=score only, 1=local, 2=minimize
            cnnScoring=0,         # rescore
            numberOfThreads=2)
        self.proj.launchProtocol(protScore, wait=False)
        return protScore


class TestGninaDocking(TestGninaBase):
    """Docking tests for ProtGninaDocking (whole protein and pockets)."""

    def testWholeProtein(self):
        print('\nDocking with GNINA on the whole protein')
        protGnina = self._runGninaWholeProtein()
        self._waitOutput(protGnina, 'outputSmallMolecules', sleepTime=10)
        assertHandle(self.assertIsNotNone,
                     getattr(protGnina, 'outputSmallMolecules', None),
                     cwd=protGnina.getWorkingDir())
        assertHandle(self.assertGreater,
                     getattr(protGnina, 'outputSmallMolecules').getSize(), 0,
                     cwd=protGnina.getWorkingDir())

    def testPockets(self):
        print('\nDocking with GNINA on a defined structural ROI')
        protROIs = self._runDefineROIs()
        self._waitOutput(protROIs, 'outputStructROIs', sleepTime=5)

        protGnina = self._runGninaPockets(protROIs)
        self._waitOutput(protGnina, 'outputSmallMolecules', sleepTime=10)
        assertHandle(self.assertIsNotNone,
                     getattr(protGnina, 'outputSmallMolecules', None),
                     cwd=protGnina.getWorkingDir())
        assertHandle(self.assertGreater,
                     getattr(protGnina, 'outputSmallMolecules').getSize(), 0,
                     cwd=protGnina.getWorkingDir())


class TestGninaScore(TestGninaBase):
    """Rescoring test for ProtGninaScore (score-only re-evaluation of poses)."""

    def testRescore(self):
        print('\nRescoring docked GNINA poses (score only, all poses in one call per subset)')
        protGnina = self._runGninaWholeProtein()
        self._waitOutput(protGnina, 'outputSmallMolecules', sleepTime=10)
        nDocked = protGnina.outputSmallMolecules.getSize()

        protScore = self._runGninaRescore(protGnina, scoreMode=0)
        self._waitOutput(protScore, 'outputSmallMolecules', sleepTime=10)
        outSet = getattr(protScore, 'outputSmallMolecules', None)
        assertHandle(self.assertIsNotNone, outSet, cwd=protScore.getWorkingDir())
        # Every docked pose must come out rescored (no pose lost).
        assertHandle(self.assertEqual, outSet.getSize(), nDocked,
                     cwd=protScore.getWorkingDir())
