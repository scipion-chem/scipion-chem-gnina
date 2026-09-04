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

import os, json
from pyworkflow.gui.tree import ListTreeProviderString
from pyworkflow.gui import dialog
from pyworkflow.object import String

from pwem.wizards import EmWizard, VariableWizard
from pwem.convert import AtomicStructHandler
from pwchem.utils import parseAtomStruct
from pwchem.wizards import (SelectChainWizard, SelectChainWizardQT, SelectResidueWizardQT,
                            SelectAtomWizardQT)
from gnina.protocols import ProtGninaDocking, ProtGninaCovalentDocking
from gnina.constants import COVALENT_WARHEAD_EXAMPLES

SelectChainWizardQT().addTarget(protocol=ProtGninaDocking,
                                targets=['flexChain'],
                                inputs=[{'fromReceptor': ['inputAtomStruct', 'inputStructROIs']}],
                                outputs=['flexChain'])


class SelectSpecificResiduesWizardQT(SelectResidueWizardQT):
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        inputParams, outputParam = self.getInputOutput(form)
        protocol = form.protocol
        inputObj = getattr(protocol, inputParams[0]).get()

        # 1. Extract the chain ID from the protocol parameters
        if len(inputParams) < 2:
            # Fallback for raw sequence objects that don't have an associated chain
            chainStr = None
            chain_id = "A"
        else:
            chainStr = getattr(protocol, inputParams[1]).get()
            try:
                struct = json.loads(chainStr)
                chain_id = struct.get("chain", "A").upper().strip()
            except Exception:
                chain_id = "A"

        # 2. Fetch the residues using your existing underlying logic (including PDBQT conversions)
        finalResiduesList = self.getResidues(form, inputObj, chainStr)

        # 3. Launch the dialog box
        provider = ListTreeProviderString(finalResiduesList)
        dlg = dialog.ListDialog(form.root, "Select Specific Residues", provider,
                                "Hold Ctrl (Windows/Linux) or Cmd (Mac) to select specific, individual residues.")

        # Guard clause in case the user cancels or closes without choosing anything
        if not dlg.values:
            return

        # 4. Iterate through ALL selected items rather than slicing just the first and last
        selected_residues_list = []
        for item in dlg.values:
            residue_data = json.loads(item.get())
            res_number = residue_data['index']

            # Format individual element as chainID:resNumber
            selected_residues_list.append("{}:{}".format(chain_id, res_number))

        # 5. Join the array into a single comma-separated string
        formatted_output_string = ",".join(selected_residues_list)

        # 6. Save the final string back into the protocol output variable
        form.setVar(outputParam[0], formatted_output_string)

SelectSpecificResiduesWizardQT().addTarget(protocol=ProtGninaDocking,
                                  targets=['flexRes'],
                                  inputs=[{'fromReceptor': ['inputAtomStruct', 'inputStructROIs']}, 'flexChain'],
                                  outputs=['flexRes'])


class SelectCovalentRecAtomWizard(SelectAtomWizardQT):
    """Pick the receptor atom of the covalent bond, as 'chain:resnum:atom'.

    pwchem's SelectAtomWizardQT needs a chain and a residue param already
    filled to know where to look; gnina takes the whole address in one string,
    so the three choices are asked here in turn and joined.

    Reading the numbering off the receptor file is the point: it is not always
    the numbering of the literature, and a wrong number fails the run.
    """
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        inputParams, outputParam = self.getInputOutput(form)
        inputObj = getattr(form.protocol, inputParams[0]).get()
        if inputObj is None:
            dialog.showError('Missing receptor', 'Select the receptor first.', form.root)
            return

        fileName = self.getInputFilename(form.protocol, inputObj, AtomicStructHandler())
        structure = parseAtomStruct(fileName)
        if structure is None:
            dialog.showError('Unreadable structure',
                             f'Could not parse {os.path.basename(fileName)}.', form.root)
            return

        chain = self.pickOne(form, self.listChains(structure), 'Receptor chains',
                             'Chain holding the reacting residue (model, chain, residues)')
        if chain is None:
            return

        residues = self.listResidues(structure, int(chain['model']), chain['chain'])
        res = self.pickOne(form, residues, 'Chain residues',
                           'Reacting residue (number, name), numbered as in the receptor file')
        if res is None:
            return

        atoms = self.editionListOfAtoms(structure, int(chain['model']), chain['chain'],
                                        int(res['index']))
        atom = self.pickOne(form, atoms, 'Residue atoms',
                            'Bonding atom: SG for cysteine, OG for serine, NZ for lysine')
        if atom is not None:
            form.setVar(outputParam[0], f'{chain["chain"]}:{res["index"]}:{atom["atom"]}')

    @staticmethod
    def pickOne(form, entries, title, message):
        """One list dialog, returning the parsed choice or None if cancelled"""
        if not entries:
            dialog.showError('Nothing to select', f'{message}\n\nNothing was found.', form.root)
            return None

        provider = ListTreeProviderString([String(entry) for entry in entries])
        dlg = dialog.ListDialog(form.root, title, provider, message)
        return json.loads(dlg.values[0].get()) if dlg.values else None

    @staticmethod
    def listChains(structure):
        """Chains that could hold a nucleophile: not the all-water one, and not
        a nameless one, which would not compose a valid address either"""
        chains = []
        for model in structure:
            for chain in model:
                nRes = sum(1 for res in chain if res.get_resname() not in ('HOH', 'WAT'))
                if nRes and chain.get_id().strip():
                    chains.append('{"model": %s, "chain": "%s", "residues": %s}'
                                  % (model.get_id(), chain.get_id(), nRes))
        return chains

    @staticmethod
    def listResidues(structure, modelId, chainId):
        """Residues of one chain, waters left out. Het residues are kept: a
        modified residue can be the nucleophile"""
        return ['{"index": %s, "residue": "%s"}' % (res.get_id()[1], res.get_resname())
                for model in structure if model.get_id() == modelId
                for chain in model if chain.get_id() == chainId
                for res in chain if res.get_resname() not in ('HOH', 'WAT')]

SelectCovalentRecAtomWizard().addTarget(protocol=ProtGninaCovalentDocking,
                                        targets=['covalentRecAtom'],
                                        inputs=[{'fromReceptor': ['inputAtomStruct',
                                                                  'inputStructROIs']}],
                                        outputs=['covalentRecAtom'])


class SelectCovalentWarheadWizard(VariableWizard):
    """Fill the SMARTS field from the list of known warheads"""
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        _, outputParam = self.getInputOutput(form)
        labels = [String(f'{name}: {smarts}') for name, smarts in COVALENT_WARHEAD_EXAMPLES]
        dlg = dialog.ListDialog(form.root, 'Warhead patterns', ListTreeProviderString(labels),
                                'Reacting group of the ligand. gnina bonds the FIRST atom of the '
                                'match, so a hand-written pattern must start with it.')
        if dlg.values:
            form.setVar(outputParam[0], dlg.values[0].get().split(': ', 1)[1])

# inputs is required by addTarget even when the wizard reads nothing.
SelectCovalentWarheadWizard().addTarget(protocol=ProtGninaCovalentDocking,
                                        targets=['covalentLigPattern'],
                                        inputs=[],
                                        outputs=['covalentLigPattern'])
