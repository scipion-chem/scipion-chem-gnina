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

from pwem.wizards import EmWizard
from pwchem.utils import runOpenBabel
from pwchem.wizards import SelectChainWizard, SelectChainWizardQT, SelectResidueWizardQT
from gnina.protocols import ProtGninaDocking

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