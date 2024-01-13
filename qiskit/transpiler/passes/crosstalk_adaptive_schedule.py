# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Crosstalk mitigation through adaptive instruction scheduling.
The scheduling algorithm is described in:
Prakash Murali, David C. Mckay, Margaret Martonosi, Ali Javadi Abhari,
Software Mitigation of Crosstalk on Noisy Intermediate-Scale Quantum Computers,
in International Conference on Architectural Support for Programming Languages
and Operating Systems (ASPLOS), 2020.
Please cite the paper if you use this pass.
The method handles crosstalk noise on two-qubit gates. This includes crosstalk
with simultaneous two-qubit and one-qubit gates. The method ignores
crosstalk between pairs of single qubit gates.
The method assumes that all qubits get measured simultaneously whether or not
they need a measurement. This assumption is based on current device properties
and may need to be revised for future device generations.
"""

import math
import operator
from itertools import chain, combinations
try:
    from z3 import Real, Bool, Sum, Implies, And, Or, Not, Optimize
    Z3_AVAIL = True
except ImportError:
    Z3_AVAIL = False
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import U1Gate, U2Gate, U3Gate, CnotGate
from qiskit.circuit import Measure
from qiskit.extensions.standard.barrier import Barrier

NUM_PREC = 10
TWOQ_XTALK_THRESH = 3
ONEQ_XTALK_THRESH = 2


class CrosstalkAdaptiveSchedule(TransformationPass):
    """
    Crosstalk mitigation through adaptive instruction scheduling.
    """
    def __init__(self, backend_prop, crosstalk_prop, weight_factor=0.5, measured_qubits=None):
        """
        CrosstalkAdaptiveSchedule initializer.
        Args:
            backend_prop (BackendProperties): backend properties object
            crosstalk_prop (dict): crosstalk properties object
                crosstalk_prop[g1][g2] specifies the conditional error rate of
                g1 when g1 and g2 are executed simultaneously.
                g1 should be a two-qubit tuple of the form (x,y) where x and y are physical
                qubit ids. g2 can be either two-qubit tuple (x,y) or single-qubit tuple (x).
                We currently ignore crosstalk between pairs of single-qubit gates.
                Gate pairs which are not specified are assumed to be crosstalk free.
                Example: crosstalk_prop = {(0, 1) : {(2, 3) : 0.2, (2) : 0.15},
                                           (4, 5) : {(2, 3) : 0.1},
                                           (2, 3) : {(0, 1) : 0.05, (4, 5): 0.05}}
                The keys of the crosstalk_prop are tuples for ordered tuples for CX gates
                e.g., (0, 1) corresponding to CX 0, 1 in the hardware.
                Each key has an associated value dict which specifies the conditional error rates
                with nearby gates e.g., (0, 1) : {(2, 3) : 0.2, (2) : 0.15} means that
                CNOT 0, 1 has an error rate of 0.2 when it is executed in parallel with CNOT 2,3
                and an error rate of 0.15 when it is executed in parallel with a single qubit
                gate on qubit 2.
            weight_factor (float): weight of gate error/crosstalk terms in the objective
                weight_factor*fidelities + (1-weight_factor)*decoherence errors.
                Weight can be varied from 0 to 1, with 0 meaning that only decoherence
                errors are optimized and 1 meaning that only crosstalk errors are optimized.
                weight_factor should be tuned per application to get the best results.
            measured_qubits (list): a list of qubits that will be measured in a particular circuit.
                This arg need not be specified for circuits which already include measure gates.
                The arg is useful when a subsequent module such as state_tomography_circuits
                inserts the measure gates. If CrosstalkAdaptiveSchedule is made aware of those
                measurements, it is included in the optimization.
        Raises:
            ImportError: if unable to import z3 solver
        """
        super().__init__()
        if not Z3_AVAIL:
            raise ImportError('z3-solver is required to use CrosstalkAdaptiveSchedule')
        self.backend_prop = backend_prop
        self.crosstalk_prop = crosstalk_prop
        self.weight_factor = weight_factor
        if measured_qubits is None:
            self.input_measured_qubits = []
        else:
            self.input_measured_qubits = measured_qubits
        self.bp_u1_err = {}
        self.bp_u1_dur = {}
        self.bp_u2_err = {}
        self.bp_u2_dur = {}
        self.bp_u3_err = {}
        self.bp_u3_dur = {}
        self.bp_cx_err = {}
        self.bp_cx_dur = {}
        self.bp_t1_time = {}
        self.bp_t2_time = {}
        self.gate_id = {}
        self.gate_start_time = {}
        self.gate_duration = {}
        self.gate_fidelity = {}
        self.overlap_amounts = {}
        self.overlap_indicator = {}
        self.qubit_lifetime = {}
        self.dag_overlap_set = {}
        self.xtalk_overlap_set = {}
        self.opt = Optimize()
        self.measured_qubits = []
        self.measure_start = None
        self.last_gate_on_qubit = None
        self.first_gate_on_qubit = None
        self.fidelity_terms = []
        self.coherence_terms = []
        self.model = None
        self.dag = None
        self.parse_backend_properties()

    def powerset(self, iterable):
        """
        Finds the set of all subsets of the given iterable
        This function is used to generate constraints for the Z3 optimization
        """
        l_s = list(iterable)
        return chain.from_iterable(combinations(l_s, r) for r in range(len(l_s)+1))

    def parse_backend_properties(self):
        """
        This function assumes that gate durations and coherence times
        are in seconds in backend.properties()
        This function converts gate durations and coherence times to
        nanoseconds.
        """
        backend_prop = self.backend_prop
        for qid in range(len(backend_prop.qubits)):
            self.bp_t1_time[qid] = int(backend_prop.t1(qid)*10**9)
            self.bp_t2_time[qid] = int(backend_prop.t2(qid)*10**9)
            self.bp_u1_dur[qid] = int(backend_prop.gate_length('u1', qid))*10**9
            u1_err = backend_prop.gate_error('u1', qid)
            if u1_err == 1.0:
                u1_err = 0.9999
            self.bp_u1_err = round(u1_err, NUM_PREC)
            self.bp_u2_dur[qid] = int(backend_prop.gate_length('u2', qid))*10**9
            u2_err = backend_prop.gate_error('u2', qid)
            if u2_err == 1.0:
                u2_err = 0.9999
            self.bp_u2_err = round(u2_err, NUM_PREC)
            self.bp_u3_dur[qid] = int(backend_prop.gate_length('u3', qid))*10**9
            u3_err = backend_prop.gate_error('u3', qid)
            if u3_err == 1.0:
                u3_err = 0.9999
            self.bp_u3_err = round(u3_err, NUM_PREC)
        for ginfo in backend_prop.gates:
            if ginfo.gate == 'cx':
                q_0 = ginfo.qubits[0]
                q_1 = ginfo.qubits[1]
                cx_tup = (min(q_0, q_1), max(q_0, q_1))
                self.bp_cx_dur[cx_tup] = int(backend_prop.gate_length('cx', cx_tup))*10**9
                cx_err = backend_prop.gate_error('cx', cx_tup)
                if cx_err == 1.0:
                    cx_err = 0.9999
                self.bp_cx_err[cx_tup] = round(cx_err, NUM_PREC)

    def cx_tuple(self, gate):
        """
        Representation for two-qubit gate
        Note: current implementation assumes that the CX error rates and
        crosstalk behavior are independent of gate direction
        """
        physical_q_0 = gate.qargs[0].index
        physical_q_1 = gate.qargs[1].index
        r_0 = min(physical_q_0, physical_q_1)
        r_1 = max(physical_q_0, physical_q_1)
        return (r_0, r_1)