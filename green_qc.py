import numpy as np
from qat.lang.AQASM import H, I, X, Y, RX, RY, RZ, CNOT, QRoutine, Program
from qat.core import Observable, Term
from qat.lang import PH
import copy
from qat.plugins import ScipyMinimizePlugin
from qat.qpus import get_default_qpu
 
 
def real_time_G_gl(tau, nqubits, nparams, Greater = True, epsilon=0, U=None, vqe_result=None):

    all_operators = [[X,X], [X,Y], [Y,X], [Y,Y]]
    coefficient_ops_greater = np.array([1, -1j, 1j, 1], dtype=np.complex64)
    coefficient_ops_lesser = np.array([1, 1j, -1j, 1], dtype=np.complex64)

    G_terms = np.zeros(4, dtype=np.complex64)

    for i_term in range(4):
        G_pp = []
        for ip,phase in enumerate([0, -np.pi/2]): # loop to measure the real and imag part of each term
            Groutine = QRoutine()
            qw = Groutine.new_wires(nqubits)
            ancilla_qubit = qw[0]
            target_qubits = [qw[i] for i in range(1, len(qw))]

            # QRoutine for the unitary operator U = P(tau_1)P(tau_2)
            operators = all_operators[i_term]
            if Greater:
                correlator = apply_PtauP(epsilon, U, tau, operators) # returns a QRoutine() for the unitary operator X(tau)X(0) 
            else:
                correlator = apply_PPtau(epsilon, U, tau, operators) # returns a QRoutine() for the unitary operator X(0)X(tau)
            controlled_unitary = correlator.ctrl()

            # Make the average measuring circuit within QRoutine
            Groutine.apply(H, ancilla_qubit)
            Groutine.apply(controlled_unitary, [ancilla_qubit]+target_qubits)
            Groutine.apply(PH(phase), ancilla_qubit)
            Groutine.apply(H, ancilla_qubit)

            # Program class for realizing the circuit using all QRoutine()'s
            new_prog = Program()
            qreg = new_prog.qalloc(nqubits)
            target_qreg = [qreg[i] for i in range(1, len(qreg))]            
            new_prog.apply(simple_circuit([new_prog.new_var(float, "\\theta_%s" % i) for i in range(nparams)]), target_qreg)
            new_prog.apply(Groutine, qreg)

            # Circuit and observables
            new_circuit = new_prog.to_circ()

            # Loading the VQE solution
            optimized_params = eval(vqe_result.meta_data["parameter_map"])
            new_circuit = new_circuit(**optimized_params)
            #new_circuit.display()
            
            z_observable = Observable(nqubits, pauli_terms=[Term(1., "Z", [0])])
            job = new_circuit.to_job(job_type="OBS", observable=z_observable)
            qpu = get_default_qpu()
            result = qpu.submit(job) # Average of Z operator over ancilla
            G_pp.append(result.value) 

        G_terms[i_term] = G_pp[0] + 1j*G_pp[1] 

    if Greater:
        coefficient_ops = coefficient_ops_greater
        G_terms = coefficient_ops*G_terms
        G_gl = -0.25*1j*np.sum(G_terms)
    else:
        coefficient_ops = coefficient_ops_lesser
        G_terms = coefficient_ops*G_terms
        G_gl = 0.25*1j*np.sum(G_terms)
    
    return G_gl


def apply_PtauP(eps, U, tau, operators):
    """
    Apply a unitary operator (e.g., X(t)X(0) ) to the qubits.
    A quantum circuit is constructed from left to right.
    In a quantum circuit the first operator is on the right and the second operator goes on the left
    """
    uroutine = QRoutine()
    q = uroutine.new_wires(2)
    uroutine.apply(operators[1], q[0]) #.controlled_by(qreg[0]) # acting on site 1, controlled by ancilla 0
    uroutine.apply(add_time_evolution_1site(eps, U, tau, forward=False), [q[0], q[1]]) #.controlled_by(qreg[0])
    uroutine.apply(operators[0], q[0]) #.controlled_by(qreg[0]) # acting on site 1, controlled by ancilla 0
    uroutine.apply(add_time_evolution_1site(eps, U, tau, forward=True),  [q[0], q[1]]) #.controlled_by(qreg[0])
    return uroutine


def apply_PPtau(eps, U, tau, operators):
    """
    Apply a unitary operator (e.g., X(0)X(t) ) to the qubits.
    A quantum circuit is constructed from left to right.
    In a quantum circuit the first operator is on the right and the second operator goes on the left
    """
    uroutine = QRoutine()
    q = uroutine.new_wires(2)
    uroutine.apply(add_time_evolution_1site(eps, U, tau, forward=False), [q[0], q[1]]) #.controlled_by(qreg[0])
    uroutine.apply(operators[1], q[0]) #.controlled_by(qreg[0]) # acting on site 1, controlled by ancilla 0
    uroutine.apply(add_time_evolution_1site(eps, U, tau, forward=True),  [q[0], q[1]]) #.controlled_by(qreg[0])
    uroutine.apply(operators[0], q[0]) #.controlled_by(qreg[0]) # acting on site 1, controlled by ancilla 0
    return uroutine

# Change this function as well - The Hamiltonian.
def add_time_evolution_1site(eps, U, tau, forward=True, num_steps=50):
    """
    Apply a Trotterized time evolution step for the Hamiltonian - 1 site Hubbard model:
    H = (U/4)*(ZZ|[0,1]) - (eps/2+U/4)*(Z|[0] + Z|[1]) 

    Parameters:
    - prog: The QLM program to which gates will be added.
    - q1, q2: Qubits on which the Hamiltonian acts.
    - mu1, mu2: Coefficients for the Z terms on qubits q1 and q2.
    - U: Coefficient for the Z_1 * Z_2 interaction term.
    - delta_tau: The Trotterized time step.
    - forward: Boolean indicating forward or reverse time evolution.
    """
    # Determine the time direction
    factor = 1.0 if forward else -1.0
    effective_delta_tau = factor * tau / num_steps # dt
    troutine = QRoutine()
    q = troutine.new_wires(2) # 2 qubits required for the one-site Hubbard model


    # Direct application of e^{-i H_1 dt} e^{-i H_2 t} - when the Hamiltonian is diagonal in the computational basis

    tau_full = factor * tau
    # 1st term
    troutine.apply(CNOT, q[0], q[1])
    troutine.apply(RZ(-(U/2)*tau_full), q[1]) # NOT YET TROTTERIZED
    troutine.apply(CNOT, q[0], q[1])
    # 2nd term
    troutine.apply(RZ((eps+U/2)*tau_full), q[0]) # NOT YET TROTTERIZED
    troutine.apply(RZ((eps+U/2)*tau_full), q[1]) # NOT YET TROTTERIZED


    # # Trotterized application of [e^{-i H_1 dt}e^{-i H_2 dt}]^N
    # for _ in range(num_steps):
    #     # 1st term
    #     troutine.apply(CNOT, q[0], q[1])
    #     troutine.apply(RZ(-(U/2)*effective_delta_tau), q[1]) # NOT YET TROTTERIZED
    #     troutine.apply(CNOT, q[0], q[1])
    #     # 2nd term
    #     troutine.apply(RZ((eps+U/2)*effective_delta_tau), q[0]) # NOT YET TROTTERIZED
    #     troutine.apply(RZ((eps+U/2)*effective_delta_tau), q[1]) # NOT YET TROTTERIZED
    
    return troutine    

# Instructions to build the variational quantum circuit
def simple_circuit(theta):
    """Take a parameter theta and return the corresponding circuit"""
    Qrout = QRoutine()
    Qrout.apply(RX(theta[0]), 1)
    Qrout.apply(H, 0)
    Qrout.apply(CNOT, 0, 1)
    
    return Qrout




