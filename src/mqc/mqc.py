from __future__ import division
import numpy as np
from misc import fs_to_au
import textwrap

class MQC(object):
    """ Class for nuclear/electronic propagator used in MQC dynamics
    """
    def __init__(self, molecule, istate, dt, nsteps, nesteps, \
        propagation, l_adjnac):
        # Initialize input values
        self.istate = istate
        self.dt = dt * fs_to_au
        self.nsteps = nsteps
        self.nesteps = nesteps

        self.propagation = propagation

        self.l_adjnac = l_adjnac

        self.rforce = np.zeros((molecule.nat, molecule.nsp))

        # initialize coefficients and densities
        molecule.states[self.istate].coef = 1. + 0.j
        molecule.rho[self.istate, self.istate] = 1. + 0.j

    def cl_update_position(self, molecule):
        """ Routine to update nulcear postions
        """
        self.calculate_force(molecule)

        molecule.vel += 0.5 * self.dt * self.rforce / np.column_stack([molecule.mass] * molecule.nsp)
        molecule.pos += self.dt * molecule.vel

    def cl_update_velocity(self, molecule):
        """ Rotine to update nulcear velocities
        """
        self.calculate_force(molecule)

        molecule.vel += 0.5 * self.dt * self.rforce / np.column_stack([molecule.mass] * molecule.nsp)
        molecule.update_kinetic()

#    def calculate_temperature(self, molecule):
#        """ Routine to calculate current temperature
#        """
#        pass
#    #    self.temperature = molecule.ekin * 2 / float(molecule.dof) * au_to_K

    def calculate_force(self):
        """ Routine to calculate the forces
        """
        pass

    def update_potential(self):
        """ Routine to update the potential of molecules
        """
        pass

    #def print_init(self, molecule, theory, thermostat, debug):
    def print_init(self, molecule, theory, thermostat):
        """ Routine to print the initial information of dynamics
        """
        # print molecule information: coordinate, velocity
        molecule.print_init()

        # print dynamics information
        qm_prog = str(theory.__class__).split('.')[1]
        qm_method = theory.__class__.__name__
        md_type = self.__class__.__name__
        if (md_type == "SH"):
            method = "FSSH"
        elif (md_type == "Eh"):
            method = "Ehrenfest"
        elif (md_type == "SHXF"):
            method = "SHXF"
        elif (md_type == "BOMD"):
            method = "BOMD"

        # TODO : naming for n.p e.p p
        dynamics_info = textwrap.dedent(f"""\
        {"-" * 68}
        {"Dynamics Information":>43s}
        {"-" * 68}
          QM Program               = {qm_prog:>16s}
          QM Method                = {qm_method:>16s}

          MQC Method               = {method:>16s}
          Time Interval (au)       = {self.dt:16.6f}
          Initial State (0:GS)     = {self.istate:>16d}
          Nucleus Propagation      = {self.nsteps:>16d}
        """)
        if (method != "BOMD"):
            dynamics_info += f"  Electron Propagation     = {self.nesteps:>16d}\n"
            dynamics_info += f"  Propagation              = {self.propagation:>16s}\n"
        print (dynamics_info, flush=True)

        # print thermostat information
        thermostat.print_init()
        # TODO : efficient method for printing thermostat information
#        print (f"{'-'*118}",flush=True)
#        print (f"{'Thermostat information':>69}",flush=True)
#        print (f"{'-'*118}",flush=True)

        # print dynamics information for each step
        if (method == "FSSH" or method == "SHXF"):
            INIT = f" #INFO{'STEP':>8s}{'State':>7s}{'Max. Prob.':>14s}{'Rand.':>12s}{'Kinetic(H)':>15s}{'Potential(H)':>15s}{'Total(H)':>13s}{'Temperature(K)':>17s}{'Norm.':>8s}"
            DEBUG2 = f" #DEBUG2{'STEP':>6s}{'Acc. Hopping Prob.':>22s}"
        elif (method == "Ehrenfest"):
            INIT = f" #INFO{'STEP':>8s}{'Kinetic(H)':>15s}{'Potential(H)':>15s}{'Total(H)':>13s}{'Temperature(K)':>17s}{'norm':>8s}"
            DEBUG2 = ""
        elif (method == "BOMD"):
            INIT = f" #INFO{'STEP':>8s}{'State':>7s}{'Kinetic(H)':>13s}{'Potential(H)':>15s}{'Total(H)':>13s}{'Temperature(K)':>17s}"
            DEBUG2 = ""
    
        DEBUG1 = f" #DEBUG1{'STEP':>6s}"
        for ist in range(molecule.nst):
            DEBUG1 += f"{'Potential_':>14s}{ist}(H)"

        dynamics_step_info = textwrap.dedent(f"""\

        {"-" * 118}
        {"Start Dynamics":>65s}
        {"-" * 118}
        """)
        dynamics_step_info += INIT
        dynamics_step_info += "\n" + DEBUG1
        dynamics_step_info += "\n" + DEBUG2
        print (dynamics_step_info, flush=True)


