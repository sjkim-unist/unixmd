from __future__ import division
from build.cioverlap import *
from bo.dftbplus.dftbplus import DFTBplus
from bo.dftbplus.dftbpar import spin_w, max_l
from misc import eV_to_au, elapsed_time
import os, shutil, re, textwrap, struct
import numpy as np

class DFTB(DFTBplus):
    """ Class for (TD)DFTB method of DFTB+ program

        :param object molecule: molecule object
        :param boolean scc: include SCC scheme
        :param double scc_tol: energy convergence for SCC iterations
        :param integer max_scc_iter: maximum number of SCC iterations
        :param boolean sdftb: include spin-polarisation scheme
        :param double unpaired_e: number of unpaired electrons
        :param double e_temp: electronic temperature for Fermi-Dirac scheme
        :param string mixer: charge mixing method used in DFTB
        :param string ex_symmetry: symmetry (singlet or triplet) in TDDFTB
        :param string sk_path: path for slater-koster files
        :param boolean periodic: use periodicity in the calculations
        :param double a(b, c)_axis: the length of cell lattice
        :param string qm_path: path for QM binary
        :param string script_path: path for DFTB+ python script (dptools)
        :param integer nthreads: number of threads in the calculations
        :param boolean mpi: use MPI parallelization
        :param string mpi_path: path for MPI binary
        :param double version: version of DFTB+ program
    """
    def __init__(self, molecule, scc=True, scc_tol=1E-6, max_scc_iter=100, \
        sdftb=False, unpaired_e=0., e_temp=0., mixer="Broyden", \
        ex_symmetry="singlet", sk_path="./", periodic=False, a_axis=0., b_axis=0., c_axis=0., \
        qm_path="./", script_path="./", nthreads=1, mpi=False, mpi_path="./", version=19.1):
        # Initialize DFTB+ common variables
        super().__init__(molecule, sk_path, qm_path, script_path, nthreads, version)

        # Initialize DFTB+ DFTB variables
        self.scc = scc
        self.scc_tol = scc_tol
        self.max_scc_iter = max_scc_iter

        self.sdftb = sdftb
        self.unpaired_e = unpaired_e

        self.e_temp = e_temp
        self.mixer = mixer

        self.ex_symmetry = ex_symmetry

        self.mpi = mpi
        self.mpi_path = mpi_path

        self.periodic = periodic
        self.a_axis = a_axis
        self.b_axis = b_axis
        self.c_axis = c_axis

        # Check excitation symmetry in TDDFTB
        if (not (self.ex_symmetry == "singlet" or self.ex_symmetry == "triplet")):
            raise ValueError (f"( {self.qm_method}.{call_name()} ) Wrong symmetry for excited state! {self.ex_symmetry}")

        # Set 'l_nacme' and 're_calc' with respect to the computational method
        # TDDFTB do not produce NACs, so we should get NACME from CIoverlap
        # TDDFTB cannot compute the gradient of several states simultaneously.
        molecule.l_nacme = True
        self.re_calc = True

        # Calculate number of basis for current system
        # Set new variable to decide the position of basis functions in terms of atoms
        self.norb = 0
#        self.check_atom = [0]
        for iat in range(molecule.nat):
            max_ang = max_l[molecule.symbols[iat]]
            if (max_ang == 's'):
                self.norb += 1
            elif (max_ang == 'p'):
                self.norb += 4
#            self.check_atom.append(self.norb)

        # Set new variable to decide the position of atoms in terms of basis functions
#        self.check_basis = []
#        for ibasis in range(self.norb):
#            for iat in range(molecule.nat):
#                ind_a = self.check_atom[iat] + 1
#                ind_b = self.check_atom[iat + 1]
#                if (ibasis + 1 >= ind_a and ibasis + 1 <= ind_b):
#                    self.check_basis.append(iat + 1)

        # Initialize NACME variables
        # There is no core orbitals in TDDFTB (fixed occupations)
        # nocc is number of occupied orbitals and nvirt is number of virtual orbitals
        self.nocc = int(int(molecule.nelec) / 2)
        self.nvirt = self.norb - self.nocc

        self.ao_overlap = np.zeros((self.norb, self.norb))
        self.mo_coef_old = np.zeros((self.norb, self.norb))
        self.mo_coef_new = np.zeros((self.norb, self.norb))
        self.ci_coef_old = np.zeros((molecule.nst, self.nocc, self.nvirt))
        self.ci_coef_new = np.zeros((molecule.nst, self.nocc, self.nvirt))

    def get_bo(self, molecule, base_dir, istep, bo_list, calc_force_only):
        """ Extract energy, gradient and nonadiabatic couplings from (TD)DFTB method

            :param object molecule: molecule object
            :param string base_dir: base directory
            :param integer istep: current MD step
            :param integer,list bo_list: list of BO states for BO calculation
            :param boolean calc_force_only: logical to decide whether calculate force only
        """
        self.copy_files(molecule, istep, calc_force_only)
        super().get_bo(base_dir, calc_force_only)
        self.write_xyz(molecule)
        self.get_input(molecule, istep, bo_list, calc_force_only)
        self.run_QM(molecule, base_dir, istep, bo_list, calc_force_only)
        self.extract_BO(molecule, base_dir, istep, bo_list, calc_force_only)
        self.move_dir(base_dir)

    def copy_files(self, molecule, istep, calc_force_only):
        """ Copy necessary scratch files in previous step

            :param object molecule: molecule object
            :param integer istep: current MD step
            :param boolean calc_force_only: logical to decide whether calculate force only
        """
        if (self.calc_coupling and not calc_force_only and istep >= 0 and molecule.nst > 1):
            # After T = 0.0 s
            shutil.copy(os.path.join(self.scr_qm_dir, "geometry.xyz"), \
                os.path.join(self.scr_qm_dir, "../geometry.xyz.pre"))
            shutil.copy(os.path.join(self.scr_qm_dir, "eigenvec.bin"), \
                os.path.join(self.scr_qm_dir, "../eigenvec.bin.pre"))
            shutil.copy(os.path.join(self.scr_qm_dir, "XplusY.DAT"), \
                os.path.join(self.scr_qm_dir, "../XplusY.DAT.pre"))

    def get_input(self, molecule, istep, bo_list, calc_force_only):
        """ Generate DFTB+ input files: geometry.gen, dftb_in.hsd

            :param object molecule: molecule object
            :param integer istep: current MD step
            :param integer,list bo_list: list of BO states for BO calculation
            :param boolean calc_force_only: logical to decide whether calculate force only
        """
        # TODO : currently, CIoverlap is not correct -> only BOMD possible with TDDFTB
        #if (self.calc_coupling):
        #    raise ValueError (f"( {self.qm_method}.{call_name()} ) only BOME possible with TDDFTB! {self.qm_method}")

        # Make 'geometry.gen' file
        os.system("xyz2gen geometry.xyz")
        if (self.periodic):
            # Substitute C to S in first line
            file_be = open('geometry.gen', 'r')
            file_af = open('tmp.gen', 'w')
            first_row = True
            for row in file_be:
                if (first_row):
                    row = f'{molecule.nat} S\n'
                    first_row = False
                file_af.write(row)
            # Add gamma-point and cell lattice information
            geom_periodic = textwrap.dedent(f"""\
            {0.0:15.8f} {0.0:15.8f} {0.0:15.8f}
            {self.a_axis:15.8f} {0.0:15.8f} {0.0:15.8f}
            {0.0:15.8f} {self.b_axis:15.8f} {0.0:15.8f}
            {0.0:15.8f} {0.0:15.8f} {self.c_axis:15.8f}
            """)
            file_af.write(geom_periodic)
            file_be.close()
            file_af.close()
            os.rename('tmp.gen', 'geometry.gen')

        # Make 'double.gen' file for overlap in TDDFTB
        # In this case, we do not need to consider periodicity
        if (self.calc_coupling and not calc_force_only and istep >= 0 and molecule.nst > 1):
            # Move previous files to currect directory
            os.rename('../geometry.xyz.pre', './geometry.xyz.pre')
            os.rename('../eigenvec.bin.pre', './eigenvec.bin.pre')
            os.rename('../XplusY.DAT.pre', './XplusY.DAT.pre')
            # Open geometry.xyz.pre
            file_af = open('double.xyz', 'w')
            file_be = open('geometry.xyz.pre', 'r')
            first_row = True
            for row in file_be:
                if (first_row):
                    row = f'{molecule.nat * 2}\n'
                    first_row = False
                file_af.write(row)
            file_be.close()
            # Open geometry.xyz
            file_be = open('geometry.xyz', 'r')
            iline = 1
            for row in file_be:
                if (iline > 2):
                    file_af.write(row)
                iline += 1
            file_be.close()
            file_af.close()
            os.system("xyz2gen double.xyz")

        # Make 'dftb_in.hsd' file
        input_dftb = ""

        # Geometry Block
        input_geom = textwrap.dedent(f"""\
        Geometry = GenFormat{{
          <<< 'geometry.gen'
        }}
        """)
        input_dftb += input_geom

        # Hamiltonian Block
        input_ham_init = textwrap.dedent(f"""\
        Hamiltonian = DFTB{{
        """)
        input_dftb += input_ham_init

        if (self.scc):
            input_ham_scc = textwrap.indent(textwrap.dedent(f"""\
              SCC = Yes
              SCCTolerance = {self.scc_tol}
              MaxSCCIterations = {self.max_scc_iter}
              Mixer = {self.mixer}{{}}
            """), "  ")
            input_dftb += input_ham_scc

            if (self.sdftb and molecule.nst == 1):
                input_ham_spin = textwrap.dedent(f"""\
                SpinPolarisation = Colinear{{
                  UnpairedElectrons = {self.unpaired_e}
                }}
                """)
                input_dftb += input_ham_spin

            if (self.sdftb or self.ex_symmetry == "triplet"):
                spin_constant = ("\n" + " " * 18).join([f"  {itype} = {{ {spin_w[f'{itype}']} }}" for itype in self.atom_type])
                input_ham_spin_w = textwrap.indent(textwrap.dedent(f"""\
                  SpinConstants = {{
                    ShellResolvedSpin = Yes
                  {spin_constant}
                  }}
                """), "  ")
                input_dftb += input_ham_spin_w

            # TODO : read initial guess from previous step
#            if (calc_force_only):
#                input_ham_restart = textwrap.indent(textwrap.dedent(f"""\
#                  ReadInitialCharges = Yes
#                """), "  ")
#                input_dftb += input_ham_restart

        # TODO: for QM/MM, point_charge??

        if (self.periodic):
            input_ham_periodic = textwrap.indent(textwrap.dedent(f"""\
              KPointsAndWeights = {{
                0.0 0.0 0.0 1.0
              }}
            """), "  ")
            input_dftb += input_ham_periodic

        angular_momentum = ("\n" + " " * 10).join([f"  {itype} = '{max_l[f'{itype}']}'" for itype in self.atom_type])
        input_ham_basic = textwrap.dedent(f"""\
          Charge = {molecule.charge}
          Filling = Fermi{{
            Temperature[K] = {self.e_temp}
          }}
          MaxAngularMomentum = {{
          {angular_momentum}
          }}
          SlaterKosterFiles = Type2FileNames{{
            Prefix = '{self.sk_path}'
            Separator = '-'
            Suffix = '.skf'
            LowerCaseTypeName = No
          }}
        }}
        """)
        input_dftb += input_ham_basic

        # Analysis Block
        input_analysis = textwrap.dedent(f"""\
        Analysis = {{
          CalculateForces = Yes
          WriteBandOut = Yes
          WriteEigenvectors = Yes
          MullikenAnalysis = Yes
        }}
        """)
        input_dftb += input_analysis

        # Options Block
        input_options = textwrap.dedent(f"""\
        Options = {{
          WriteDetailedXml = Yes
          WriteDetailedOut = Yes
          TimingVerbosity = -1
        }}
        """)
        input_dftb += input_options

        # ExcitedState Block
        if (molecule.nst > 1):

            # Calculate excited state force for target state
            if (bo_list[0] > 0):
                ex_force = "Yes"
                rst = bo_list[0]
            else:
                ex_force = "No"
                rst = bo_list[0] + 1

            # Set number of excitations in TDDFTB
            # This part can be modified by users
            if (molecule.nat <= 5):
                num_ex = molecule.nst + 2
            elif (molecule.nat > 5 and molecule.nat <= 15):
                num_ex = 2 * molecule.nst + 2
            else:
                num_ex = 3 * molecule.nst + 2

            # Write XplusY data?
            if (self.calc_coupling):
                xpy = "Yes"
            else:
                xpy = "No"

            input_excited = textwrap.dedent(f"""\
            ExcitedState = Casida{{
              NrOfExcitations = {num_ex}
              StateOfInterest = {rst}
              Symmetry = {self.ex_symmetry}
              WriteTransitions = Yes
              WriteMulliken = Yes
              WriteXplusY = {xpy}
              ExcitedStateForces = {ex_force}
            }}
            """)
            input_dftb += input_excited

        # ParserOptions Block
        if (self.version == 19.1):
            parser_version = 7
        elif (self.version == 20.1):
            parser_version = 8

        input_parseroptions = textwrap.dedent(f"""\
        ParserOptions = {{
          ParserVersion = {parser_version}
        }}
        """)
        input_dftb += input_parseroptions

        # Parallel Block
        if (self.mpi):
            if (self.sdftb and self.nthreads > 1):
                groups = 2
            else:
                groups = 1
            input_parallel = textwrap.dedent(f"""\
            Parallel = {{
              Groups = {groups}
              UseOmpThreads = No
              Blacs = BlockSize {{ 32 }}
            }}
            """)
            input_dftb += input_parallel

        # Write 'dftb_in.hsd.geom' file
        file_name = "dftb_in.hsd.geom"
        with open(file_name, "w") as f:
            f.write(input_dftb)

        # Write 'dftb_in.hsd.double' file
        if (self.calc_coupling and not calc_force_only and istep >= 0 and molecule.nst > 1):

            # New input for dftb
            input_dftb = ""

            # Geometry Block
            input_geom = textwrap.dedent(f"""\
            Geometry = GenFormat{{
              <<< 'double.gen'
            }}
            """)
            input_dftb += input_geom
            input_dftb += input_ham_init
            input_dftb += input_ham_basic

            # Options Block
            input_options = textwrap.dedent(f"""\
            Options = {{
              WriteDetailedXml = Yes
              WriteDetailedOut = Yes
              WriteHS = Yes
              TimingVerbosity = -1
            }}
            """)
            input_dftb += input_options

            file_name = "dftb_in.hsd.double"
            with open(file_name, "w") as f:
                f.write(input_dftb)

    def run_QM(self, molecule, base_dir, istep, bo_list, calc_force_only):
        """ Run (TD)DFTB calculation and save the output files to QMlog directory

            :param object molecule: molecule object
            :param string base_dir: base directory
            :param integer istep: current MD step
            :param integer,list bo_list: list of BO states for BO calculation
            :param boolean calc_force_only: logical to decide whether calculate force only
        """
        # Set run command
        qm_command = os.path.join(self.qm_path, "dftb+")
        if (self.mpi):
            # MPI setting
            os.environ["OMP_NUM_THREADS"] = "1"
            mpi_command = os.path.join(self.mpi_path, "mpirun")
            command = f"{mpi_command} -np {self.nthreads} {qm_command} > log"
        else:
            # OpenMP setting
            os.environ["OMP_NUM_THREADS"] = f"{self.nthreads}"
            command = f"{qm_command} > log"

        # Run DFTB+ for calculation of overlap matrix
        if (self.calc_coupling and not calc_force_only and istep >= 0 and molecule.nst > 1):
            shutil.copy("dftb_in.hsd.double", "dftb_in.hsd")
            os.system(command)

        # Run DFTB+ method for molecular dynamics
        shutil.copy("dftb_in.hsd.geom", "dftb_in.hsd")
        os.system(command)

        # Copy the output file to 'QMlog' directory
        tmp_dir = os.path.join(base_dir, "QMlog")
        if (os.path.exists(tmp_dir)):
            log_step = f"log.{istep + 1}.{bo_list[0]}"
            shutil.copy("log", os.path.join(tmp_dir, log_step))

    def extract_BO(self, molecule, base_dir, istep, bo_list, calc_force_only):
        """ Read the output files to get BO information

            :param object molecule: molecule object
            :param string base_dir: base directory
            :param integer istep: current MD step
            :param integer,list bo_list: list of BO states for BO calculation
            :param boolean calc_force_only: logical to decide whether calculate force only
        """
        # Read 'detailed.out' file
        # TODO: the qmmm information is written in this file
        file_name = "detailed.out"
        with open(file_name, "r") as f:
            detailed_out = f.read()

        # Read 'EXC.DAT' file
        if (molecule.nst > 1):
            file_name = "EXC.DAT"
            with open(file_name, "r") as f:
                exc_out = f.read()

        # Energy
        if (not calc_force_only):
            for states in molecule.states:
                states.energy = 0.

            energy = re.findall('Total energy:\s+([-]\S+) H', detailed_out)
            energy = np.array(energy[0])
            energy = energy.astype(float)
            molecule.states[0].energy = energy

            if (molecule.nst > 1):
                tmp_e = f'[=]+\n' + ('\s+([-]*\S+)\s+\S+\s+\d+\s+->\s+\d+\s+\S+\s+\S+\s+[ST]') * molecule.nst
                energy = re.findall(tmp_e, exc_out)
                energy = np.array(energy[0])
                energy = energy.astype(float)
                energy *= eV_to_au
                for ist in range(1, molecule.nst):
                    molecule.states[ist].energy = molecule.states[0].energy + energy[ist - 1]

        # Force
        if (not calc_force_only):
            for states in molecule.states:
                states.force = np.zeros((molecule.nat, molecule.nsp))

        tmp_f = 'Total Forces' + '\n\s+\d*\s+([-]*\S+)\s+([-]*\S+)\s+([-]*\S+)' * molecule.nat
        force = re.findall(tmp_f, detailed_out)
        force = np.array(force[0])
        force = force.astype(float)
        force = force.reshape(molecule.nat, 3, order='C')
        molecule.states[bo_list[0]].force = np.copy(force)

        # NACME
        if (self.calc_coupling and not calc_force_only):
            molecule.nacme = np.zeros((molecule.nst, molecule.nst))
            if (istep >= 0):
                # TODO : current TDNAC gives too large values
                self.CI_overlap(molecule, base_dir)
                # Read 'NACME.DAT'
#                ist = 0
#                jst = 0
#                nline = 1
#                file_name_in = "NACME.DAT"
#                with open(file_name_in, "r") as f_in:
#                    lines = f_in.readlines()
#                    for line in lines:
#                        field = line.split()
#                        if (nline % 2 == 0):
#                            # TODO : current TDNAC gives too large values
#                            #molecule.nacme[ist, jst] = field[0]
#                            jst += 1
#                            if (jst == molecule.nst):
#                                ist += 1
#                                jst = 0
#                        nline += 1

    # TODO : elapsed_time is needed?
#    @elapsed_time
    def CI_overlap(self, molecule, base_dir):
        """ Read the necessary files and generate NACME file and
            this is an experimental feature and not used

            :param object molecule: molecule object
            :param string base_dir: base directory
        """
        # TODO: here, the required process = save the variables related to NACME

        # Read upper right block of 'oversqr.dat' file (< t | t+dt >)
        file_name_in = "oversqr.dat"

        # 1st method using np.method
#        self.ao_overlap = np.zeros((self.norb, self.norb))
#        over_mat = np.loadtxt(file_name_in, skiprows=5, dtype=np.float)
#        nan_ind = np.argwhere(np.isnan(over_mat))
##        for row, col in nan_ind:
##            ind_a = self.check_basis[row%self.norb]
##            ind_b = self.check_basis[col%self.norb]
##            if (ind_a == ind_b):
##                if (row%self.norb == col%self.norb):
##                    over_mat[row, col] = 1.
##                else:
##                    over_mat[row, col] = 0.
#        for row in range(self.norb):
#            for col in range(self.norb, 2 * self.norb):
#                self.ao_overlap[row, col - self.norb] = over_mat[row, col]
#        np.savetxt("test1", self.ao_overlap, fmt=f"%6.3f")
#        np.savetxt("test3", over_mat, fmt=f"%6.3f")

        # 2nd method using direct read - faster than 1st method
        self.ao_overlap = np.zeros((self.norb, self.norb))
        with open(file_name_in, "r") as f_in:
            lines = f_in.readlines()
            row = 0
            iline = 0
            for line in lines:
                # Skip first five lines and read upper block
                if (iline in range(5, 5 + self.norb)):
                    col = 0
                    count = False
                    field = line.split()
                    for element in field:
                        # Read right block
                        if (count):
                            # TODO : how to consider too large value, 26.6 in this case?
                            # TODO : set to 1?
                            if (element == 'NaN'):
                                # TODO : Choose only onsite (same-atom) block, is this essential part?
#                                ind_a = self.check_basis[row]
#                                ind_b = self.check_basis[col]
#                                if (ind_a == ind_b):
#                                    if (row == col):
#                                        new_val = 1.
#                                    else:
#                                        new_val = 0.
                                if (row == col):
                                    # Diagonal element in onsite (same-atom) block
                                    new_val = 1.
                                else:
                                    # Off-diagonal element in onsite (same-atom) block
                                    new_val = 0.
                            else:
                                new_val = float(element)
                            # Set overlap matrix element
                            if (col in range(self.norb)):
                                self.ao_overlap[row, col] = new_val
                        col += 1
                        # Read right block
                        if (col > self.norb - 1):
                            col -= self.norb
                            count = True
                    row += 1
                iline += 1
        np.savetxt("test2", self.ao_overlap, fmt=f"%6.3f")

        wf_overlap(self, molecule)

#        # Set new variable to decide the number of basis functions for atoms
#        check_atom = [0]
#        num_basis = 0
#        core_elec = 0.
#        for iat in range(molecule.nat):
#            max_ang = max_l[molecule.symbols[iat]]
#            if (max_ang == 'p'):
#                num_basis += 4
#                core_elec += 2.
#                check_atom.append(num_basis)
#            elif (max_ang == 's'):
#                num_basis += 1
#                check_atom.append(num_basis)
#
#        # Set new variable to decide the position of atoms in basis functions
#        check_basis = []
#        for ibasis in range(num_basis):
#            for iat in range(molecule.nat):
#                ind_a = check_atom[iat] + 1
#                ind_b = check_atom[iat + 1]
#                if (ibasis + 1 >= ind_a and ibasis + 1 <= ind_b):
#                    check_basis.append(iat + 1)
#
#        # Write 'INPUT' file
#        ncore = 0
#        nocc = int(int(molecule.nelec - core_elec) / 2) - ncore
#        nvirt = num_basis - nocc - ncore
#
#        file_name_out = "INPUT"
#        f_out = open(file_name_out, "w")
#
#        f_print = f"{num_basis:5d} {ncore:4d} {nocc:4d} {nvirt:4d} {molecule.nst:3d} 5.16767" + "\n"
#        f_out.write(f_print)
#
#        f_out.close()



#        # Write 'MOCOEF' file
#        file_name_out = "MOCOEF"
#        f_out = open(file_name_out, "w")
#
#        file_name_in = "eigenvec.bin"
#        mocoef = []
#        with open(file_name_in, "rb") as f_in:
#            dummy = np.fromfile(f_in, dtype=np.integer, count = 1)
#            for ibasis in range(num_basis):
#                dummy = np.fromfile(f_in, dtype=np.integer, count = 1)
#                data = np.fromfile(f_in, dtype=np.float64, count = num_basis)
#                mocoef.append(data)
#            mocoef = np.array(mocoef)
#            mocoef = np.transpose(mocoef)
#            mocoef = [val for sublist in mocoef for val in sublist]
#            for ibasis in range(num_basis):
#                ind_a = num_basis * ibasis
#                ind_b = num_basis * (ibasis + 1)
#                f_print = " ".join([f"{mocoef[ind]:13.8f}" for ind in range(ind_a, ind_b)]) + "\n"
#                f_out.write(f_print)
#
#        f_out.close()
#
#        # Write 'MOCOEFOLD' file
#        file_name_out = "MOCOEFOLD"
#        f_out = open(file_name_out, "w")
#
#        file_name_in = "eigenvec.bin.pre"
#        mocoef = []
#        with open(file_name_in, "rb") as f_in:
#            dummy = np.fromfile(f_in, dtype=np.integer, count = 1)
#            for ibasis in range(num_basis):
#                dummy = np.fromfile(f_in, dtype=np.integer, count = 1)
#                data = np.fromfile(f_in, dtype=np.float64, count = num_basis)
#                mocoef.append(data)
#            mocoef = np.array(mocoef)
#            mocoef = np.transpose(mocoef)
#            mocoef = [val for sublist in mocoef for val in sublist]
#            for ibasis in range(num_basis):
#                ind_a = num_basis * ibasis
#                ind_b = num_basis * (ibasis + 1)
#                f_print = " ".join([f"{mocoef[ind]:13.8f}" for ind in range(ind_a, ind_b)]) + "\n"
#                f_out.write(f_print)
#
#        f_out.close()
#
#        # Write 'CICOEF' file
#        file_name_out = "CICOEF"
#        f_out = open(file_name_out, "w")
#
#        file_name_in = "XplusY.DAT"
#        with open(file_name_in, "r") as f_in:
#            lines = f_in.readlines()
#            nline = 1
#            ind_b = -1
#            for line in lines:
#                if (nline == 1):
#                    field = line.split()
#                    nmat = int(field[0])
#                    nexc = int(field[1])
#                    nstd = int(nmat / 6) + 1
#                    if (nmat % 6 != 0):
#                        nstd += 1
#                    xply = np.zeros((nocc, nvirt, nexc))
#                else:
#                    if ((nline - 1) % nstd == 1):
#                        ind_occ = nocc - 1
#                        ind_virt = 0
#                        ind_b += 1
#                    else:
#                        field = line.split()
#                        for element in field:
#                            xply[ind_occ, ind_virt, ind_b] = float(element)
#                            if (ind_occ == 0):
#                                ind_occ = nocc - 1
#                                ind_virt += 1
#                            else:
#                                ind_occ -= 1
#                nline += 1
#
#        for iexc in range(nexc):
#
#            # Normalize the CI coefficients
#            norm_val = 0.
#            for iocc in range(nocc):
#                for ivirt in range(nvirt):
#                    norm_val += xply[iocc, ivirt, iexc] ** 2
#            norm_val = np.sqrt(norm_val)
#            for iocc in range(nocc):
#                for ivirt in range(nvirt):
#                    xply[iocc, ivirt, iexc] /= norm_val
#
#            f_print = f"{iexc + 1:4d}" + "\n"
#            f_out.write(f_print)
#
#            for iocc in range(nocc):
#                for ivirt in range(nvirt):
#                    f_print = f"{xply[iocc, ivirt, iexc]:13.8f}"
#                    if (ivirt == nvirt - 1):
#                        f_print += "\n"
#                    f_out.write(f_print)
#
#        f_out.close()
#
#        # Write 'CICOEFOLD' file
#        file_name_out = "CICOEFOLD"
#        f_out = open(file_name_out, "w")
#
#        file_name_in = "XplusY.DAT.pre"
#        with open(file_name_in, "r") as f_in:
#            lines = f_in.readlines()
#            nline = 1
#            ind_b = -1
#            for line in lines:
#                if (nline == 1):
#                    field = line.split()
#                    nmat = int(field[0])
#                    nexc = int(field[1])
#                    nstd = int(nmat / 6) + 1
#                    if (nmat % 6 != 0):
#                        nstd += 1
#                    xply = np.zeros((nocc, nvirt, nexc))
#                else:
#                    if ((nline - 1) % nstd == 1):
#                        ind_occ = nocc - 1
#                        ind_virt = 0
#                        ind_b += 1
#                    else:
#                        field = line.split()
#                        for element in field:
#                            xply[ind_occ, ind_virt, ind_b] = float(element)
#                            if (ind_occ == 0):
#                                ind_occ = nocc - 1
#                                ind_virt += 1
#                            else:
#                                ind_occ -= 1
#                nline += 1
#
#        for iexc in range(nexc):
#
#            # Normalize the CI coefficients
#            norm_val = 0.
#            for iocc in range(nocc):
#                for ivirt in range(nvirt):
#                    norm_val += xply[iocc, ivirt, iexc] ** 2
#            norm_val = np.sqrt(norm_val)
#            for iocc in range(nocc):
#                for ivirt in range(nvirt):
#                    xply[iocc, ivirt, iexc] /= norm_val
#
#            f_print = f"{iexc + 1:4d}" + "\n"
#            f_out.write(f_print)
#
#            for iocc in range(nocc):
#                for ivirt in range(nvirt):
#                    f_print = f"{xply[iocc, ivirt, iexc]:13.8f}"
#                    if (ivirt == nvirt - 1):
#                        f_print += "\n"
#                    f_out.write(f_print)
#
#        f_out.close()
#
#        # TODO: this is temporary path, the directory for tdnac.x can be changed
##        tdnac_command = os.path.join(base_dir, "../tdnac.x")
##        command = f"{tdnac_command}"
##        os.system(command)


