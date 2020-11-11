import argparse
import os
import numpy as np

def statistical_analysis():
    """ Python utility script for UNI-xMD output analysis
    """
    parser = argparse.ArgumentParser(description="Python script for UNI-xMD output analysis")
    parser.add_argument('-n', '-ntrajs', action='store', dest='ntrajs', type=int, \
        help="Total trajectory number", required=True)
    parser.add_argument('-s', '-nsteps', action='store', dest='nsteps', type=int, \
        help="Total step number for analysis", required=True)
    parser.add_argument('-t', '-nstates', action='store', dest='nstates', type=int, \
        help="Total state number for analysis", required=True)
    args = parser.parse_args()

    # Indexing for numbering filename
    index = len(str(args.ntrajs))
    
    # Include step 0 
    nsteps1 = args.nsteps + 1
    
    averaged_running_state(args.ntrajs, index, nsteps1, args.nstates)
    averaged_density_matrix(args.ntrajs, index, nsteps1, args.nstates)
    averaged_nacme(args.ntrajs, index, nsteps1, args.nstates)

def averaged_running_state(ntrajs, index, nsteps, nstates):
    """ BO population analysis based on the running state of each trajectories
    """
    f_write = ""

    header = "#    Running state based averaged BO population"
    f_write += header

    # define empty array for summation
    avg_state = np.zeros((nstates, nsteps))
    # define variable for count trajectories except halted trajectories
    mtrajs = ntrajs

    for itraj in range(ntrajs):
        path = os.path.join(f"./TRAJ_{itraj + 1:0{index}d}/md/", "SHSTATE")

        with open(path, 'r') as f:
            # Skip header and read rest
            line = f.readline()
            line = f.read()
            lines = line.split()

        # read state information of entire steps
        rstate = np.array(lines[1::2][:nsteps], dtype=np.int)
        try:
            # sum over counted state number of each states 
            avg_state += np.array([(rstate == ist) for ist in range(nstates)], dtype=np.float)
        except ValueError:
            # exclude halted trajectories from total trajectory number
            mtrajs -= 1

    # average array and print
    avg_state /= mtrajs
    avg_data = "".join([("\n" + f"{istep:8d}" + "".join([f"{avg_state[istate, istep]:15.8f}" \
        for istate in range(nstates)])) for istep in range(nsteps)])
    f_write += avg_data

    typewriter(f_write, "AVG_POPRUN")

def averaged_density_matrix(ntrajs, index, nsteps, nstates):
    """ Electronic coherence analysis and BO population analysis
        based on the density matrix of each trajectories
    """
    f1_write = ""
    f2_write = ""

    header = "#    Averaged electronic coherence"
    f1_write += header

    header = "#    Density matrix based averaged BO population"
    f2_write += header

    # calculate number of off-diagonal elements from given nstates
    nstates_pair = int(nstates * (nstates - 1) / 2)
    # define empty array for summation
    avg_coh = np.zeros((nstates_pair, nsteps))
    avg_pop = np.zeros((nstates, nsteps))
    # define variable for count trajectories except halted trajectories
    mtrajs = ntrajs

    for itraj in range(ntrajs):
        path = os.path.join(f"./TRAJ_{itraj + 1:0{index}d}/md/", "BOPOP")

        with open(path, 'r') as f:
            # Skip header and read rest
            line = f.readline()
            line = f.read()
            lines = line.split()
            lines_list = list(map(float, lines))

        try:
            # sum over population of each states
            avg_pop += np.array([lines[istate::(nstates + 1)][:nsteps] for istate in range(1, nstates + 1)], dtype=np.float)
            # sum over coherence of each states, obtained from multiply istate population and jstate population
            avg_coh += np.array([np.multiply(lines_list[istate::(nstates + 1)][:nsteps],lines_list[jstate::(nstates + 1)][:nsteps]) \
                for istate in range(1, nstates + 1) for jstate in range(istate + 1, nstates + 1)])
        except ValueError:
            # exclude halted trajectories from total trajectory number
            mtrajs -= 1

    # average array and print
    avg_coh /= mtrajs
    avg_pop /= mtrajs

    avg_data = "".join([("\n" + f"{istep:8d}" + "".join([f"{avg_coh[istate, istep]:15.8f}" \
        for istate in range(nstates_pair)])) for istep in range(nsteps)])
    f1_write += avg_data

    avg_data = "".join([("\n" + f"{istep:8d}" + "".join([f"{avg_pop[istate, istep]:15.8f}" \
        for istate in range(nstates)])) for istep in range(nsteps)])
    f2_write += avg_data

    typewriter(f1_write, "AVG_COHRHO")
    typewriter(f2_write, "AVG_POPRHO")

def averaged_nacme(ntrajs, index, nsteps, nstates):
    """ Non-adiabatic coupling matrix analysis 
    """
    f_write = ""

    header = "#    Averaged Non-Adiabatic Coupling Matrix Eliments: off-diagonal"
    f_write += header

    # calculate number of off-diagonal elements from given nstates
    nstates_pair = int(nstates * (nstates - 1) / 2)
    # define empty array for summation
    avg_nacme = np.zeros((nstates_pair, nsteps))
    # define variable for count trajectories except halted trajectories
    mtrajs = ntrajs

    for itraj in range(ntrajs):
        path = os.path.join(f"./TRAJ_{itraj + 1:0{index}d}/md/", "NACME")

        with open(path, 'r') as f:
            # Skip header and read rest
            line = f.readline()
            line = f.read()
            lines = line.split()

        try:
            # sum over nacme of each state-state pair
            avg_nacme += abs(np.array([lines[istate::(nstates_pair + 1)][:nsteps] for istate in range(1, nstates_pair + 1)], dtype=np.float))
        except ValueError:
            # exclude halted trajectories from total trajectory number
            mtrajs -= 1

    # average array and print
    avg_nacme /= mtrajs
    avg_data = "".join([("\n" + f"{istep:8d}" + "".join([f"{avg_nacme[istate, istep]:15.8f}" \
        for istate in range(nstates_pair)])) for istep in range(nsteps)])
    f_write += avg_data

    typewriter(f_write, "AVG_NACME")

def typewriter(string, file_name):
    """ Function to write a string in filename
    """
    with open(file_name, "w") as f:
        f.write(string + "\n")

if (__name__ == "__main__"):
    statistical_analysis()

