import tkinter as tk
from tkinter import *
from tkinter import filedialog, Tk
from tkinter import messagebox
from tkinter import ttk

import matplotlib
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions, MolDrawing

# External tool to calculate chemical bond information
import xyz2mol

matplotlib.use('TkAgg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

import os
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


# The ring detection is processed by class FindRing
class FindRing:
    def __init__(self, bond_distance_upper, bond_distance_lower, cluster_size, span, axis, file_name, fileOut):
        self.sanity = None
        self.count = None
        self.planeAllowance = None
        self.file = None
        self.bond_distance_upper = bond_distance_upper
        self.bond_distance_lower = bond_distance_lower
        self.cluster_size = cluster_size
        self.span = span
        self.axis = axis
        self.file_name = file_name
        self.fileOut = fileOut
        self.sep = None

    # read data from the data (.xyz) file
    # the code only considers one single timeStep at a single run
    # The first row of the data file should be the number of atom
    # The second row of the data file indicates the timeStep
    # The code assumes that the co-ordinates start from the third row.
    # Returns the input carbon coordinates in the form of panda dataframe
    def get_data(self):
        df = pd.read_csv(
            filepath_or_buffer=self.file_name,
            header=None,
            sep=self.sep,
            skiprows=2,
            engine='python'
        )
        if df.shape[1] == 4:
            df = df[~df[0].astype(str).str.startswith('2')]
        df = df[df.columns[-3:]]
        # df.to_csv(self.fileOut + '/' + str((df.shape[0])) + '_input.csv', index=None)
        return df

    # Get the timeStep from the input data file
    # Read and Return timestep
    def get_timeStep(self, file_name, string_to_search):
        line_number = 0
        TimeStep = None
        with open(file_name, 'r') as read_obj:
            for line in read_obj:
                line_number += 1
                if string_to_search in line:
                    TimeStep = line.rstrip('\n')
                if line_number == 5:
                    break
        return TimeStep

    # Writes the results into .xyz formatted file
    # Returns the aromatic carbon coordinates in panda dataframe
    def write_to_xyz(self, data, frames):
        result = pd.concat(frames)
        result = result.drop_duplicates()
        result = result.reset_index(drop=True)
        XYZ = result
        add_number = pd.DataFrame({len(result)}, index=[0])
        if self.get_timeStep(self.file_name, 'Timestep') == None:
            declare_TimeStep = pd.DataFrame({'Atoms. Timestep: 0'}, index=[0])
        else:
            declare_TimeStep = pd.DataFrame({self.get_timeStep(self.file_name, 'Timestep')}, index=[1])
        XYZ['particle_type'] = 'c'
        XYZ = pd.concat([declare_TimeStep, XYZ.iloc[:]]).reset_index(drop=True)
        XYZ = pd.concat([add_number, XYZ.iloc[:]]).reset_index(drop=True)
        XYZ = XYZ[['particle_type', 0, 1, 2]]
        XYZ.to_csv(self.fileOut + '/' + e11.get() + '_' + self.file, index=None,
                   header=None, sep='\t')
        result = result[[0, 1, 2]]
        for i in range(3):
            result[3 - i] = result[2 - i]
        result = result.reindex(columns=data.columns)
        return result

    # Division scheme for adaptive decomposition
    # Span is the length of the decomposition cube
    # Returns number of division
    def division_adaptive(self, df, span, axis):
        x = np.array(df)
        n = df.shape[0]
        minimum = np.amin(x, axis=0)[axis]
        maximum = np.amax(x, axis=0)[axis]
        division = int(np.ceil(maximum - minimum) / span)
        return division

    # Division scheme based on Number of ATOMS in each division
    # Returns number of division
    def division_number(self, df, cluster_size):
        n = df.shape[0]
        division = int(np.ceil(n / cluster_size))

        return division

    # Compare cycles to eliminate duplicates
    # Check two cycle and Returns if they are the same (getting rid of orientation)

    def checkDuplicate(self, neddleCycle, cycles):
        neddleCycle = np.asarray(neddleCycle)
        cycles = np.asarray(cycles)

        for cycle in cycles:
            if len(cycle) != neddleCycle.shape[0]:
                continue
            if np.all(np.isin(neddleCycle, cycle)):
                return True
        return False

    # For imposing planarity
    # Takes four points, form equation of plane using first three and takes box product with the fourth point
    # Check and return if 4 points are in a plane
    def planar_check(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, x, y, z):
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1
        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = (- a * x1 - b * y1 - c * z1)
        return abs(a * x + b * y + c * z + d) <= np.cos(np.pi * (90 - self.planeAllowance) / 180) * np.sqrt(
            x ** 2 + y ** 2 + z ** 2)

    # Check and Return if one cycle is a subset of others (gives idea about molecular rings)
    def isSubSet(self, current, toCompare):
        return np.in1d(toCompare, current).all()

    # Sanity check run
    # takes the output from the adaptive split and run them again with non-overlapping axis split
    # The ring statistics from sanity run is more accurate
    # Prints sanity check results

    def statisticsPrint(self, data, result):
        dummy = 0
        test = np.asarray(result)
        dummy_indices, frames, ring_count = self.dummy_run(test)
        aromatic_count = self.count_ring_stat(dummy, dummy_indices, result, ring_count, 0)
        molecular_count = self.count_ring_stat(dummy, dummy_indices, result, ring_count, 1)
        plane_count = self.count_ring_stat(dummy, dummy_indices, result, ring_count, 2)

        print("--------------------------------------------------------------------------------------------------")

        print("Total Aromatic Carbon: \t", len(result))
        print("Total Aliphatic Carbon Number: \t", data.shape[0] - len(result))
        print("Total Existing Rings\t", aromatic_count)
        print("Total Planar Rings\t", plane_count)
        print("Total Molecular Rings\t", molecular_count)
        print("Percentage of Aromatic components\t:", len(result) / data.shape[0])
        print("Percentage of Aliphatic components\t:", 1 - len(result) / data.shape[0])

    # Sanity check run function
    # Used in statisticsPrint function
    def dummy_run(self, test):

        print("--------------------------------------------------------------------------------------------------")
        print(str(self.count) + ':\t' + 'Sanity Check for: \t' + self.file)
        print("--------------------------------------------------------------------------------------------------")
        division = self.division_number(test, self.cluster_size)
        split = self.split_data_axis(test, self.axis, division)
        Indices, ring_count, AroCount = self.ring_detection(division, split)
        frames = [Indices]
        dummy_indices = pd.concat(frames).drop_duplicates()

        return dummy_indices, frames, ring_count

    # Calculate the overlapping decomposition statistics.
    def count_ring_stat(self, aroCount, dummy_indices, result, ring_count, kind):
        a = ring_count.copy()
        total = 0
        actual = len(result)
        number = []
        keys = []
        if aroCount != 0:
            multiplier = actual / aroCount
        else:
            multiplier = actual / len(dummy_indices)
        for i in a[kind]:
            keys.append(i)
        for i in a[kind]:
            number.append(int(round(np.ceil((a[kind][i]) * multiplier))))
        count_ring = dict(zip(keys, number))

        return count_ring

    # Print the overlapping decomposition results
    def statistics(self, data, Indices, result, ring_count, aroCount):
        aromatic_count = self.count_ring_stat(aroCount, Indices, result, ring_count, 0)
        molecular_count = self.count_ring_stat(aroCount, Indices, result, ring_count, 1)
        plane_count = self.count_ring_stat(aroCount, Indices, result, ring_count, 2)

        print("--------------------------------------------------------------------------------------------------")

        print("Total Aromatic Carbon: \t", len(result))
        print("Total Aliphatic Carbon Number: \t", data.shape[0] - len(result))
        print("Total Existing Rings\t", aromatic_count)
        print("Total Planar Rings\t", plane_count)
        print("Total Molecular Rings\t", molecular_count)
        print("Percentage of Aromatic components\t:", len(result) / data.shape[0])
        print("Percentage of Aliphatic components\t:", 1 - len(result) / data.shape[0])

    # Split based on distance from origin
    # Not used
    def split_data_origin(self, df, division):
        x = np.array(df)
        n = df.shape[0]
        d = np.zeros(n)

        for c in range(n):
            d[c] = np.sqrt(x[c, 0] * x[c, 0] + x[c, 1] * x[c, 1] + x[c, 2] * x[c, 2])

        main = np.hstack((d.reshape(n, 1), x.reshape(n, 3))).real
        main_sorted_arg = np.argsort(main[:, 0])
        main_sort = np.asarray(main[main_sorted_arg])
        main_sort = np.asarray(main_sort)
        main_sort = np.delete(main_sort, 0, 1)
        split = np.array_split(main_sort, division)
        print("--------------------------------------------------------------------------------------------------")
        print("Starting code for Origin")
        print("--------------------------------------------------------------------------------------------------")
        return split

    # Adaptive split scheme to make division in a given axis
    # Not used
    def split_data_adaptive(self, data, axis, span):
        x = np.array(data)
        n = data.shape[0]
        minimum = np.amin(x, axis=0)[axis]
        maximum = np.amax(x, axis=0)[axis]
        split = []
        division = int(np.ceil(maximum - minimum) / span)
        for i in range(2 * division):
            split.append(x[((minimum + i * span / 2) <= x[:, axis]) & (x[:, axis] <= (minimum + span + i * span / 2))])
        print("--------------------------------------------------------------------------------------------------")
        print("Starting code for Adaptive Run", )
        print("--------------------------------------------------------------------------------------------------")
        return split

    # Adaptive split scheme to make division based on origin
    # Not used
    def split_data_adaptive_origin(self, df, test):
        x = np.array(df)
        n = df.shape[0]
        d = np.zeros(n)

        split = []
        a = []

        span = 10

        for c in range(n):
            d[c] = np.sqrt(x[c, 0] * x[c, 0] + x[c, 1] * x[c, 1] + x[c, 2] * x[c, 2])

        main = np.hstack((d.reshape(n, 1), x.reshape(n, 3))).real
        main_sorted_arg = np.argsort(main[:, 0])
        main_sort = np.asarray(main[main_sorted_arg])
        main_sort = np.asarray(main_sort)
        x = main_sort

        minimum_x = np.amin(x, axis=0)[1]
        maximum_x = np.amax(x, axis=0)[1]
        minimum_y = np.amin(x, axis=0)[2]
        maximum_y = np.amax(x, axis=0)[2]
        minimum_z = np.amin(x, axis=0)[3]
        maximum_z = np.amax(x, axis=0)[3]
        span_x = maximum_x - minimum_x
        span_y = maximum_y - minimum_y
        span_z = maximum_z - minimum_z

        division_x = int(np.ceil(span_x / span))
        division_y = int(np.ceil(span_y / span))
        division_z = int(np.ceil(span_z / span))

        minimum = np.amin(main_sort, axis=0)[0]
        maximum = np.amax(main_sort, axis=0)[0]
        division = int(np.ceil(maximum - minimum) / span)

        for i in range(2 * division_x):
            for j in range(2 * division_y):
                for k in range(2 * division):
                    for l in range(2 * division_z):
                        split.append(x[((minimum_x + i * span / 2) <= x[:, 1]) & (
                                x[:, 1] <= (minimum_x + span + i * span / 2)) & (
                                               (minimum_y + j * span / 2) <= x[:, 2]) & (
                                               x[:, 2] <= (minimum_y + span + j * span / 2)) & (
                                               (minimum + l * test / 2) <= x[:, 0]) & (
                                               x[:, 0] <= (minimum + span + l * test / 2)) & (
                                               (minimum_z + k * span / 2) <= x[:, 2]) & (
                                               x[:, 3] <= (minimum_z + span + k * span / 2))])

        for i in range(len(split)):
            a.append((np.delete(split[i], 0, 1)))

        better_split = []

        for i in range(len(a)):
            if len(a[i]) > 4:
                better_split.append(a[i])

        division = len(better_split)

        print("--------------------------------------------------------------------------------------------------")
        print("Starting code for Origin Adaptive Run", )
        print("--------------------------------------------------------------------------------------------------")
        return better_split, division

    # Adaptive split scheme to cubic decomposition
    # This one is used to split input domain
    # Returns split array of the input domain
    def split_data_adaptive_cubic(self, data):
        x = np.array(data)
        n = data.shape[0]

        split = []

        minimum_x = np.amin(x, axis=0)[0]
        maximum_x = np.amax(x, axis=0)[0]
        minimum_y = np.amin(x, axis=0)[1]
        maximum_y = np.amax(x, axis=0)[1]
        minimum_z = np.amin(x, axis=0)[2]
        maximum_z = np.amax(x, axis=0)[2]
        span_x = maximum_x - minimum_x
        span_y = maximum_y - minimum_y
        span_z = maximum_z - minimum_z

        division_x = int(np.ceil(span_x / self.span))
        division_y = int(np.ceil(span_y / self.span))
        division_z = int(np.ceil(span_z / self.span))

        for i in range(2 * division_x):
            for j in range(2 * division_y):
                for k in range(2 * division_z):
                    split.append(x[((minimum_x + i * self.span / 2) <= x[:, 0]) & (
                            x[:, 0] <= (minimum_x + self.span + i * self.span / 2)) & (
                                           (minimum_y + j * self.span / 2) <= x[:, 1]) & (
                                           x[:, 1] <= (minimum_y + self.span + j * self.span / 2)) & (
                                           (minimum_z + k * self.span / 2) <= x[:, 2]) & (
                                           x[:, 2] <= (minimum_z + self.span + k * self.span / 2))])

        better_split = []
        for i in range(len(split)):
            if len(split[i]) > 4:
                better_split.append(split[i])
        # better_split rejects the domain containing less than 5 carbon atoms
        division = len(better_split)

        print("--------------------------------------------------------------------------------------------------")
        print("Starting code for Adaptive Run", )
        print("--------------------------------------------------------------------------------------------------")
        return better_split, division

    # Split scheme to make division in a given axis
    # Used for sanity check
    # Returns the split domains of initial results
    def split_data_axis(self, df, axis, division):
        x = np.array(df)
        n = df.shape[0]

        main_sorted_arg = np.argsort(x[:, axis])
        main_sort = np.asarray(x[main_sorted_arg])
        splits = np.array_split(main_sort, division)
        print("--------------------------------------------------------------------------------------------------")
        print("Starting code for Axis: \t", axis)
        print("--------------------------------------------------------------------------------------------------")
        return splits

    # For plotting the results
    def plot_domain(self, result):
        test = np.asarray(result)

        # Plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        x = test[:, 0]
        y = test[:, 1]
        z = test[:, 2]
        ax1.scatter(x, y, z, alpha=0.7, cmap='rainbow', marker='o', c='b')

        plt.show()

    # Actual ring detection is done in this function
    def ring_detection(self, division, split):
        AroCount = 0
        Actual_Ring_count = {}
        Molecular_Ring_count = {}
        Total_Planar_Ring_count = {}
        cord = []
        Indices = []

        for runs in range(division):
            x = split[runs]
            n = split[runs].shape[0]
            d = np.zeros(n)

            for c in range(n):
                d[c] = np.sqrt(x[c, 0] * x[c, 0] + x[c, 1] * x[c, 1] + x[c, 2] * x[c, 2])
            dist = np.hstack((d.reshape(n, 1), x.reshape(n, 3))).real
            dist_arg = np.argsort(dist[:, 0])
            distance = np.asarray(dist[dist_arg])
            distance = np.delete(distance, 0, 1)
            # Create the distance matrix
            graph = np.logical_and(distance_matrix(distance[:, 0:], distance[:, 0:]) <= self.bond_distance_upper,
                                   distance_matrix(distance[:, 0:], distance[:, 0:]) >= self.bond_distance_lower)
            np.fill_diagonal(graph, False)
            # Using networkx module to extract all the cycles using the distance matrix
            G = nx.from_numpy_matrix(graph, create_using=nx.DiGraph)
            all_cycles = list(nx.simple_cycles(G))

            cycles = []

            for cycle in all_cycles:
                if 32 > len(cycle) > 4 and not self.checkDuplicate(cycle, cycles):
                    cycles.append(cycle)

            # Assigning the detected rings in different buckets based on carbon count
            molecular_array = []
            ring_array = []

            i = 0
            for r in cycles:
                if (len(r)) < 8:
                    ring_array.append(r)
                    i += 1
                    continue
                if i > 0:
                    prev_item = cycles[i - 1]

                    if self.isSubSet(r, prev_item):
                        molecular_array.append(r)
                        i += 1
                        continue

                if i < len(cycles) - 1:
                    next_item = cycles[i + 1]

                    if self.isSubSet(r, next_item):
                        molecular_array.append(r)
                        i += 1
                        continue
                #             ring_array.append(r)
                i += 1

            a = {}
            b = {}

            for q in ring_array:
                if len(q) in a:
                    a[len(q)] = a[len(q)] + 1
                else:
                    a[len(q)] = 1
                if len(q) > 1:
                    if len(q) in Actual_Ring_count:
                        Actual_Ring_count[len(q)] = Actual_Ring_count[len(q)] + 1
                    else:
                        Actual_Ring_count[len(q)] = 1

            for l in molecular_array:
                if len(l) in b:
                    b[len(l)] = b[len(l)] + 1
                else:
                    b[len(l)] = 1
                if len(l) > 1:
                    if len(l) in Molecular_Ring_count:
                        Molecular_Ring_count[len(l)] = Molecular_Ring_count[len(l)] + 1
                    else:
                        Molecular_Ring_count[len(l)] = 1

            planar_ring = []
            plane = False
            for rings in ring_array:
                space = []
                for index in rings:
                    test = distance[index]
                    space.append(test)
                points = pd.DataFrame(np.asarray(space))
                points = np.asarray(points)
                for i in range(len(points) - 3):
                    plane = self.planar_check(points[i][0], points[i][1], points[i][2], points[i + 1][0],
                                              points[i + 1][1],
                                              points[i + 1][2], points[i + 2][0], points[i + 2][1], points[i + 2][2],
                                              points[i + 3][0], points[i + 3][1], points[i + 3][2])
                    if not plane:
                        break
                if plane:
                    planar_ring.append(rings)

            # index list
            if var2.get():
                for rings in planar_ring:
                    for index in rings:
                        coordinates = distance[index]
                        cord.append(np.asarray(coordinates))
                    Indices = pd.DataFrame(np.asarray(cord))
            else:
                for rings in ring_array:
                    for index in rings:
                        coordinates = distance[index]
                        cord.append(np.asarray(coordinates))
                    Indices = pd.DataFrame(np.asarray(cord))
            # index list

            r = {}
            for p in planar_ring:
                if len(p) in r:
                    r[len(p)] = r[len(p)] + 1
                else:
                    r[len(p)] = 1
                if len(p) > 1:
                    if len(p) in Total_Planar_Ring_count:
                        Total_Planar_Ring_count[len(p)] = Total_Planar_Ring_count[len(p)] + 1
                    else:
                        Total_Planar_Ring_count[len(p)] = 1

            ARnumber = 0
            ALnumber = 0

            #         print("Batch No. \t",runs)
            #         print("--------------------------------------------------------------------------------------------------")
            if len(cycles) != 0:
                if np.asarray(ring_array).shape[0] > 0:
                    ARnumber = np.unique(np.hstack(ring_array)).shape[0]
                ALnumber = split[runs].shape[0] - ARnumber
                PercentageAromatic = ARnumber / (ARnumber + ALnumber)
                PercentageAliphatic = ALnumber / (ARnumber + ALnumber)
            #             print("Existing Rings\t",a)
            #             print("Molecular Rings\t",b)
            #             print("Planar Rings\t",r)
            #             print("Number of Aromatic Carbon Atom\t:",ARnumber)
            #             print("Number of Aliphatic Carbon Atom\t:",ALnumber)
            #             print("Percentage of Aromatic components\t:",PercentageAromatic)
            #             print("Percentage of Aliphatic components\t:",PercentageAliphatic)
            else:
                #             print("No Ring Found")
                continue

            #     #Plot
            #         fig = plt.figure()
            #         ax1 = fig.add_subplot(111,projection='3d')
            #         ax1.set_xlabel("X")
            #         ax1.set_ylabel("Y")
            #         ax1.set_zlabel("Z")
            #         # x, y, z = np.loadtxt('test.csv', delimiter=',',unpack=True)
            #         x=split[runs][:,0]
            #         y=split[runs][:,1]
            #         z=split[runs][:,2]
            #         ax1.scatter(x,y,z, alpha=0.7, cmap='rainbow', marker='o',c='b')

            #         plt.show()

            AroCount += ARnumber

        # Whether to impose planarity argument for the calculation
        if var2.get():
            ring_count = [Total_Planar_Ring_count, Molecular_Ring_count, Total_Planar_Ring_count]
        else:
            ring_count = [Actual_Ring_count, Molecular_Ring_count, Total_Planar_Ring_count]

        #     print("--------------------------------------------------------------------------------------------------")

        #     print("Total Aromatic Carbon: \t",AroCount)
        #     print("Total Aliphatic Carbon Number: \t",df.shape[0]-AroCount)
        #     print("Total Existing Rings\t",Actual_Ring_count)
        #     print("Total Planar Rings\t",Total_Planar_Ring_count)
        #     print("Total Molecular Rings\t",Molecular_Ring_count)
        #     print("Percentage of Aromatic components\t:",AroCount/df.shape[0])
        #     print("Percentage of Aliphatic components\t:",1-AroCount/df.shape[0])

        #     print("--------------------------------------------------------------------------------------------------")
        #     print("Total Rings Found For This Run")
        #     print("--------------------------------------------------------------------------------------------------")
        ExistingRingCoOrdinate = np.asarray(Indices)

        # #Plot
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(111,projection='3d')
        #     ax1.set_xlabel("X")
        #     ax1.set_ylabel("Y")
        #     ax1.set_zlabel("Z")
        # # x, y, z = np.loadtxt('test.csv', delimiter=',',unpack=True)
        #     x=ExistingRingCoOrdinate[:,0]
        #     y=ExistingRingCoOrdinate[:,1]
        #     z=ExistingRingCoOrdinate[:,2]
        #     ax1.scatter(x,y,z, alpha=0.7, cmap='rainbow', marker='o',c='b')

        #     plt.show()

        return Indices, ring_count, AroCount

    # Putting it all together for FindRing class
    def main(self):
        try:
            data = self.get_data()
        except:
            wrongData = Tk()
            wrongData.withdraw()
            messagebox.showerror("Error", "Something is wrong with the datafile")
            return None
        split, division = self.split_data_adaptive_cubic(data)
        Indices, ring_count, AroCount = self.ring_detection(division, split)
        frames = [Indices]
        result = self.write_to_xyz(data, frames)
        self.statistics(data, Indices, result, ring_count, AroCount)
        if self.sanity:
            self.statisticsPrint(data, result)
        self.result = result
        self.original = self.get_data()


# For using a configuration file in stead of the GUI
# A configuration.txt file can be used to read the input parameters.

def read_file():
    f = open("configuration.txt", "r")
    x = {}
    if f.mode == 'r':
        f1 = f.readlines()
        for p in f1:
            if len(p.split('=')) == 2:
                x[p.split('=')[0].strip()] = p.split('=')[1]
    f.close()

    keys = ['bond_distance_lower', 'bond_distance_upper', 'cluster_size', 'span', 'axis', 'fileOut', 'input_files',
            'separator', 'extension', 'Planar Allowance', 'Identifier', 'Planarity']
    for i in keys:
        if i not in x:
            print(i + " not found in configuration file")
            return None

    try:
        bond_distance_upper = float(x['bond_distance_upper'])
        bond_distance_lower = float(x['bond_distance_lower'])
        cluster_size = int(x['cluster_size'])
        span = int(x['span'])
        axis = int(x['axis'])
        fileOut = str(x['fileOut']).replace('\n', '')
        split = str(x['input_files']).replace('\n', '')
        if split == 'space':
            split = ' '
        sep = str(x['separator']).replace('\n', '')
        ext = str(x['extension']).replace('\n', '')
        planarAllowance = float(x['Planar Allowance'])
        e11 = str(x['Identifier']).replace('\n', '')
        var2 = int(x['Planarity'])

        return FindRing(bond_distance_upper, bond_distance_lower, cluster_size, span, axis, file_name,
                        fileOut), split, ext, sep, planarAllowance


    except:
        print("something went wrong. wrong data format in configuration file.")
        return None


# Plot the rings from the results

def plot_domain_gui(result, original, file):
    test = np.asarray(result)
    orig = np.asarray(original)
    plots: Tk = Tk()
    plots.wm_title("Ring Locations in the Cluster:\t" + file)

    fig = Figure(figsize=(10, 10), dpi=100)

    canvas = FigureCanvasTkAgg(fig, master=plots)
    canvas.draw()

    ax1 = fig.add_subplot(111, projection='3d')

    # Hide grid lines
    ax1.grid(False)

    # Hide axes ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])

    ax1.axis('off')

    x = orig[:, 0]
    y = orig[:, 1]
    z = orig[:, 2]

    # for i in range(len(orig)):
    #     for j in range(len(orig)):
    #         if 2 >= np.linalg.norm(orig[i] - orig[j]) >= 1.2:
    #             p = [orig[i, 0], orig[j, 0]]
    #             q = [orig[i, 1], orig[j, 1]]
    #             r = [orig[i, 2], orig[j, 2]]
    #             ax1.plot(p, q, r, color='r', marker='o', linewidth=2)

    x = test[:, 0]
    y = test[:, 1]
    z = test[:, 2]
    for i in range(len(test)):
        for j in range(len(test)):
            if 2 >= np.linalg.norm(test[i] - test[j]) >= 1.2:
                p = [test[i, 0], test[j, 0]]
                q = [test[i, 1], test[j, 1]]
                r = [test[i, 2], test[j, 2]]
                ax1.plot(p, q, r, color='b', marker='o', linewidth=2)

    toolbar = NavigationToolbar2Tk(canvas, plots)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    canvas.flush_events()
    # plots.mainloop()


# Plot the chemical bond information from SMILEs string
def MolToMPL(mol, size=(300, 300), kekulize=True, wedgeBonds=True, imageType=None, fitImage=False,
             options=None, **kwargs):
    if not mol:
        raise ValueError('Null molecule provided')
    from rdkit.Chem.Draw.mplCanvas import Canvas
    canvas = Canvas(size)
    if options is None:
        options = DrawingOptions()
        options.bgColor = None
    if fitImage:
        options.dotsPerAngstrom = int(min(size) / 10)
        options.atomLabelFontSize = 7
        options.bondLineWidth = 2
        options.coordScale = 1
    options.wedgeDashedBonds = wedgeBonds
    drawer = MolDrawing(canvas=canvas, drawingOptions=options)

    omol = mol
    if kekulize:
        from rdkit import Chem
        mol = Chem.Mol(mol.ToBinary())
        Chem.Kekulize(mol)

    if not mol.GetNumConformers():
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)

    drawer.AddMol(mol, **kwargs)
    omol._atomPs = drawer.atomPs[mol]
    for k, v in omol._atomPs.items():
        omol._atomPs[k] = canvas.rescalePt(v)
    canvas._figure.set_size_inches(float(size[0]) / 100, float(size[1]) / 100)
    return canvas._figure


# The main function for the GUI
def RunAnalysis():
    try:
        bond_distance_upper = float(e2.get())
        bond_distance_lower = float(e1.get())
        cluster_size = int(e3.get())
        span = float(e4.get())
        axis = int(axisin.current())
        split = (e6.get())
        fileOut = (e7.get())
        sep = (e8.get())
        ext = (e9.get())
        planarAllowance = float(e10.get())
    except:
        error: Tk = Tk()
        error.withdraw()
        messagebox.showerror("Error", "Wrong Set of Input Parameters")
        return None

    def listdir_nohidden(path):
        for f in sorted(os.listdir(path)):
            if not f.startswith(b'.') and os.fsdecode(f).endswith(ext):
                yield f

    output = Tk()
    output.title("Ring Detection Log")
    t1 = Text(output, height=33, width=100, bg='white', fg='black', bd=4, font=("Courier", 12))

    class PrintToT1(object):
        def write(self, s):
            t1.insert(END, s)
            f = open(FR.fileOut + '/' + e11.get() + '.txt', 'a')
            f.write(s)
            f.close()
            f1 = open(FR.fileOut + '/' + e11.get() + '_' + FR.file.split('.')[0] + '.txt', 'a')
            f1.write(s)
            f1.close()

        def flush(self):
            pass

    if sep == 'space':
        sep = ' '
    directory = os.fsencode(split)
    count = 1
    for file in listdir_nohidden(directory):
        fileName = os.fsdecode(file)
        file_name = split + '/' + os.fsdecode(file)
        FR = FindRing(bond_distance_upper, bond_distance_lower, cluster_size, span, axis, file_name, fileOut)
        FR.file = fileName
        FR.sep = sep
        FR.planeAllowance = planarAllowance
        FR.count = count

        if var1.get():
            FR.sanity = 1
        else:
            FR.sanity = 0

        sys.stdout = PrintToT1()
        print('--------------------------------------------------------------------------------------------------')
        print(str(count) + ':\t' + os.fsdecode(file))
        FR.main()

        if var3.get():
            plot_domain_gui(FR.result, FR.original, FR.file)
        # using xyz2mol to calculate chemical information
        if do_smiles.get():
            print('--------------------------------------------------------------------------------------------------')
            print(str(count) + ':\t' + format.get() + '_' + os.fsdecode(file).split('.')[0] + '\t: START')
            print('--------------------------------------------------------------------------------------------------')
            smile_file = FR.fileOut + '/' + e11.get() + '_' + FR.file
            smiles = xyz2mol.main(smile_file, ignore_charge.get(), None, ignore_chiral.get(), use_Huckel.get(),
                                  format.get(),
                                  int(charge.get()))
            DrawingOptions.atomLabelFontSize = 55
            DrawingOptions.dotsPerAngstrom = 100
            DrawingOptions.bondLineWidth = 3.0

            mol = Chem.MolFromSmiles(smiles)
            # mol = Chem.rdmolops.AddHs(mol)
            if show_molecule.get() and do_smiles.get():
                MolToMPL(mol, size=(800, 800), kekulize=True, wedgeBonds=True, fitImage=True)

            Draw.MolToFile(mol, FR.fileOut + '/' + e11.get() + '_' + os.fsdecode(file).split('.')[0] + '.png',
                           useSVG=True,
                           size=(1500, 1500), fitImage=True)

            print('--------------------------------------------------------------------------------------------------')
            print(str(count) + ':\t' + format.get() + '_' + os.fsdecode(file).split('.')[0] + '\t: END')
            print('--------------------------------------------------------------------------------------------------')

        count = count + 1
    print('--------------------------------------------------------------------------------------------------')

    t1.configure(state='disabled')
    t1.pack()

    output.mainloop()


def browsefunc():
    e6.delete(0, tk.END)
    filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("xyz files", "*.xyz"), ("all files", "*.*")))
    e6.insert(tk.END, filename)


def browseDir_e7():
    e7.delete(0, tk.END)
    filename = filedialog.askdirectory()
    e7.insert(tk.END, filename)


def browseDir_e6():
    e6.delete(0, tk.END)
    filename = filedialog.askdirectory()
    e6.insert(tk.END, filename)


master: Tk = Tk()

master.title("Parameter Selection")
Label(master,
      text="Bond Distance (Lower)").grid(row=0)
Label(master,
      text="Bond Distance (Upper)").grid(row=1)
Label(master,
      text="Cluster Size (for Sanity Check)").grid(row=2)
Label(master,
      text="Span").grid(row=3)
Label(master,
      text="Axis (for Sanity Check)").grid(row=4)

Label(master,
      text="Separator").grid(row=7)
Label(master,
      text="Extension").grid(row=8)
Label(master,
      text="Input Files").grid(row=9)
Label(master,
      text="Output Files ").grid(row=10)
Label(master,
      text="(e.g. 100)").grid(row=2, column=2)
Label(master,
      text="(e.g. 10)").grid(row=3, column=2)
Label(master,
      text='(e.g. \\t,comma(,),space,:, etc.)').grid(row=7, column=2)
Label(master,
      text="(e.g. .xyz, .csv etc.)").grid(row=8, column=2)
Label(master,
      text="Plane Allowance (in degree)").grid(row=5)
Label(master,
      text="(e.g. 1, 2 , 3 etc.)").grid(row=5, column=2)
Label(master,
      text="Identifier").grid(row=11, column=0)
Label(master,
      text="(Prefix for outputs)").grid(row=11, column=2)

n = tk.StringVar()
axisin = ttk.Combobox(master, width=19, textvariable=n)
axisin['values'] = ('X',
                    'Y',
                    'Z')
axisin.grid(column=1, row=4)
axisin.current(0)

v1 = StringVar(master, value='1.2')
v2 = StringVar(master, value='2')
v3 = StringVar(master, value='100')
v4 = StringVar(master, value='10')
v6 = StringVar(master,
               value='/Users/khaledmosharrafmukut/tweaks/MDRING/pyCharm/input')
v7 = StringVar(master,
               value='/Users/khaledmosharrafmukut/tweaks/MDRING/pyCharm/output')
v8 = StringVar(master, value='space')
v9 = StringVar(master, value='.xyz')
v10 = StringVar(master, value='2')
v11 = StringVar(master, value='Result')

e1 = Entry(master, textvariable=v1)
e2 = Entry(master, textvariable=v2)
e3 = Entry(master, textvariable=v3)
e4 = Entry(master, textvariable=v4)
e6 = Entry(master, textvariable=v6)
e7 = Entry(master, textvariable=v7)
e8 = Entry(master, textvariable=v8)
e9 = Entry(master, textvariable=v9)
e10 = Entry(master, textvariable=v10)
e11 = Entry(master, textvariable=v11)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e8.grid(row=7, column=1)
e9.grid(row=8, column=1)
e10.grid(row=5, column=1)
e6.grid(row=9, column=1)
e7.grid(row=10, column=1)
e11.grid(row=11, column=1)

var1 = IntVar()
ttk.Checkbutton(master, text="Sanity Check", variable=var1).grid(row=6, column=1, sticky='n')
var3 = IntVar()
ttk.Checkbutton(master, text="plot", variable=var3).grid(row=6, column=2, sticky='n')
var2 = IntVar()
ttk.Checkbutton(master, text="impose planarity", variable=var2).grid(row=6, column=0, sticky='n')

b1 = ttk.Button(master, text="input directory", command=browseDir_e6)
b1.grid(row=9, column=2)

b2 = ttk.Button(master, text="output directory", command=browseDir_e7)
b2.grid(row=10, column=2)

Label(master,
      borderwidth=3, relief="ridge", background='wheat1', text="SMILES string and Visualization Parameters").grid(
    row=13, column=1)

ignore_charge = IntVar()
ttk.Checkbutton(master, text="ignore charge", variable=ignore_charge).grid(row=15, column=0, sticky='n')
ignore_chiral = IntVar()
ttk.Checkbutton(master, text="ignore chirality", variable=ignore_chiral).grid(row=15, column=1, sticky='n')
use_Huckel = IntVar()
ttk.Checkbutton(master, text="use Huckel connectivity", variable=use_Huckel).grid(row=15, column=2, sticky='n')
Label(master, text="Charge").grid(row=14, column=1, sticky='e')
v12 = StringVar(master, value='0')
charge = Entry(master, textvariable=v12)
charge.grid(row=14, column=2)
do_smiles = IntVar()
ttk.Checkbutton(master, text="CHEM calculations", variable=do_smiles).grid(row=14, column=0, sticky='n')

show_molecule = IntVar()
ttk.Checkbutton(master, text="show molecule", variable=show_molecule).grid(row=13, column=2, sticky='n')

format = tk.StringVar()
format = ttk.Combobox(master, width=14, textvariable=format)
format['values'] = ('sdf',
                    'smiles')
format.grid(column=1, row=14)
format.current(1)

ttk.Button(master,
           text='Run',
           command=RunAnalysis).grid(row=16,
                                     column=0,
                                     sticky=W,
                                     pady=4)
ttk.Button(master,
           text='Quit',
           command=master.destroy).grid(row=16,
                                        column=2,
                                        sticky=W,
                                        pady=4)

master.mainloop()
