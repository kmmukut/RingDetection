import tkinter as tk
from tkinter import *
from tkinter import filedialog, Tk
from tkinter import messagebox
from tkinter import ttk

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import distance_matrix


class FindRing:
    def __init__(self, bond_distance_upper, bond_distance_lower, cluster_size, span, axis, file_name):
        self.bond_distance_upper = bond_distance_upper
        self.bond_distance_lower = bond_distance_lower
        self.cluster_size = cluster_size
        self.span = span
        self.axis = axis
        self.file_name = file_name
        # For GUI coding:
        self.result = None
        self.fileOut = (e7.get())

    def get_data(self):
        df = pd.io.parsers.read_csv(
            #     filepath_or_buffer='points',
            filepath_or_buffer=self.file_name,
            #     filepath_or_buffer='test.csv',
            header=None,
            sep='\t',
            #     sep=',',
            skiprows=2
        )
        if df.shape[1] == 4:
            df = df[~df[0].astype(str).str.startswith('2')]
        df = df[df.columns[-3:]]
        df.to_csv(self.fileOut + '/' + str((df.shape[0])) + '_input.csv', index=None)
        return df

    def write_to_xyz(self, data, frames):
        result = pd.concat(frames)

        result = result.drop_duplicates()
        result = result.reset_index(drop=True)
        result.to_csv(self.fileOut + '/' + str(len(result)) + '_RingIndices.csv', index=None, header=None)

        XYZ = result

        add_number = pd.DataFrame({len(result)}, index=[0])
        declare_TimeStep = pd.DataFrame({'Atoms. Timestep: 1200000'}, index=[1])
        XYZ['particle_type'] = 3
        XYZ = pd.concat([declare_TimeStep, XYZ.iloc[:]]).reset_index(drop=True)
        XYZ = pd.concat([add_number, XYZ.iloc[:]]).reset_index(drop=True)
        XYZ = XYZ[['particle_type', 0, 1, 2]]
        XYZ.to_csv(self.fileOut + '/' + 'RingIndices_' + str(len(result)) + '.xyz', index=None, header=None, sep='\t')
        result = result[[0, 1, 2]]

        for i in range(3):
            result[3 - i] = result[2 - i]
        result = result.reindex(columns=data.columns)

        return result

    def division_adaptive(self, df, span, axis):
        x = np.array(df)
        n = df.shape[0]
        minimum = np.amin(x, axis=0)[axis]
        maximum = np.amax(x, axis=0)[axis]
        division = int(np.ceil(maximum - minimum) / span)
        return division

    def division_number(self, df, cluster_size):
        n = df.shape[0]
        division = int(np.ceil(n / cluster_size))

        return division

    def checkDuplicate(self, neddleCycle, cycles):
        neddleCycle = np.asarray(neddleCycle)
        cycles = np.asarray(cycles)

        for cycle in cycles:
            if len(cycle) != neddleCycle.shape[0]:
                continue
            if np.all(np.isin(neddleCycle, cycle)):
                return True
        return False

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

        # if (abs(a * x + b * y + c * z + d) <= 0.5):
        #     return 0
        # else:
        #     return 1

        return abs(a * x + b * y + c * z + d) <= 1

    def isSubSet(self, current, toCompare):
        return np.in1d(toCompare, current).all()

    def statisticsPrint(self, data, result):
        dummy = 0
        test = np.asarray(result)
        dummy_indices, frames, ring_count = self.dummy_run(test)
        aromatic_count = self.count_ring_stat(dummy, dummy_indices, result, ring_count, 0)
        molecular_count = self.count_ring_stat(dummy, dummy_indices, result, ring_count, 1)
        plane_count = self.count_ring_stat(dummy, dummy_indices, result, ring_count, 2)

        print("--------------------------------------------------------------------------------------------------")

        print("Total Aromatic Carbon: \t", len(result))
        print("Total Alephatic Carbon Number: \t", data.shape[0] - len(result))
        print("Total Existing Rings\t", aromatic_count)
        print("Total Planar Rings\t", plane_count)
        print("Total Molecular Rings\t", molecular_count)
        print("Percentage of Aromatic components\t:", len(result) / data.shape[0])
        print("Percentage of Alephatic components\t:", 1 - len(result) / data.shape[0])

    def dummy_run(self, test):

        print("--------------------------------------------------------------------------------------------------")
        print("Sanity Check:")
        print("--------------------------------------------------------------------------------------------------")
        division = self.division_number(test, self.cluster_size)
        split = self.split_data_axis(test, self.axis, division)
        Indices, ring_count, AroCount = self.ring_detection(test, division, split)
        frames = [Indices]
        dummy_indices = pd.concat(frames).drop_duplicates()

        return dummy_indices, frames, ring_count

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

    def statistics(self, data, Indices, result, ring_count, aroCount):
        aromatic_count = self.count_ring_stat(aroCount, Indices, result, ring_count, 0)
        molecular_count = self.count_ring_stat(aroCount, Indices, result, ring_count, 1)
        plane_count = self.count_ring_stat(aroCount, Indices, result, ring_count, 2)

        print("--------------------------------------------------------------------------------------------------")

        print("Total Aromatic Carbon: \t", len(result))
        print("Total Alephatic Carbon Number: \t", data.shape[0] - len(result))
        print("Total Existing Rings\t", aromatic_count)
        print("Total Planar Rings\t", plane_count)
        print("Total Molecular Rings\t", molecular_count)
        print("Percentage of Aromatic components\t:", len(result) / data.shape[0])
        print("Percentage of Alephatic components\t:", 1 - len(result) / data.shape[0])

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

        division = len(better_split)

        print("--------------------------------------------------------------------------------------------------")
        print("Starting code for Adaptive Run", )
        print("--------------------------------------------------------------------------------------------------")
        return better_split, division

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

    def ring_detection(self, df, division, split):
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

            graph = np.logical_and(distance_matrix(distance[:, 0:], distance[:, 0:]) <= self.bond_distance_upper,
                                   distance_matrix(distance[:, 0:], distance[:, 0:]) >= self.bond_distance_lower)
            np.fill_diagonal(graph, False)
            G = nx.from_numpy_matrix(graph, create_using=nx.DiGraph)
            all_cycles = list(nx.simple_cycles(G))

            cycles = []

            for cycle in all_cycles:
                if 32 > len(cycle) > 4 and not self.checkDuplicate(cycle, cycles):
                    cycles.append(cycle)

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
                PercentageAlephatic = ALnumber / (ARnumber + ALnumber)
            #             print("Existing Rings\t",a)
            #             print("Molecular Rings\t",b)
            #             print("Planar Rings\t",r)
            #             print("Number of Aromatic Carbon Atom\t:",ARnumber)
            #             print("Number of Alephatic Carbon Atom\t:",ALnumber)
            #             print("Percentage of Aromatic components\t:",PercentageAromatic)
            #             print("Percentage of Alephatic components\t:",PercentageAlephatic)
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

        ring_count = [Actual_Ring_count, Molecular_Ring_count, Total_Planar_Ring_count]

        #     print("--------------------------------------------------------------------------------------------------")

        #     print("Total Aromatic Carbon: \t",AroCount)
        #     print("Total Alephatic Carbon Number: \t",df.shape[0]-AroCount)
        #     print("Total Existing Rings\t",Actual_Ring_count)
        #     print("Total Planar Rings\t",Total_Planar_Ring_count)
        #     print("Total Molecular Rings\t",Molecular_Ring_count)
        #     print("Percentage of Aromatic components\t:",AroCount/df.shape[0])
        #     print("Percentage of Alephatic components\t:",1-AroCount/df.shape[0])

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

    def main(self):
        try:
            data = self.get_data()
        except:
            wrongData = Tk()
            wrongData.withdraw()
            messagebox.showerror("Error", "Something is wrong with the datafile")
            return None

        split, division = self.split_data_adaptive_cubic(data)
        Indices, ring_count, AroCount = self.ring_detection(data, division, split)
        frames = [Indices]
        result = self.write_to_xyz(data, frames)
        self.statistics(data, Indices, result, ring_count, AroCount)
        #         self.plot_domain(result)
        self.statisticsPrint(data, result)
        # For GUI coding:
        self.result = result


def read_file():
    f = open("configuration.txt", "r")
    x = {}
    if f.mode == 'r':
        f1 = f.readlines()
        for p in f1:
            if len(p.split('=')) == 2:
                x[p.split('=')[0].strip()] = p.split('=')[1]
    f.close()

    keys = ['bond_distance_lower', 'bond_distance_upper', 'cluster_size', 'span', 'axis', 'file_name']
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
        file_name = str(x['file_name']).replace('\n', '')
        return FindRing(bond_distance_upper, bond_distance_lower, cluster_size, span, axis, file_name)
    except:
        print("something went wrong. wrong data format in configuration file.")
        return None


# FR = read_file()
# if FR is not None:
#     FR.main()


def show_entry_fields():
    print(
        "Bond Distance (Lower): %s\nBond Distance (Upper): %s\nCluster Size: %s\nSpan: %s\nAxis: %s\nFile Name: %s" % (
            e1.get(), e2.get(), e3.get(), e4.get(), e5.get(), e6.get()))


def plot_domain_gui(result):
    test = np.asarray(result)
    plots = Tk()
    plots.title("Ring Locations in the Cluster")

    # Plot
    fig = plt.figure(figsize=(6, 8), dpi=100)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    x = test[:, 0]
    y = test[:, 1]
    z = test[:, 2]
    ax1.scatter(x, y, z, alpha=0.7, cmap='rainbow', marker='o', c='b')

    scatter = FigureCanvasTkAgg(fig, plots)
    scatter.get_tk_widget().pack()
    ttk.Button(plots,
               text='close',
               command=plots.destroy).pack()


#     plots.mainloop()

#     plots.mainloop()


def RunAnalysis():
    try:
        bond_distance_upper = float(e2.get())
        bond_distance_lower = float(e1.get())
        cluster_size = int(e3.get())
        span = float(e4.get())
        axis = int(e5.get())
        file_name = (e6.get())
    except:
        error = Tk()
        error.withdraw()
        messagebox.showerror("Error", "Wrong Set of Input Parameters")
        return None

    FR = FindRing(bond_distance_upper, bond_distance_lower, cluster_size, span, axis, file_name)

    output = Tk()
    output.title("Ring Detection Log")

    t1 = Text(output, height=33, width=100, bg='white', fg='black', bd=4, font=("Courier", 12))

    class PrintToT1(object):
        def writing(self, s):
            t1.insert(END, s)
            f = open(FR.fileOut + '/' + str((FR.get_data().shape[0])) + "_results.txt", "a")
            f.write(s)
            f.close()

    sys.stdout = PrintToT1()

    print("Results for dataset: ", file_name)
    print("--------------------------------------------------------------------------------------------------")

    FR.main()

    t1.configure(state='disabled')
    t1.pack()
    plot_domain_gui(FR.result)


#     output.mainloop()


def browsefunc():
    e6.delete(0, tk.END)
    filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("xyz files", "*.xyz"), ("all files", "*.*")))
    e6.insert(tk.END, filename)


def browseDir():
    e7.delete(0, tk.END)
    filename = filedialog.askdirectory()
    e7.insert(tk.END, filename)


master: Tk = Tk()
master.title("Parameter Selection")
Label(master,
      text="Bond Distance (Lower)").grid(row=0)
Label(master,
      text="Bond Distance (Upper)").grid(row=1)
Label(master,
      text="Cluster Size").grid(row=2)
Label(master,
      text="Span").grid(row=3)
Label(master,
      text="Axis").grid(row=4)
Label(master,
      text="File Name").grid(row=5)
Label(master,
      text="Save file location: ").grid(row=6)
Label(master,
      text="(e.g. 100)").grid(row=2, column=2)
Label(master,
      text="(e.g. 10)").grid(row=3, column=2)
Label(master,
      text="(e.g. 1 or 2 or 3)").grid(row=4, column=2)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)
e6.grid(row=5, column=1)
e7.grid(row=6, column=1)

b1 = ttk.Button(master, text="select file", command=browsefunc)
b1.grid(row=5, column=2)

b2 = ttk.Button(master, text="output directory", command=browseDir)
b2.grid(row=6, column=2)

ttk.Button(master,
           text='Run',
           command=RunAnalysis).grid(row=7,
                                     column=0,
                                     sticky=W,
                                     pady=4)
ttk.Button(master,
           text='Quit',
           command=master.destroy).grid(row=7,
                                        column=1,
                                        sticky=W,
                                        pady=4)
master.mainloop()
