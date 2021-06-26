import math
from statistics import mean
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_kfcv_file(filename: str, criterion_accuracy: bool = True) \
        -> Tuple[List[float], List[float], List[float], List[float]]:

    states: List[int] = []
    sz_evals: List[float] = []
    sp_evals: List[float] = []
    re_evals: List[float] = []

    f = open(filename)

    lines: List[str] = f.readlines()

    while lines:
        header = lines.pop(0)

        _, _, length, _ = header.split()
        min_acc: float = 3
        st0, sz0, sp0, re0 = 0, 0, 0, 0
        for _ in range(int(length)):
            line = lines.pop(0)
            st, e, sz, sp, re = line.split()
            new_acc = float(re) + float(sp) + (0 if criterion_accuracy else float(sz))
            if new_acc < min_acc:
                min_acc = new_acc
                st0 = st
                sz0 = sz
                sp0 = sp
                re0 = re
        # Keep only the most accurate FSM
        states.append(int(st0))
        sz_evals.append(float(sz0))
        sp_evals.append(float(sp0))
        re_evals.append(float(re0))

    f.close()
    return states, sz_evals, sp_evals, re_evals


def read_scalability_file(filename: str) \
        -> Tuple[List[float], List[float], List[float]]:

    states: List[int] = []
    durations: List[float] = []
    solution_set_length: List[int] = []

    f = open(filename)

    lines: List[str] = f.readlines()

    while lines:
        header = lines.pop(0)

        _, length, t = header.split()
        solution_set_length.append(int(length))
        durations.append(float(t))

        for _ in range(int(length)):
            line = lines.pop(0)
            states.append(int(line))

    f.close()
    return states, solution_set_length, durations


def get_stats(values: List[List[float]]) -> Tuple[List[float], List[float], List[float]]:
    mean_arr: List[float] = []
    max_arr: List[float] = []
    min_arr: List[float] = []

    for value in values:
        mean_arr.append(mean(value))
        max_arr.append(max(value))
        min_arr.append(min(value))

    return mean_arr, max_arr, min_arr


def plot_values(mean_vals: Tuple[List[float], List[float]],
                max_vals: Tuple[List[float], List[float]],
                min_vals: Tuple[List[float], List[float]],
                trace_numbers: List[int],
                dashed_linear_func: bool = False,
                show_average: bool = False):

    if not show_average:
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex='all')
        ax0.errorbar(trace_numbers, mean_vals[0], yerr=0, fmt='-o')
        ax0.set_title("solution set size")
        ax1.errorbar(trace_numbers, mean_vals[1], yerr=0, fmt='-o')
        ax1.set_title("runtime")
    else:
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex='all')
        ax0.errorbar(trace_numbers, mean_vals[0], yerr=0, fmt='-o')
        ax0.set_title("specificity")
        ax1.errorbar(trace_numbers, mean_vals[1], yerr=0, fmt='-o')
        ax1.set_title("recall")
        means = [(mean_vals[0][x] + mean_vals[1][x]) / 2 for x in range(len(mean_vals[1]))]
        ax2.errorbar(trace_numbers, means, yerr=0, fmt='-o')
        ax2.set_title("accuracy")


def invert(vals: List[float]) -> List[float]:
    return [1 - x for x in vals]


if __name__ == "__main__":
    prefixes: List[str] = ["", "ls_", "sosa_", "rs_", ""]
    names: List[str] = [r"PSA$_{acc}$", "TSSHC", "SA", "RSSHC", r"PSA$_{obj}$"]
    markers: List[str] = ["o", "^", "s", "X", "*"]

    specificity_means: List[List[float]] = []
    recall_means: List[List[float]] = []
    size_means: List[List[float]] = []
    states_means: List[List[float]] = []
    solution_set_means: List[List[float]] = []
    runtime_means: List[List[float]] = []
    trace_nums = [100 + x * 50 for x in range(19)]

    for k, prefix in enumerate(prefixes):

        kfcv_files = [f"{prefix}kfcv_{i}_output.txt" for i in trace_nums]
        scalability_files = [f"{prefix}scalability_{i}_output.txt" for i in trace_nums]

        sp_list: List[List[float]] = []
        re_list: List[List[float]] = []
        size_list: List[List[float]] = []
        states_list: List[List[float]] = []
        solution_set_list: List[List[float]] = []
        runtime_list: List[List[float]] = []

        # Get the list of specificity and recall values
        for kfcv_file in kfcv_files:
            st, sz, sp, re = read_kfcv_file(kfcv_file, criterion_accuracy=(names[k] == r"PSA$_{acc}$"))
            states_list.append(st)
            size_list.append(sz)
            sp_list.append(sp)
            re_list.append(re)

        sp_values: Tuple[List[float], List[float], List[float]] = get_stats(sp_list)
        re_values: Tuple[List[float], List[float], List[float]] = get_stats(re_list)
        st_values: Tuple[List[float], List[float], List[float]] = get_stats(states_list)
        sz_values: Tuple[List[float], List[float], List[float]] = get_stats(size_list)

        specificity_means.append(sp_values[0])
        recall_means.append(re_values[0])
        states_means.append(st_values[0])
        size_means.append(sz_values[0])

        # Get the list of specificity and recall values
        for scalability_file in scalability_files:
            _, ssv, rt = read_scalability_file(scalability_file)
            solution_set_list.append(ssv)
            runtime_list.append(rt)

        ssv_values: Tuple[List[float], List[float], List[float]] = get_stats(solution_set_list)
        rt_values: Tuple[List[float], List[float], List[float]] = get_stats(runtime_list)

        runtime_means.append(rt_values[0])
        solution_set_means.append(ssv_values[0])
    fig2, ax2 = plt.subplots(1, 1)

    for count, (sp, re) in enumerate(zip(specificity_means, recall_means)):
        means = [(sp[x] + re[x]) / 2 for x in range(len(sp))]
        ax2.plot(trace_nums, invert(means), marker=markers[count], label=names[count])
        plt.ylim([0.5, 1])
        print(f"{names[count]}: {invert(means)}")

    for x in range(19):
        s = f"{trace_nums[x]} & "
        # for y in [3, 1, 2, 0]:
        #     sp = math.floor((1 - specificity_means[y][x]) * 100) / 100
        #     s += f"{sp} & "
        for y in [3, 1, 2, 0, 4]:
            re = math.floor((1 - size_means[y][x]) * 100) / 100
            s += f"{re} & "
        for y in [3, 1, 2, 0, 4]:
            m = math.floor((1 - (specificity_means[y][x] + recall_means[y][x]) / 2) * 100) / 100
            s += f"{m}"
            if y != 4:
                s += " & "
            else:
                s += "\\\\"
        print(s)

    # for x in range(19):
    #     print(solution_set_means[0][x])
    plt.xlabel("Number of log traces")
    plt.ylabel(r"Accuracy = $\frac{SP+REC}{2}$")
    ax2.legend(loc='best', fontsize="x-small")
    plt.grid()
    plt.show()

    fig3, ax3 = plt.subplots(1, 1)
    for count, rt in enumerate(runtime_means):
        if count == 4:
            break
        ax3.plot(trace_nums, rt, marker=markers[count], label=names[count] if names[count] != r"PSA$_{accuracy}$" else "PSA")
        print(f"{names[count]}: {rt}")

    for x in range(19):
        s = f"{trace_nums[x]} & "
        for y in [3, 1, 2, 0]:
            m = math.floor(runtime_means[y][x] * 100) / 100
            s += f"{m}"
            if y != 0:
                s += " & "
            else:
                s += "\\\\"
        print(s)

    plt.xlabel("Number of log traces")
    plt.ylabel("Runtime (s)")
    ax3.legend(loc='best', fontsize="x-small")
    plt.grid()
    plt.show()

    fig4, ax4 = plt.subplots(1, 1)
    for count, (st, sz, sp, re) in enumerate(zip(states_means, size_means, specificity_means, recall_means)):
        print(f"size for {names[count]}: {[sz[x] for x in range(len(sp))]}")
        means = [(sz[x] + sp[x] + re[x]) / 3 for x in range(len(sp))]
        ax4.plot(trace_nums, invert(sz), marker=markers[count], label=names[count])
        plt.ylim([0, 1])
        # print(f"{names[count]}: {invert(means)}")

    # for x in range(19):
    #     s = f"{trace_nums[x]} & "
    #     for y in [3, 1, 2, 0]:
    #         sp = math.floor((1 - specificity_means[y][x]) * 100) / 100
    #         s += f"{sp} & "
    #     for y in [3, 1, 2, 0]:
    #         re = math.floor((1 - recall_means[y][x]) * 100) / 100
    #         s += f"{re} & "
    #     for y in [3, 1, 2, 0]:
    #         m = math.floor((1 - (specificity_means[y][x] + recall_means[y][x]) / 2) * 100) / 100
    #         s += f"{m}"
    #         if y != 0:
    #             s += " & "
    #         else:
    #             s += "\\\\"
    #     print(s)


    for x in range(19):
        s = ""
        for y in [4]:
            m = math.floor((1 - (specificity_means[y][x] + recall_means[y][x] + size_means[y][x]) / 3) * 100) / 100
            s += f"{m} & "
        print(s)

    plt.xlabel("Number of log traces")
    plt.ylabel(r"Compression = $\frac{|states(FSM)|}{|states(initial)|}$")
    ax4.legend(loc='best', fontsize="x-small")
    plt.grid()
    plt.show()
