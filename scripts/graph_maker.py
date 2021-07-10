import numpy as np
import matplotlib.pyplot as plt

from whatthelog.definitions import PROJECT_ROOT


def main():
    with open(PROJECT_ROOT.joinpath(f"resources/metrics_fitness_function.txt"), 'r') as f:
        lines = f.readlines()
        lines = [line[:-3] for line in lines]
        lines = [line.split(", ") for line in lines]
        total_rewards = lines[0]
        total_sizes = lines[1]
        total_f1scores = lines[2]
        total_recalls = lines[3]
        total_precisions = lines[4]
        total_specificities = lines[5]

    with open(PROJECT_ROOT.joinpath(f"resources/metrics_fitness_function_edited.csv"), 'w+') as f:
        f.write("episode, reward, f1_score, recall, specificity, precision, size\n")
        for i in range(len(total_rewards)):
            f.write(str(i)
                    + ", " + total_rewards[i]
                    + ", " + total_f1scores[i]
                    + ", " + total_recalls[i]
                    + ", " + total_specificities[i]
                    + ", " + total_precisions[i]
                    + ", "+ total_sizes[i]
                    + "\n")

    # total_rewards = np.zeros(150)
    # total_sizes = np.zeros(150)
    # total_f1scores = np.zeros(150)
    # total_recalls = np.zeros(150)
    # total_precisions = np.zeros(150)
    # total_specificities = np.zeros(150)
    #
    # for i in range(5):
    #     with open(PROJECT_ROOT.joinpath(f"resources/metrics_{i}.txt"), 'r') as f:
    #         lines = f.readlines()
    #         lines = [line[:-3] for line in lines]
    #         lines = [line.split(", ") for line in lines]
    #         lines = [[float(value) for value in line] for line in lines]
    #         total_rewards += lines[0]
    #         total_sizes += lines[1]
    #         total_f1scores += lines[2]
    #         total_recalls += lines[3]
    #         total_precisions += lines[4]
    #         total_specificities += lines[5]
    #
    # total_rewards /= 5
    # total_sizes /= 5
    # total_f1scores /= 5
    # total_recalls /= 5
    # total_precisions /= 5
    # total_specificities /= 5
    #
    # plt.plot(range(150), total_f1scores)
    # plt.ylabel("F1 Score")
    # # plt.savefig(PROJECT_ROOT.joinpath("out/plots/f1scores_scaled_reward.png"))
    # plt.show()
    #
    # plt.plot(range(150), total_rewards)
    # plt.ylabel("Total reward")
    # # plt.savefig(PROJECT_ROOT.joinpath("out/plots/total_rewards_scaled_reward.png"))
    # plt.show()
    #
    #
    # plt.plot(range(150), total_recalls)
    # plt.ylabel("Recall")
    # # plt.savefig(PROJECT_ROOT.joinpath("out/plots/recall_scaled_reward.png"))
    # plt.show()
    #
    # plt.plot(range(150), total_precisions)
    # plt.ylabel("Precision")
    # # plt.savefig(PROJECT_ROOT.joinpath("out/plots/precision_scaled_reward.png"))
    # plt.show()
    #
    #
    # plt.plot(range(150), total_specificities)
    # plt.ylabel("Specificity")
    # # plt.savefig(PROJECT_ROOT.joinpath("out/plots/specificity_scaled_reward.png"))
    # plt.show()
    #
    # plt.plot(range(150), total_sizes)
    # plt.ylabel("Size Compression")
    # # plt.savefig(PROJECT_ROOT.joinpath("out/plots/size compression_scaled_reward.png"))
    # plt.show()



if __name__ == '__main__':
    main()