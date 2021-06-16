import random
import sys

from whatthelog.datasetcreator.dataset_factory import DatasetFactory
from whatthelog.definitions import PROJECT_ROOT


sys.path.append("./../")

random.seed(5)


def main():
    # pt = PrefixTreeFactory().get_prefix_tree(PROJECT_ROOT.joinpath("resources/test/"), PROJECT_ROOT.joinpath("resources/config.json"))
    DatasetFactory(
        PROJECT_ROOT.joinpath("resources/traces_large/")).create_data_set(100, 20, 20,
                                                                      unique_graph=True,
                                                                      remove_trivial_loops=False,
                                                                      visualize_tree=True,
                                                                      name_tree="prefix_tree_original.png")
    print("Finished!")


if __name__ == "__main__":
    main()
