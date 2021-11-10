import os
from typing import List

from whatthelog.exceptions import UnidentifiedLogException
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory


def file_to_abbading_format(log_trace: List[str], syntax_tree: SyntaxTree, accept_trace: bool) -> List[int]:
    res: List[int] = [1 if accept_trace else 0, len(log_trace)]

    for log_entry in log_trace:
        search_res: SyntaxTree = syntax_tree.search(log_entry)

        if search_res is None or search_res.index == -1:
            raise UnidentifiedLogException()

        res.append(search_res.index)

    return res


def directory_to_abbadingo_format(logs_directory: str, config_file: str, accept_traces: bool, output_file: str) -> None:

    if not os.path.isdir(logs_directory):
        raise NotADirectoryError("Log directory not found!")
    if not os.path.isfile(config_file):
        raise FileNotFoundError("Config file not found!")

    syntax_tree: SyntaxTree = SyntaxTreeFactory(True).parse_file(config_file)
    res: List[str] = []
    max_num: int = 0
    total_traces: int = len([name for name in os.listdir(logs_directory)
                             if os.path.isfile(os.path.join(logs_directory, name))])
    for filename in os.listdir(logs_directory):
        lines: List[str] = []
        with open(f"{tracesdir}/{filename}") as file:
            lines = file.readlines()

        processed_file: List[int] = file_to_abbading_format(lines, syntax_tree, accept_traces)
        max_num = max(max_num, max(processed_file[2:]))
        res.append(" ".join(map(str, processed_file)))

    with open(output_file, "w") as file:
        file.write(f"{total_traces} {max_num}\n")
        file.writelines(map(lambda x: x + "\n", res))

if __name__ == "__main__":
    tracesdir = "../tests/resources/testlogs"
    cofnigfile = "../resources/config.json"

    directory_to_abbadingo_format(tracesdir, cofnigfile, True, "../tests/resources/abbadingo_input_1.a")
