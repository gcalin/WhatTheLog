# WhatTheLog

A Python tool for the interpretation and modeling of log traces.

## Introduction

This tool was built as part of a research project from four TU Delft CSE bachelor students. The project aims to tackle the problem of analyzing and monitoring complex software systems through their log traces, and it does so by using prefix trees and various unsupervised learning methods.

This tool implements algorithms for generating high-level finite state machines representing a system from sets of log traces. The state machines represent trained models, and they implement functions to test new traces for compliance with the model. This allows checking for unusual execution paths which can be symptoms of bugs, interference from malicious actors, and other unwanted behavior. The trained models aim to be concise, human-readable and accurate as much as possible.

This tool was built and tested on log traces from the [XRP Ledger](https://github.com/ripple/rippled)'s consensus protocol.

## Usage

TODO

## Architecture and Flow

### The Syntax Tree

The tool's setup begins with the definition of a syntax for the log traces, in other words converting unstructured log statements into structured data. This allows the software to distinguish between static and dynamic parts of single log statements, and to identify when two statements represent the same state. Although [promising research](https://ieeexplore.ieee.org/abstract/document/9134790) has been conducted on the automation of this process, this project does not focus on log parsing and therefore requires the syntax definition to be done manually.

The syntax must be defined as a prefix tree, in JSON format. An example of such a file can be found [here](resources/config.json). Every node in this tree represents a substring, or a regex pattern, common to one or more templates of log statements. For example, take the following log statements:

```
2020-Mar-01 16:22:46.577321447 NFO Entering consensus process, watching, synced=no
2020-Mar-01 16:24:02.774757607 DBG normal consensus
```

This trace could be represented by the following syntax tree:

```
{
	
  "name": "timestamp",
  "prefix": "(\\d{4}-[A-Z][a-z][a-z]-\\d{2}\\s\\d{2}\\:\\d{2}\\:\\d{2}\\.\\d{9})\\s",
  "isRegex": true,
  "children": [
  	{
  		"name": "info",
		"prefix": "NFO\\s",
		"isRegex": false,
		"children": [
			{
	          "name": "EnteringConsensus",
	          "prefix": "(Entering\\sconsensus\\sprocess,\\s(validating|watching),\\ssynced=(yes|no))",
	          "isRegex": true,
	          "children": []
	        }
		]
  	},
  	{
  		"name": "debug",
		"prefix": "DBG ",
		"isRegex": false,
		"children": [
			{
				"name": "NormalConsensus",
				"prefix": "normal consensus",
				"isRegex": false,
				"children": []
			}
		]
  	}
  ]
}
```

Traces are evaluated left to right, therefore a parent node represents a prefix which is common to one or more log templates, such as the timestamp in this example. The `"name"` field must be unique and it cannot be `"termination"`. The `"prefix"` field can be either a string or a regex pattern, and the `"isRegex"` field specifies whether or not the name should be interpreted as a regex pattern or a regular string.

***Note***: only leaf nodes represent valid log templates, in our example the log `2020-Mar-01 16:22:46.577321447 NFO` would not be valid based on the syntax tree.

The syntax configuration file is parsed into an instance of [`SyntaxTree`](whatthelog/syntaxtree/syntax_tree.py) by the [`SyntaxTreeFactory`](whatthelog/syntaxtree/syntax_tree_factory.py) class. This instance is then used to classify log statements during training.

### The Prefix Tree

The structure of an whole set of log traces is represented once again as a prefix tree. However, whereas the syntax tree used nodes to represent substrings and branches to represent log statements, this tree uses nodes to represent log statements and branches to represent full log traces or execution paths. A single log trace can therefore be parsed as a linear tree, where each node has at most 1 child. This tree can then grow by parsing more log traces as new branches. This is in fact the first step of training of the model.

Consider the following traces, with letters representing unique log templates:

```
A
B
C
D
```
```
A
B
D
E
```
```
A
F
G
```

These can be represented by the following prefix tree:

```
      A
     / \
    B   F
   / \   \
  C   D   G
 /     \
D       E
```

This tree can be viewed as the most naive state machine representing the system under analysis. Here every valid execution path is represented by a separate branch, and every log statement is a separate state.

This prefix tree is defined in the [`PrefixTree`](watthelog/prefixtree/prefix_tree.py) class and instances of it can be generated from a set of logs using the [`PrefixTreeFactory`](whatthelog/prefixtree/prefix_tree_factory.py). Instances of the tree can be pickled to a file and unpickled from it using methods in the factory class.

***Note*** that at this stage the model must be a valid tree, therefore loops and recursion are parsed linearly. This can generate valid trees like this one:

```
        A
       / \
      B   F
     / \   \
    B   D   G
   /     \
  C       E
 /         \
B           E
```

***Note***: the current prefix tree implementation adds placeholder root and termination nodes for internal handling, so the tree from the initial example would be internally stored as:

```
        R
        |
        A
       / \
      B   F
     / \   \
    C   D   G
   /  	 \   \
  D       E   T
 /         \
T           T
```

Where `R` is the placeholder root and `T` are the placeholder termination nodes.

### The Graph

The Prefix Tree is represented internally as directed graph, which is implemented in the [`StateGraph`](whatthelog/prefixtree/state_graph.py) class. This graph implementation stores the following fields:

```
self.edges = SparseMatrix()
self.states: List[State] = []
self.state_indices_by_id: Dict[int, int] = {}
self.states_by_prop: Dict[int, int] = {}
```

The `edges` field stores a sparse matrix of edge values encoded as strings. If a cell `(x, y)` holds a string `s` then the graph has an edge from node number `x` to node number `y`, and the edge has properties that can be decoded from `s`. A data class [`EdgeProperties`](whatthelog/prefixtree/edge_properties.py) is provided to represent values stored inside an edge, it implements methods to serialize and deserialize instances of the class to and from the string values stored in the matrix. The matrix is internally represented as a list of strings in the form `x.y.s` for the sake of memory efficiency, and it is accessed in `O(log(n))` using binary search.

The `states` field holds a list of instances of the [`State`](whatthelog/prefixtree/state.py) class. This class represent an occurrence of a state in the prefix tree (such as `A` or `B` in our previous prefix tree example). A separate instance of this class is stored for every occurrence of the state in the tree. The `State` class holds a reference to an instance of [`StateProperties`](whatthelog/prefixtree/state_properties.py), where the actual log template is stored. Only one instance of `StateProperties` is stored for every unique set of log templates, and multiple `State` instances can point to the same state properties. Using once again our tree example from earlier: the tree will store one `State` instance for every occurrence of `A`, but all those states will point to one single instance of `StateProperties`, which holds the log template of `A`. Two states with the same set of properties are considered equivalent.

The `state_indices_by_id` field stores a mapping from the id of a state instance to the index of that state in the `states` list. This is necessary to allow checking for state membership in the tree in `O(1)`.

The  `states_by_prop` field stores a mapping from the hash of a `StateProperties` instance to the id of a `State` instance with those properties. Since the hash of `StateProperties` is deterministic based on an instance's field values, this mapping allows searching for existing equivalent states during the insertion of a new state in `O(1)`. If an existing equivalent is found, the new state will be set to point to the existing properties, in order to reduce redundancy.

The `merge_states` method can be used to merge two states. When merging `State`s s1 and s2 with templates t1 and t2, and neighbours lists n1 and n2, the result would be a state s3 with template `t1 | t2` and neighbours `n1 U n2`. If either state is terminal or the start state, so will the resulting merged state.

### The State Machines

TODO

## Scripts

### `prefix_tree_generator.py`

This script shows the basic usage of the `PrefixTreeFactory` class for generating and pickling a full prefix tree from a set of traces. Since the tree generation can take time for large datasets, this script also times the execution of the process. It can also be used to profile and analyze the memory usage of the tree parsing process.

### `tree_profiler.py`

This script is used to analyze the memory usage of a `PrefixTree` instance from a pickle file. It shows the estimated size in memory of the various components of the tree implementation.

### `log_scrambler.py`

This script is used to generate invalid log traces from a set of valid ones for the purpose of testing the models. This is done by applying a random selection of mutations to the original traces. Each mutation is applied to one randomly selected line and all the adjacent lines that fit the same template, this set is called a "selection" of lines. There are three mutation operations:

- **Deletion**: delete the selected lines
- **Adjacent Swap**: swap the selected lines with the next adjacent selection
- **Random Swap**: swap the selected lines with a different random selection from the same file

This implementation provided through `process_file(...)` does not guarantee the invalidity of the selected traces, however testing shows an average of illegal traces between 90% and 95% of the total. An alternative that applies mutations until a false trace is produced is implemented in `produce_false_trace(...)`, which applies random mutations to a trace and verifies its validity against a state model until the trace is rejected.

### `match_trace.py`

This scripts checks if a state model accepts a given trace. It uses two pointers, one pointing at the a state in the state model and one pointing at a line in the log trace. Initially, the state pointer is at the start state, and the log pointer is at the first line of trace. If the first line of the trace matches the initial state, the line pointer is moved to the second and the the state pointer is moved to any state that has an incoming edge from the current one that matches the log pointer. If no such state exists, the search fails and `None` is returned. If a state that matches the log exists, the procedure continues iteratively until either no matching state is found for the log entry or until all log entries have been exhausted. If all entries have been matched, this means there is a path in the state graph that corresponds to the log trace. If this is the case, and if the last state in the path points to a terminal state, that path is returned as a list of states.

### `accuracy_calculator.py`

This script is used to quantitatively evaluate the accuracy of a state model given a list of log traces, using k-fold cross-validation (KFCV). It takes in a directory of traces, and randomly partitions it into `k` roughly equally sized chunks that are used for the KFCV algorithm. It performs `k` iterations, where the traces corresponding to fold `i` are moved to a new directory and left unchanged. Each trace of the moved traces is then used to produce a negative trace in a third directory, using the `log_scrambler` script. After this separation has been made, the state model is produced using the untouched `k-1` traces. The model's recall (<img src="https://render.githubusercontent.com/render/math?math=\color{red}\frac{|TP|}{|TP|%2B|FN|}">) is then evaluated using the true logs in the current fold, and its specificity (<img src="https://render.githubusercontent.com/render/math?math=\color{red}\frac{|TN|}{|TN|%2B|FP|}">) is computed using the false traces generated from the current fold. When the current interation is finished, the positive traces of the current fold are moved back to the original directory, and the false traces and the additional directories are removed. The prodcedure is repeated for each fold, and the results are collected and returned at the end of the procedure, where the means and standard deviations for both accuracy and recall across all iterations are also computed.

### `log_filter.py`

This script is used to filter a log dataset for lines that do not match a provided Syntax Tree. It can be useful during the manual compilation of a syntax file to identify templates which have yet to be defined in the syntax configuration.


## Running the tests

A small test suite is provided in the `tests` directory. It uses the `pytest` framework and can be ran by simply running `pytest` as a command in the project's root directory.


## Contributors

- Tommaso Brandirali (tommaso.brandirali@gmail.com)
- Calin Georgescu (C.A.Georgescu@student.tudelft.nl)
- Pandelis Symeonidis (P.L.Symeonidis@student.tudelft.nl)
- Thomas Werthenbach (t.a.k.werthenbach@student.tudelft.nl)
