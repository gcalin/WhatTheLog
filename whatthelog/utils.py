# -*- coding: utf-8 -*-
"""
Created on Tuesday 04/27/2021
Author: Tommaso Brandirali
Email: tommaso.brandirali@gmail.com
"""

#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import linecache
import os
import tracemalloc

from itertools import zip_longest

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.auto_printer import AutoPrinter

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Utility Functions
#****************************************************************************************************

def get_peak_mem(snapshot, key_type='lineno') -> int:
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    return sum(stat.size for stat in snapshot.statistics(key_type))

# --- Parser for memory usage data from a tracemalloc snapshot ---
# Source: https://stackoverflow.com/a/45679009
def profile_mem(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print(f"Total allocated size: {bytes_tostring(total)}")

# --- Parser for human-readable file sizes ---
# Source: https://web.archive.org/web/20111010015624/http://blogmag.net/blog/read/38/Print_human_readable_file_size
def bytes_tostring(num):
    for x in ['bytes','KB','MB','GB','TB']:
        if num < 1024.0:
            return "%3.1f%s" % (num, x)
        num /= 1024.0

# --- Block splitter for counting line endings ---
def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

# --- Groups list in groups of n and fills any holes in the end with the fillvalue ---
def group(iterable, n, fillvalue = None):
    args = [iter(iterable)] * n
    return list(zip_longest(fillvalue=fillvalue, *args))
