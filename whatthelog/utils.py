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

import tracemalloc


#****************************************************************************************************
# Utility Functions
#****************************************************************************************************

def get_peak_mem(snapshot, key_type='lineno') -> int:
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    return sum(stat.size for stat in snapshot.statistics(key_type))

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
