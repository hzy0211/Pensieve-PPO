import os


COOKED_TRACE = 'bandwidth.txt'


def load_trace(cooked_trace=COOKED_TRACE):
    f = open(cooked_trace, "r")
    all_cooked_bw = []
    for eachline in f.readlines():
        all_cooked_bw.append(float(eachline))
    
    return all_cooked_bw
