#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute scale for world encoder.

usage: compute_scale.py  <data_root>

options:
    -h, --help                Show this help message and exit
"""
import sys
import glob
import numpy as np
from docopt import docopt
from hparams import hparams, hparams_debug_string

print(__doc__)
if __name__ == "__main__":
    args = docopt(__doc__)
    data_root = args["<data_root>"]

    f0max = 0
    spmax = -1.e6
    spmin = 1.e6
    apmax = -1.e6
    apmin = 1.e6
    for fname in glob.glob(data_root+"/*.npy"):
        data = np.load(fname)
        f0 = data[:,0]
        sp = data[:,1:(hparams.coded_env_dim+1)]
        ap = data[:,(hparams.coded_env_dim+1):hparams.num_mels]
        f0max = max(f0.max(), f0max)
        spmax = max(sp.max(), spmax)
        spmin = min(sp.min(), spmin)
        apmax = max(ap.max(), apmax)
        apmin = min(ap.min(), apmin)
        print(f0.max(), sp.max(), sp.min(), ap.max(), ap.min())  
        
    print(f0max, spmax, spmin, apmax, apmin)    
    sys.exit(0)