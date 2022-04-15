# -*- coding: utf-8 -*-
"""Spyder에서 Console과 Debugger Console 시작 시 Import 한 Module.

Created on Wed May 26 10:02:51 2021

@author: 71815
"""
# Matlab command
# Third party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure, hist, plot, subplots
from mkl_fft import fft, fft2, ifft, ifft2
from mkl_random import choice, randint, randn, standard_normal, uniform
from numpy import (
    amax, amin, angle, arange, arccos, arcsin, arctan, argmax, argmin, array,
    conj, cos, cross, cumsum, deg2rad, diff, dot, exp, hstack, interp,
    linspace, mean, newaxis, ones, pi, rad2deg, sin, sqrt, tan, unwrap, vstack,
    zeros)
from numpy.fft import fftfreq, fftshift, ifftshift
from numpy.linalg import norm, svd
from scipy.special import cosdg, sindg, tandg

# Local imports
from helper.helper import imagesc
from helper.plot_pg import imagescpg, plotpg
