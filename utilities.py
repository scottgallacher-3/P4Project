"""Module for necessary but not central helper/utility functions for `mapcompare.py`.
These could be included within other functions, as they are only accessed there.
Instead, keep separate and only import to be used when needed.
"""

import numpy as np
import scipy.ndimage

def matchdims(map1,map2):
    #given two arrays (not map objects), truncate them to their lowest shared dimensions to be able to sum them
    #note: should probably choose to add rows/columns rather than remove data
    m1,m2 = map1.copy(),map2.copy()
    
    m1dims, m2dims = m1.shape, m2.shape
    diffs = [(m1dims[0] - m2dims[0]), (m1dims[1] - m2dims[1])]  # - find difference in number of rows & columns between arrays
    
    #two types of slices for which dimension is the largest between the two maps
    #use modulo % to check divisibility, // to do whole number division
    #then slice from each end of array (avoid bias/truncating on one side)
    #take the divisor result from both sides, then take the remainder from end of array
    #balances as much as possible, but odd-numbered differences will be asymmetric (just take remaining 1 from end of array)
    #use (len(array) - number) to do backwards slice - needed for minus zero slice which is treated as zero
    #using len() gives an absolute index rather than relative
    #addition of abs() makes things easier
    
    #for rows
    rem,div = (diffs[0] % 2), (diffs[0]//2)
    if m1dims[0] > m2dims[0]:
        m1 = m1[abs(div):m1.shape[0]-abs(div+rem),:]
    elif m1dims[0] < m2dims[0]:
        m2 = m2[abs(div):m2.shape[0]-abs(div+rem),:]
        
    #for columns
    rem,div = (diffs[1] % 2), (diffs[1]//2)
    if m1dims[1] > m2dims[1]:
        m1 = m1[:, abs(div):m1.shape[1]-abs(div+rem)]
    elif m1dims[1] < m2dims[1]:
        m2 = m2[:, abs(div):m2.shape[1]-abs(div+rem)]
    
    return m1,m2


def ztestf(*args):  # - optimisation of rotation around z-axis of upper map w.r.t lower map
    angz = args[0][0]  # - optimize.minimize gives angz as [0.] (why?) so needs extracted from list as well as args tuple
    lowermap, uppermap = args[1],args[2]

    dims = uppermap.shape
    
    #now get new rotated array (rotating around z/in x-y plane)
    #using scipy.ndimage.rotate()
    #order=0 means no additional interpolation of values when rotating
    #reshape=False maintains original dimensions (used in case of rotating outside of original shape - not applicable here)
    #use mode="constant" and cval=np.nan to fill out all (and perhaps new) invalid points with nan
    newheights = scipy.ndimage.rotate(uppermap, angz, order=0, reshape=False, mode="constant", cval=np.nan)
    
    #simulate the surface contact, using simple 1 point of contact
    #one map flipped horizontally and negated, their addition describes the interface heights
    interface = -newheights[::-1] + lowermap
    interface += abs(np.nanmin(interface))  # - set contact point to be zero height (negative values are non-physical intersection of surfaces)
    interface -= np.nanmean(interface)
    
    #again minimising peak-to-valley height, though a sum of squares approach could be used
    peakvalley = np.nanmax(interface) - np.nanmin(interface)
    
    return peakvalley