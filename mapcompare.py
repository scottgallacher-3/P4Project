"""Functions to operate alongside the ZygoMap class from `mapobject.py`.
These will work together to allow a pair of maps to be combined,
such that the interface between them may also be created as a ZygoMap.

The combination of maps assumes the "upper" map is flipped over to face downwards relative to the "lower" map.
Then, the interface map is created as the gap between the surfaces when the maps are brought into contact in this way.
This emulates the profile of a bond between the surfaces.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.ndimage
from itertools import combinations

from mapobject import ZygoMap


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


def combinemaps(lowermap, uppermap, optimised=True, output=True):
    m1,m2 = lowermap.heights, uppermap.heights
    #function to combine ZygoMap objects
    #flips and negates values of the 2nd map "uppermap"
    #emulating the surface placed faced down on the other
    #returns ZygoMap object from the combination of the lower map and the transformed uppermap
    #where the magnitude of largest negative has been added back as an offset to prevent non-physical overlap of surfaces
    
    #check for equal shapes (changed to automatically crop to smallest shared values)
    #use matchdims() function, will return arrays with equal rows,columns for direct addition of arrays
    if m1.shape != m2.shape:
        m1,m2 = matchdims(m1,m2)
    
    #find optimised angle of rotation (of uppermap with respect to lowermap)
    #minimise the peakvalley height with rotation angle around z-axis
    #NOTE: previous rotation method not working the same for z rotations
    #using scipy.ndimage.rotate (with order=0 to maintain array values (no spline interpolation))
    #large angular range needed -> need brute() function to get the accurate value
    #define angular range by ranges=(slice(0,360),) (the slice object is preferred by brute function definition)
    optimalangle = 0
    if optimised == True:
        optimalangle = scipy.optimize.brute(ztestf, ranges=((slice(0,360),)), args=(m1,m2))[0]
    
        #apply this angle with scipy.ndimage.rotate() to get new array of the rotated uppermap
        #use this array directly for the interface (in place of uppermap.heights), and leave each individual map untouched
        #should store some indicator for user of the optimal angle used (property of interfacemap ?)
        newheights = scipy.ndimage.rotate(m2, optimalangle, order=0, reshape=False, mode="constant", cval=np.nan)
    else:
        newheights = m2.copy()

    #flip and negate uppermap, then add to lower map for interface
    #note: using only simple one-point contact
    #for more realistic simulation, want to find 3 points or simulate how the upper surface would "settle" onto lower
    #experimented (manually simulating 2 rotations based on position of point around centre)
    #but not a clear successful method, would leave as future work
    interface = -newheights[::-1] + m1
    interface += abs(np.nanmin(interface))  # - add back largest overlap, to leave maps just touching
    interface -= np.nanmean(interface)  # - centre in z-axis around mean
        
    #construct as ZygoMap object
    #providing basic details about its construction (the combination of which maps at what angle)
    if hasattr(lowermap, "mapname") and hasattr(uppermap, "mapname"):
        interfacemap = ZygoMap(array=interface, map1=lowermap, map2=uppermap, angle=optimalangle,
                               mapname="{0}+{1}".format(lowermap.mapname, uppermap.mapname))
    else:
        interfacemap = ZygoMap(array=interface)
    
    #give user some knowledge on the optimised set-up
    #this could be moved elsewhere possibly for better access ->  added __str__() method to allow user to print() attributes
    #added flags for optimisation and output
    #set output to false for auto-comparison, so can display at end in a sorted order
    if optimised and output:
        if getattr(lowermap, "mapname") and getattr(uppermap, "mapname"):  # - if each defined from files, can use their filenames as references
            print("Maps combined for optimal angle of {0:.2f} degrees\n\
            {1} clockwise w.r.t {2}".format(optimalangle,uppermap.mapname,lowermap.mapname))
        else:
            print("Maps combined for optimal angle of {0:.2f} degrees\n\
            (map2 clockwise w.r.t map1)".format(optimalangle))
        print("Optimised peak-to-valley height: {0:.1f} nm".format(interfacemap.peakvalley * 1e9))
        print("Optimised RMS height: {0:.1f} nm".format(interfacemap.rms * 1e9))
        
    
    return interfacemap


def comparebonds(zmaps, sort="both", print_output=False, plot=False):
    #NOTE: using itertools.combinations module
    #for a list (or maybe dict as well ?) of map objects, iterate through every pair combination
    #comparing bonds by peak-to-valley height & RMS height
    #additionally, can print information about optimal angle (around z-axis) to combine each pair
    
    #attain array from generator object returned by combinations (using n=2 items per combo)
    mpairs = np.array(list(combinations(zmaps, 2)))
    
    #combine maps, creating interface object for each pair (of ZygoMap class, defined with array rather than filename)
    #will store the PV & RMS values for each in numpy arrays, to then be sorted best to worst
    #choice to store the interfaces ? - for low number of arrays this should be ok
    
    #initialise empty arrays which will store the PV/RMS as calculated (necessary if not storing each interface)
    pvValues = np.zeros(len(mpairs))
    rmsValues = np.zeros(len(mpairs))
    
    interfacemaps = np.zeros(len(mpairs), dtype=object)  # - to store each combo pair's interfacemap (ZygoMap object)
    #note: may not be so simple - only local to function, may need to either return interfacemaps or create as global variable
    for i in range(len(mpairs)):
        combo = mpairs[i]
        interfacemap = combinemaps(*combo, output=False)
        pvValues[i] = interfacemap.peakvalley
        rmsValues[i] = interfacemap.rms
        
        interfacemaps[i] = interfacemap
        
    #now, sort for display
    #sort the mpairs list differently for either pv or rms, storing each separately
    #allow "sort" keyword to limit comparison to only pv or only rms (defaults to "both")
    lowestpv = mpairs[pvValues.argsort()]
    lowestrms = mpairs[rmsValues.argsort()]
    
    #maps constructed between pairs, and ordered
    #this can be returned
    pv_sorted_maps = interfacemaps[pvValues.argsort()]
    rms_sorted_maps = interfacemaps[rmsValues.argsort()]
    
    #sort combos by lowest peak-valley height
    if sort in ("both","pv"):
        if print_output:
            print("Sample Bonds sorted by lowest peak-to-valley height:\n")
        for i in range(len(lowestpv)):
            combo = lowestpv[i]
            if print_output:
                print("{} - ".format(i+1),*[m.filename for m in combo])
                print("Peak-to-valley Height: {0:.1f} nm".format(pvValues[pvValues.argsort()][i] * 1e9))  # - convert to nanometres
                print("RMS Height: {0:.1f} nm".format(rmsValues[pvValues.argsort()][i] * 1e9))
                print("\n")
    
    #sort combos by lowest rms height
    if sort in ("both","rms"):
        if print_output:
            print("Sample Bonds sorted by lowest RMS height:\n")
        for i in range(len(lowestrms)):
            combo = lowestrms[i]
            if print_output:
                print("{} - ".format(i+1),*[m.filename for m in combo])
                print("RMS Height: {0:.1f} nm".format(rmsValues[rmsValues.argsort()][i] * 1e9))
                print("Peak-to-valley Height: {0:.1f} nm".format(pvValues[rmsValues.argsort()][i] *1e9))
                print("\n")
            
    if plot is True:
        if sort in ("both","pv"):
            #sort the found pv & rms heights, ordering by the lowest pv in both cases (keep both values aligned per combo)
            plt.figure(figsize=(10,6))
            plt.plot(pvValues[pvValues.argsort()], "o", label="Peak-Valley")
            plt.plot(rmsValues[pvValues.argsort()], "o", label="RMS")
            plt.title("Bond height values comparison (sorted by lowest peak-valley value)\n")
            
            plt.xlabel("Maps used in simulated bond")
            plt.ylabel("Height of bond interface (nm)")
            plt.xticks(range(len(mpairs)), np.array([m.map1.split(".")[0]+"\n"+m.map2.split(".")[0] for m in interfacemaps])[pvValues.argsort()], rotation=0)
            plt.yticks(np.arange(0, max(pvValues) + 10e-9, 10e-9))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(-9,-9), useMathText=True)
            plt.axhline(60e-9, linestyle="solid", linewidth=3, color="r")
            plt.legend(loc="upper left", fontsize=14)
            plt.grid()
            plt.show()
            
        if sort in ("both","rms"):
            #sort the found pv & rms heights, ordering by the lowest rms in both cases (keep both values aligned per combo)
            plt.figure(figsize=(10,6))
            plt.plot(pvValues[rmsValues.argsort()], "o", label="Peak-Valley")
            plt.plot(rmsValues[rmsValues.argsort()], "o", label="RMS")
            plt.title("Bond height values comparison (sorted by lowest RMS value)\n")
            
            plt.xlabel("Maps used in simulated bond")
            plt.ylabel("Height of bond interface (nm)")
            plt.xticks(range(len(mpairs)), np.array([m.map1.split(".")[0]+"\n"+m.map2.split(".")[0] for m in interfacemaps])[rmsValues.argsort()], rotation=0)
            plt.yticks(np.arange(0, max(pvValues) + 10e-9, 10e-9))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(-9,-9), useMathText=True)
            plt.axhline(60e-9, linestyle="solid", linewidth=3, color="r")
            plt.legend(loc="upper left", fontsize=14)
            plt.grid()
            plt.show()
            
    return pv_sorted_maps, rms_sorted_maps  # - nothing right now (simplest for usability) but could offer the sorted arrays and all interfacemaps