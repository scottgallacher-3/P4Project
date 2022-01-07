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
import pandas as pd

from mapobject import ZygoMap
from utilities import matchdims, ztestf


def combinemaps(lowermap, uppermap, newname=None, optimised=True, output=True):
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
        if newname is None:
            newname = "{0}+{1}".format(lowermap.mapname, uppermap.mapname)
            
        interfacemap = ZygoMap(array=interface, map1=lowermap, map2=uppermap, angle=optimalangle, mapname=newname)
        
    else:
        interfacemap = ZygoMap(array=interface, mapname=newname)
    
    #give user some knowledge on the optimised set-up
    #this could be moved elsewhere possibly for better access ->  added __str__() method to allow user to print() attributes
    #added flags for optimisation and output
    #set output to false for auto-comparison, so can display at end in a sorted order
    if optimised and output:
        if getattr(lowermap, "mapname") and getattr(uppermap, "mapname"):  # - if each defined from files, can use their filenames as references
            print(("Combined interface map: '{0}'"
                   "\nMaps combined for optimal angle of {1:.2f} degrees."
                   "\n'{2}' clockwise w.r.t '{3}'").format(interfacemap.mapname, optimalangle,uppermap.mapname,lowermap.mapname))
        else:
            print(("Maps combined for optimal angle of {0:.2f} degrees"
                   "map2 clockwise w.r.t map1").format(optimalangle))
        print("Optimised peak-to-valley height: {0:.1f} nm".format(interfacemap.peakvalley * 1e9))
        print("Optimised RMS height: {0:.1f} nm".format(interfacemap.rms * 1e9))
        
    
    return interfacemap


def comparebonds(zmaps, plot=False):
    if isinstance(zmaps, dict):
        mpairs = np.array(list(combinations(zmaps.values(), 2)))
    elif isinstance(zmaps, list) or isinstance(zmaps, tuple):
        mpairs = np.array(list(combinations(zmaps, 2)))
    else:
        raise TypeError(("'comparebonds' function requires an iterable of type 'dict', 'list', or 'tuple'. "
                         "The iterable must contain only objects of type 'ZygoMap'"))
    
    #combine maps, creating interface object for each pair (of ZygoMap class, defined with array rather than filename)
    #will store the PV & RMS values for each in numpy arrays, to then be sorted best to worst
    
    interfacemaps = np.zeros(len(mpairs), dtype=object)  # - to store each combo pair's interfacemap (ZygoMap object)
    for i in range(len(mpairs)):
        combo = mpairs[i]
        interfacemap = combinemaps(*combo, output=False)
        interfacemaps[i] = interfacemap
        
    #create a pandas DataFrame to concentrate relevant quantities/attributes to one object
    #there is plenty of scope to use the data further, so best to provide maps and metrics alongside
    #provide: the interface 'ZygoMap' objects ["Bond Map"],
    #names of combined maps ["map1"],["map2"],
    #peakvalley ["PV"] and rms ["RMS"] heights.
    sorted_bonds = pd.DataFrame({"Bond Map": interfacemaps,
                                 "map1": [bond.map1.mapname for bond in interfacemaps], 
                                 "map2": [bond.map2.mapname for bond in interfacemaps], 
                                 "PV": [bond.peakvalley for bond in interfacemaps], 
                                 "RMS": [bond.rms for bond in interfacemaps]})
    
    #sort by peakvalley as standard
    sorted_bonds = sorted_bonds.sort_values(by="PV")
            
    if plot is True:
        #sort the found pv & rms heights, ordering by the lowest pv in both cases (keep both values aligned per combo)
        plt.figure(figsize=(12,5))
        plt.plot(sorted_bonds.sort_values(by="PV")["PV"].values, "o", label="Peak-Valley")
        plt.plot(sorted_bonds.sort_values(by="PV")["RMS"].values, "o", label="RMS")
        plt.title("Bond height values comparison (sorted by lowest peak-valley value)\n")

        plt.xlabel("Maps comprising simulated bond")
        plt.ylabel("Height of bond interface (nm)")
        plt.xticks(range(len(mpairs)), np.array([bond.mapname for bond in sorted_bonds.sort_values(by="PV")["Bond Map"]]), rotation=50)
        plt.yticks(np.arange(0, sorted_bonds["PV"].max() + 10e-9, 10e-9))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-9,-9), useMathText=True)
        plt.tick_params(axis="x", labelsize=6)
        plt.axhline(60e-9, linestyle="solid", linewidth=3, color="r")
        plt.legend(loc="upper left", fontsize=14)
        plt.grid()
        plt.show()
            
            
        #sort the found pv & rms heights, ordering by the lowest rms in both cases (keep both values aligned per combo)
        plt.figure(figsize=(12,5))
        plt.plot(sorted_bonds.sort_values(by="RMS")["PV"].values, "o", label="Peak-Valley")
        plt.plot(sorted_bonds.sort_values(by="RMS")["RMS"].values, "o", label="RMS")
        plt.title("Bond height values comparison (sorted by lowest RMS value)\n")

        plt.xlabel("Maps comprising simulated bond")
        plt.ylabel("Height of bond interface (nm)")
        plt.xticks(range(len(mpairs)), np.array([bond.mapname for bond in sorted_bonds.sort_values(by="RMS")["Bond Map"]]), rotation=50)
        plt.yticks(np.arange(0, sorted_bonds["PV"].max() + 10e-9, 10e-9))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-9,-9), useMathText=True)
        plt.tick_params(axis="x", labelsize=6)
        plt.axhline(60e-9, linestyle="solid", linewidth=3, color="r")
        plt.legend(loc="upper left", fontsize=14)
        plt.grid()
        plt.show()
            
    return sorted_bonds
