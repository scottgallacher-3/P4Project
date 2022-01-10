"""Module for the ZygoMap class.
Contains all the code to read in, prepare map, store in array format, and collect relevant variables as attributes.
The class is able to create a map from either:
 - a user-provided MetroPro ASCII .txt file
 - a user-provided 2D array of height values.
Processing Steps:
 - 1. Interpolate the grid to smooth over any missing values.
     (This can be optionally re-applied after creation by users
     to change grid density and increase the resolution of heights array.)
 - 2. Initial storage of relevant quantities
      - Main analytical metrics - RMS height variation, peak-to-valley absolute height difference
      - Other (self-accessed) values - full grid coordinates, centroid of valid array points, positions of furthest valid rows/columns.
 - 3. Removal of any tilt from initial sample by solving for the ordinary least-squares plane that describes the overall tilt,
      and subtracting the plane offset to see only the variation of values around the tilted plane.
 - 4. Update relevant quantities based on the flattened array.
 - 5. Crop the array and centre to frame only the valid data.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate  # - interpolation function to fit data to a grid
import os  # - extracting e.g. filename without folders or extensions


class ZygoMap:
    """
    `ZygoMap` class to represent `Zygo` surface measurements as a 2D array of height values.
        
    Specifically designed to work with `MetroPro` ASCII-formatted data files
    containing surface measurements taken with `Zygo` interferometer technology.
    Phase data is extracted and converted to an array of height values at each
    point on a 2D grid. The surface profile is described through 2 key metrics,
    the peak-to-valley height and the RMS height.
    
    Maps can also be constructed directly from a pre-existing array.
    
    `ZygoMap` is also designed to represent combinations of maps using the height gap
    of the interface between them when they form a face-to-face bond. A separate
    function, `combinemaps` will simulate a bond between two maps, and construct a
    `ZygoMap` for the interface heights. These interface/bond maps can be operated in
    the same way as individual maps, but have additional attributes that reflect their
    creation.
    
    See __init__() constructor for details of map creation.
    
    
    Parameters:
        
    'filename' : str, optional
        - The local filepath of a `MetroPro` .txt data file of surface measurements.
        - If this is not supplied, then 'array' must be given to construct the object.
    'array' : numpy array, shape == (M,N), optional
        - Pre-existing array of height values on a 2D grid, to be processed
          and stored in 'heights'.
        - If 'array' is not None, then this will be used to construct the object,
          and 'filename' will be ignored whether it is given or not.

    'mapname' : str, optional
        - Name to be given as an identifier for the map object.
        - If 'filename' given, set by default to take the stem of the file (without .txt extension).
        - Otherwise, if 'map1' and 'map2' exist, set by default to join the individual
          map names with a "+" between them.
        - Otherwise, 'mapname' will be None. The user can supply their own names.
    'map1' : `ZygoMap` object, optional
        - Map object for a constituent map that has been used to construct the current one.
        - This is the "lower" map in the bond.
    'map2' : `ZygoMap` object, optional
        - Map object for a constituent map that has been used to construct the current one.
        - This is the "upper" map in the bond.
    'angle' : float, optional
        - The relative in-plane rotation angle which was applied to 'map2'
          to minimise the simulated bond profile.
    'flatten' : bool, optional
        - Flag to toggle the automatic tilt-removal process. Default is True.
        - If user is providing a pre-defined array, they may want to keep the data as-is,
          then flattening/tilt-removal can be turned off with `flatten=False`.

    Returns: None
    
    
    Key Methods:
    
    '__init__()' - Constructor for the class. Listed parameters above are passed here.
    
    'interpolate_grid' - Interpolate height data to fill in missing data, and fit to a grid.
    
    'crop' - Exclude data points using a radial crop, then centre and frame the valid data.
    
    'untilt' - Remove tilts present in the data by subtracting the best-fit plane describing the tilt.
    
    'plot' - Display either a 2D or 3D representation of the data.
    
    
    Key Attributes:
    
    'heights' : numpy array, shape == (M,N)
        - Main array for the height data, which is updated when processing.
    'peakvalley' : float
        - Main measurement metric for the height data.
        - Describes the overall height of the surface, measured from its
          highest point (peak) to lowest point (valley).
        - Individual 'peak' and 'valley' points are also available separately.
    'rms' : float
        - Secondary measurment metric for the height data.
        - Describes variation of height values around the mean, measured as the
          root-mean-square of all values in the array.
        
        
    Other/Utility Attributes:
    
    'heights0' : numpy array, shape == (K,L)
        - 'heights' is mutable, so a copy of initial data is kept as read-in.
    'heights1' : numpy array, shape == (K,L)
        - A copy of inital interpolated data is kept, before cropping or tilt
          removal is applied.
    'map1' : `ZygoMap` object
        - If the object is created as a bond interface, this represents the
          "lower" of the two maps making the bond.
    'map2' : `ZygoMap` object
        - If the object is created as a bond interface, this represents the
          "upper" of the two maps making the bond.
    'maps' : tuple of `ZygoMaps`
        - Access both 'map1' and 'map2' at once.
    'is_cropped' : bool
        - Flag to indicate whether the current data has been cropped.
    'r0' : float
        - Original radius of the uncropped data.
    'radius' : float
        - Current radius of data, cropped or uncropped.
        
    """
    
    #structure: definition of methods for the class (to perform operations on the ZygoMap object - referred to as "self"):
    # - "zygoread(self, filename)" - extracting header fields and measurement data from MetroPro .txt files
    # ------------------------------ note that the data is not parsed into variables yet, just returned in more useful format
    # - "crop(self, radius=0)" - (for "heights" array) remove data outwith the radius (no crop by default) and centre on the valid data
    # - "testf(self, *args)" - test/loss function, passed to optimisation algorithm within untilt()
    # - "untilt(self, array=None)" - tilt removal by finding a planar least-squares fit to describe the slope of the points, and subtracting
    #########
    # - "__str__(self)" - special method which tells Python how to display the object e.g. when calling print(ZygoMap)
    # ------------------- will print out a brief summary of the current object, depending on how it was created
    #########
    # - "__init__(self, filename=None, array=None, map1=None, map2=None, angle=None, flatten=True)"
    # -- special method which is always run first by Python when creating object
    # -- the main function of the class, initialises the object and performs the pre-processing steps using the methods listed above
    # -- any options given during e.g. "ZygoMap(option1=..., option2=...,)" get passed straight to __init__()
    
    def __init__(self, filename=None, array=None, mapname=None, map1=None, map2=None, angle=None, flatten=True):
        """
        Construct a `ZygoMap` object and pre-process with interpolation, cropping, and tilt-removal.
        
        Facilitates construction from three distinct sources:
            - Read-in data from a locally-stored `MetroPro` ASCII .txt file via the 'filename' parameter.
              Designed for specific formatting as detailed in the `MetroPro` manual.
              (See: MetroPro Reference Guide, Section 12, pgs 12-17.
                    https://www.seas.upenn.edu/~nanosop/documents/MetroProReferenceGuide0347_M.pdf)
            
            - Directly from a single pre-existing array source, via the 'array' parameter.
            
            - Indirectly from an array representing the interface height between two maps when 
              combined face-to-face to form a bond. This is performed via the `combinemaps`
              function, which also supplies details of the maps in the bond, 'map1' and 'map2',
              and the angle at which their bond profile was minimised.
              
        A number of pre-processing steps are performed during initialisation:
            - Surface measurements are stored in an array called 'heights'.
            - Original data is kept as a copy, in an array called 'heights0'.
            - Interpolation is performed to fill in any missing values,
              and stored in an array called 'heights1'.
            - Any tilt present in the data is removed to show the true variation of the heights.
            - The data is cropped and centred so that the array extends to frame only the valid area.
            - Key attributes are stored, including the 2 key surface height metrics,
              peak-to-valley height and root-mean-square height.
              
        Parameters:
        
        'filename' : str, optional
            - The local filepath of a `MetroPro` .txt data file of surface measurements.
            - If this is not supplied, then 'array' must be given to construct the object.
        'array' : numpy array, shape == (M,N), optional
            - Pre-existing array of height values on a 2D grid, to be processed
              and stored in 'heights'.
            - If 'array' is not None, then this will be used to construct the object,
              and 'filename' will be ignored whether it is given or not.
              
        'mapname' : str, optional
            - Name to be given as an identifier for the map object.
            - If 'filename' given, set by default to take the stem of the file (without .txt extension).
            - Otherwise, if 'map1' and 'map2' exist, set by default to join the individual
              map names with a "+" between them.
            - Otherwise, 'mapname' will be None. The user can supply their own names.
        'map1' : `ZygoMap` object, optional
            - Map object for a constituent map that has been used to construct the current one.
            - This is the "lower" map in the bond.
        'map2' : `ZygoMap` object, optional
            - Map object for a constituent map that has been used to construct the current one.
            - This is the "upper" map in the bond.
        'angle' : float, optional
            - The relative in-plane rotation angle which was applied to 'map2'
              to minimise the simulated bond profile.
        'flatten' : bool, optional
            - Flag to toggle the automatic tilt-removal process. Default is True.
            - If user is providing a pre-defined array, they may want to keep the data as-is,
              then flattening/tilt-removal can be turned off with `flatten=False`.
        
        Returns: None
        
        """
        
        #initialise ZygoMap object, process to remove tilts and store information about surface and/or from file header
        #2 ways to make ZygoMap object:
        # - 1) reading in header and data arrays from ASCII .txt file (MetroPro formatting - see reference guide)
        # - -  defines a map for a single component, as specified by file (use "filename" keyword)
        # - 2) supplying an array directly to be processed (using "array" keyword)
        # - -  particularly needed to store the interface map for combinations of other maps and their optimal angle
        # - -  also allows simple user-defined MxN arrays to be provided
        
        #initialise all variables, whether None or otherwise
        #they can all be checked for None elsewhere, but it's best to make sure they exist
        self.filename = filename
        self.mapname = mapname
        self.array = array
        self.map1 = map1
        self.map2 = map2
        self.maps = (self.map1, self.map2)
        self.angle = angle
        self.flatten = flatten
        
        if (filename is None) & (array is None):
            raise AttributeError("""Cannot construct map object - please explicitly supply either
                                 'filename' as a local file location or
                                 'array' as a 2D numpy array
                                 as arguments to begin processing.""")
            
        elif (filename is not None) & (array is not None):
            filename = None
            self.filename = None  # - possibly unnecessary - could change all checks after initial setting to self.[argument] instead of [argument]
            print("""Warning: Both 'filename' and 'array' arguments were provided. The initialisation will proceed using 'array' argument.""")
        
        if array is not None:
            if isinstance(array, np.ndarray):
                if (len(array.shape) == 2) & (0 not in array.shape):
                    self.heights = array.copy()
                else:
                    raise ValueError("Invalid array dimensions - NxM 2D array shape required.")
            else:
                raise AttributeError("""Could not process given argument. 'array' argument requires an NxM 2D numpy array.""")
                
            #allow map object to be created from scratch (i.e. make interface as a map object directly)
            #set info about interface's source maps and their combination
            #leave default as None, this is only applicable to interfaces created out of combinemaps
            #which provides the 2 filenames and optimised angle
            #test if this is a combination of maps (interface) or user-defined single map ("map1","map2","angle" do not apply)
            self.heights = array.copy()
        
        elif filename is not None:
            self.filename = filename
            if self.mapname is None:
                self.mapname = os.path.splitext(os.path.basename(self.filename))[0]
            
            #get header and data from file by user-defined function "zygoread" (change to class method ?)
            fields, data = self.zygoread(filename)

            #header extraction
            self.stringConstant = fields[0][0]
            chunk = fields[0][1].split()
            self.softwareType, self.majorVersion, self.minorVersion, self.bugVers = [int(n) for n in chunk]

            self.softwareDate = fields[1][0]

            chunk = fields[2][0].split()
            self.intensOriginX, self.intensOriginY, self.intensWidth, self.intensHeight, self.Nbuckets, self.intensRange = [int(n) for n in chunk]

            chunk = fields[2][1].split()
            self.phaseOriginX, self.phaseOriginY, self.phaseWidth, self.phaseHeight = [int(n) for n in chunk]

            self.comment = fields[3][0]

            self.partSerNum = fields[4][0]
            self.partNum = fields[5][0]

            chunk = fields[6][0].split()
            self.source = int(chunk.pop(0))  # - want 1st and last separately (they are int, rest are float)
            self.timeStamp = int(chunk.pop(-1))  # - use .pop(index) to separate the item from the list
            self.intfScaleFactor, self.wavelengthIn, self.numericAperture, self.obliquityFactor, self.magnification, self.cameraRes = [float(n) for n in chunk]

            chunk = fields[6][1].split()
            self.cameraWidth, self.cameraHeight, self.systemType, self.systemBoard, self.systemSerial, self.instrumentId = [int(n) for n in chunk]

            self.objectiveName = fields[7][0]

            #want both index 6 & 7 seperately, as they need to be floats
            #convert the rest to int as before
            chunk = fields[8][0].split()  # - looks messier but should use this throughout to reduce repeated splitting
            self.targetRange = float(chunk.pop(6))  # - remove item at index 6 and returns it (and modifies original list)
            self.lightLevel = float(chunk.pop(6))  # - do it again as the index 7 is now at index 6 in the modified list
            self.acquireMode, self.intensAvgs, self.PZTCal, self.PZTGain, self.PZTGainTolerance, self.AGC, self.minMod, self.minModPts = [int(n) for n in chunk]

            chunk = fields[8][1].split()
            self.disconFilter = float(chunk.pop(4))
            self.phaseRes, self.phaseAvgs, self.minimumAreaSize, self.disconAction, self.connectionOrder, self.removeTiltBias, self.dataSign, self.codeVType = [int(n) for n in chunk]

            self.subtractSysErr = int(fields[8][2])

            self.sysErrFile = fields[9][0]

            chunk = fields[10][0].split()
            self.refractiveIndex, self.partThickness = [float(n) for n in chunk]

            self.zoomDesc = fields[11][0]


            #extract intensity and phase data as numpy arrays (reshape to header parameters)
            self.intensitymap = np.array(data[0], dtype=float).reshape(self.intensHeight, self.intensWidth)
            self.phasemap = np.array(data[1], dtype=float).reshape(self.phaseHeight, self.phaseWidth)

            #handle invalid values (given in MetroPro manual)
            self.intensitymap[self.intensitymap >= 64512] = np.nan
            self.phasemap[self.phasemap >= 2147483640] = np.nan

            #create arrays in terms of number of waves, and height itself (in metres)
            #by given formula
            if self.phaseRes == 0:
                self.R = 4096
            elif self.phaseRes == 1:
                self.R = 32768
                
            #conversion formula from MetroPro manual
            self.waves = self.phasemap*(self.intfScaleFactor*self.obliquityFactor)/self.R
            self.heights = self.waves*self.wavelengthIn
             
            
        
        ########################
        ########################
        ##pre-processing maps
        
        #apply cropping first (user-defined radius ? or default ?)
        #self.cropped = self.crop(self.heights)
        #orderings/logistics of this needs fixed: which array is edited? when? what effect should user cropping give?
        #create a copy of the initial height array, this allows crop to act on those values and provide new self.heights
        #without data loss
        self.heights0 = self.heights.copy()
        self.heights1 = self.interpolate_grid()
        
        #grid points for use in some methods (?) (just using array i,j position index (can scale later))
        self.y0, self.x0 = np.indices(self.heights0.shape)  # - original grid defined by x0,y0 (needed for self.crop)
        self.y, self.x = np.indices(self.heights.shape)  # - variable grid to adapt to current condition of heights
        
        #adjust to centre of valid points (centre of surface)
        self.validrows, self.validcols = np.where(np.isfinite(self.heights1))
        self.centre = int(np.nanmean(self.validcols)), int(np.nanmean(self.validrows))
        self.r0 = max(self.validcols.max() - self.centre[0], self.validrows.max() - self.centre[1])  # - maximum (initial) radius from given data
        self.x0 -= self.centre[0]
        self.y0 -= self.centre[1]
        self.x -= self.centre[0]
        self.y -= self.centre[1]
        
#         self.cropped = self.heights[:]  # - slice notation actually still links the variables, need np.copy() instead
        #store initial attributes just so they are not missing at any point
        #they will be inaccurate initially (e.g. based on tilted map), but get updated via untilt()
        self.peak, self.valley = np.nanmax(self.heights1), np.nanmin(self.heights1)
        self.peakvalley = self.peak - self.valley
        self.rms = np.sqrt(np.nanmean(self.heights1**2))
        
                
        #remove tilt if present
        #note: added "flatten" keyword to allow user to use the map as read from file
        #still cropped to centre the view, but without removing any tilt
        #and also make sure to adjust valid rows & columns for this fully flattened array
        #use a 3rd heights array - a flattened but not cropped version
        #so will have: self.heights0 - the original data, self.heights1 - flattened version, self.heights - flattened and cropped to centre on valid area
#         self.heights1 = self.heights0.copy()
        if flatten:
            self.untilt(self.heights1)
            self.validrows, self.validcols = np.where(np.isfinite(self.heights1))  # - update the valid points for flattened map
            
            
            self.heights = self.heights1.copy()  # - keep heights1 stored; use heights as main array object for further uses
        
            self.crop(reflatten=True)
        else:
            self.crop(reflatten=False)
        
        return
    
    
    #use __str__ method to provide user summary on calling print(ZygoMap)
    #three possible paths, depending if single file object; interface created of two maps; or simply a user-defined array
    def __str__(self):
        """Printed representation of the current object depending on creation/attributes."""
        
        if self.filename is not None:
            if self.is_cropped:
                return ("ZygoMap object '{0}' for file: '{1}'."
                        "\ncropped to radius: {2}"
                        "\nPeak-to-valley height: {3:.1f} nm"
                        "\nRMS height: {4:.1f} nm").format(self.mapname, self.filename, self.radius, self.peakvalley*1e9, self.rms*1e9)
            else:
                return ("ZygoMap object '{0}' for file: '{1}'."
                        "\nPeak-to-valley height: {2:.1f} nm"
                        "\nRMS height: {3:.1f} nm").format(self.mapname, self.filename, self.peakvalley*1e9, self.rms*1e9)
        
        
        elif None not in (self.map1, self.map2, self.angle):
            if None not in (self.map1.mapname, self.map2.mapname):
                if self.is_cropped:
                    return ("ZygoMap interface object '{0}' for '{1}' & '{2}' combined at angle {3:.2f} degrees."
                            "\ncropped to radius: {4}"
                            "\nPeak-to-valley height of bond: {5:.1f} nm"
                            "\nRMS height of bond: {6:.1f} nm").format(self.mapname, self.map1.mapname, self.map2.mapname, self.angle,
                                                                       self.radius, self.peakvalley*1e9, self.rms*1e9)
                else:
                    return ("ZygoMap interface object '{0}' for '{1}' & '{2}' combined at angle {3:.2f} degrees."
                            "\nPeak-to-valley height of bond: {4:.1f} nm"
                            "\nRMS height of bond: {5:.1f} nm").format(self.mapname, self.map1.mapname, self.map2.mapname, self.angle,
                                                                       self.peakvalley*1e9, self.rms*1e9)
                
        elif self.mapname is not None:
            if self.is_cropped:
                return ("ZygoMap object for user-provided array '{0}'."
                        "\ncropped to radius: {1}"
                        "\nPeak-to-valley height: {2:.1f} nm"
                        "\nRMS height: {3:.1f} nm").format(self.mapname, self.radius, self.peakvalley*1e9,self.rms*1e9)
            else:
                return ("ZygoMap object for user-provided array '{0}'."
                        "\nPeak-to-valley height: {1:.1f} nm"
                        "\nRMS height: {2:.1f} nm").format(self.mapname, self.peakvalley*1e9,self.rms*1e9)
        
        else:
            if self.is_cropped:
                return ("ZygoMap object for user-provided array."
                        "\ncropped to radius: {0}"
                        "\nPeak-to-valley height: {1:.1f} nm"
                        "\nRMS height: {2:.1f} nm").format(self.radius, self.peakvalley*1e9,self.rms*1e9)
            else:
                return ("ZygoMap object for user-provided array."
                        "\nPeak-to-valley height: {0:.1f} nm"
                        "\nRMS height: {1:.1f} nm").format(self.peakvalley*1e9,self.rms*1e9)
        
        
#     @staticmethod
    def zygoread(self, filename):
        """
        For a given `MetroPro` text file, parse and separate the header fields from the measurement data.
        
        Works with the specific formatting of `MetroPro` ASCII data files.
        See: MetroPro Reference Guide, Section 12, pgs 12-17
             https://www.seas.upenn.edu/~nanosop/documents/MetroProReferenceGuide0347_M.pdf
             
        Parameters:
        
        'filename' : str
            - path to a locally-stored `MetroPro` file in ASCII text format.
            
        Returns:
        
        'fields' : list
            - Extracted values for header data.
            - Stored as a list of lists, with varying data types in each field.
            - Some elements of 'fields' need to be further split into the individual fields
              as detailed in the `MetroPro` manual.
        'data' : list
            - Extracted values for intensity data, followed by phase data.
            - Stored as a list of lists: 'data[0]' contains the intensity values as strings;
              'data[1]' contains the phase values as strings.
        
        """
        
        #works with specific ASCII format .txt files (documented in MetroPro reference guide)
        with open(filename, "r") as f:
            fstrings = f.read().split("\"")  # - split by quotation marks (easier to seperate string fields from data)

            fields = []
            data = []
            section = 0
            for elt in fstrings:
                if "#" in elt:  # - use this test to show end of header section, then switch to next section
                    section += 1
                    pass

                if section == 0:  # - processing header section
                    #multiple fields are stored within single strings, so need to split by newline to narrow down
                    #numbers stored within strings can be extracted afterwards
                    #excess artifacts can be filtered out using if (True) test on elements of split string
                    #fails on empty string, thus keeping only the relevant fields
                    elt = elt.split("\n")
                    values = [_ for _ in elt if _]  # - filter bad elements (e.g. "" which have no data and return False)
                    #test for non-empty lists (indicates no data was found in values list)
                    if values:
                        fields.append(values)

                elif section == 1:  # - move to data extraction for intensities and phases
                    #the initial split left the both datasets in a single string - separate by "#"
                    #split string containing a dataset then iterate through the resulting lines of 10
                    #append all values to a 'data' array, filter as before
                    #splitting by "#" will result in 2 lists stored in the overall 'data' list
                    #i.e. can extract: intensities = data[0], phases = data[1]
                    for line in elt.split("#"):
                        values = [_ for _ in line.split() if _]
                        if values:
                            data.append(values)
                        
        return fields, data
    
    
    def interpolate_grid(self, array=None, x_step=1, y_step=1):
        """
        Perform interpolation for missing data in a 2D array, and fit all values onto a specified grid.
        
        Using the `scipy.interpolate.griddata` function, the original positions and values
        in the array are used to estimate any missing data. A new grid is also supplied
        for the positions to re-fit the interpolated data to.
        
        This can be run using default parameters as initial filling of missing points by
        fitting to the same grid. Alternatively, the grid step can be increased/decreased
        to give a denser/sparser array.
        
        Parameters:
        
        'array' : numpy array, shape == (M,N), optional
            - the array containing values to be interpolated.
            - By default, this will be the object's main 'heights' array.
        'x_step' : float, optional
            - step size in the x-axis for the new grid.
            - Default is 1.0, which fits to the same x step size as the original data.
        'y_step' : float, optional
            - step size in the y-axis for the new grid.
            - Default is 1.0, which fits to the same y step size as the original data.
            
        Returns:
        
        'interped' : numpy array, shape == (M / y_step, N / x_step)
            - array of interpolated values on the new grid
            - if 'x_step' and 'y_step' are both 1.0, this has the same shape as 'array'.
        
        """
        
        #initial interpolation stage to cover any missing values
        #using scipy's griddata
        #the valid values are passed in, along with the new grid to fit to
        #this can be the exact same grid, or an up/down-scaled version using grid_step argument
        if array is None:
            array = self.heights0

        dims = array.shape
        x,y = np.indices(dims)
        vectarray = np.dstack([x,y,array])
        vectarray = vectarray.reshape(vectarray.size//3, 3)

        sample_xaxis = np.arange(0, dims[0], x_step)
        sample_yaxis = np.arange(0, dims[1], y_step)

        ax_dims = sample_xaxis.shape[0], sample_yaxis.shape[0]

        X = np.vstack(np.tile(sample_xaxis, ax_dims[1]).reshape(ax_dims[1], ax_dims[0])).T
        Y = np.vstack(np.tile(sample_yaxis, ax_dims[0]).reshape(ax_dims[0], ax_dims[1]))

        va = vectarray[np.isfinite(vectarray[:,2])]
        interped = scipy.interpolate.griddata((va[:,0], va[:,1]), va[:,2], (X,Y), method="cubic")

        return interped
    
    
    def crop(self, radius=0, reflatten=True):
        """
        Crop the object's surface data to within a given radius, and centre array to frame the valid data.
        
        Allows cropping of the data by a user-defined amount - either through an exact radius (if 'radius' > 0),
        or a relative trim from edge (if 'radius' < 0). A boolean mask array is created by testing if points in
        the array are inside or outside of the crop radius. Points outside are set to `NaN`, indicating invalid
        data.
        
        The array is sliced to remove any invalid points beyond the furthest rows/columns which have valid points.
        This centres on the data which is valid, reducing the overall array size also.
        
        NOTE: 'crop' will modify the current object, however data can be restored when `radius=0`.
        
        Parameters:
        
        'radius' : float, optional
            - The selected radius to crop array to. Can be positive, negative, or zero. Default is zero.
            - If 'radius' > 0, this indicates an exact crop radius.
              The cropped array will have axes of max. length 2 * radius (plus/minus 1).
            - If 'radius' < 0, this indicates a relative trim inwards from the edges of the data.
              The length of cropped array axes will depend on the input array dimensions.
            - If 'radius' == 0, then no cropping is applied - the full data will be kept or restored.
              `crop(radius=0)` is run during initialisation, only for centring and framing the valid data.
              
        Returns:
        
        'self' : `ZygoMap` object
            - The modified version of the current object.
        
        """        
        
        #allow user to crop to extract only data within some radius
        #most needed to avoid large edge effects (discontinuities)
        #use the (stored) centred x and y positions to check against radius
        #make a new cropped array where points outwith radius are set to nan
        #and "zoom in" to store only the array rows & columns within the valid range
        #will run during __init__(), with default radius = 0, so can avoid editing if radius is default
        #and only do if user chose a (non-zero) radius
        #thus only the centring of view by array slicing is performed (no need for separate functions)
        if radius < 0:
            radius = self.r0 - abs(radius)
        
        #set cropped array based on original state of heights (so not cropping multiple times and losing data)
#         cropped = self.heights0.copy()
        cropped = self.heights1.copy()
        self.is_cropped = False
        
        if radius != 0 and radius < self.r0:
            #if radius non-zero, we will be setting valid points to invalid (nan)
            #use centred x,y grid points for full data array to make a mask for points outside radius
            #then just change these points to nan (thus matching the pre-existing background)
            outsideR = self.x0**2 + self.y0**2 > radius**2
            cropped[outsideR] = np.nan
            
            self.is_cropped = True
            #could add a flag/callback here to automatically re-apply rotations after crops (otherwise advise user to do it)
            #e.g. if callback = True -> untilt()
        
        #now find the extreme bounds of valid points and slice the array to show only the data within
        validrows, validcols = np.where(np.isfinite(cropped))
        lft, rgt, upp, low = np.nanmin(validcols), np.nanmax(validcols), np.nanmin(validrows), np.nanmax(validrows)
        
        
        cropped = cropped[upp:low+1, lft:rgt+1]
        self.heights = cropped.copy()
        
        if reflatten:
            self.untilt(self.heights)
        
        #update valid positions, after rows/columns removed
        self.validrows, self.validcols = np.where(np.isfinite(self.heights))
        self.centre = int(np.nanmean(self.validcols)),int(np.nanmean(self.validrows))
        self.y, self.x = np.indices(self.heights.shape)
        self.x -= self.centre[0]
        self.y -= self.centre[1]
        
        if radius != 0 and radius < self.r0:
            self.radius = abs(radius)
        else:
            self.radius = self.r0
        
        return self
    
    
    def untilt(self, array=None):
        """
        Find the best-fit plane to surface height data, and subtract this to remove any slope/tilt in the data.
        
        Surface measurements often show tilts in the data which dominates the scaling so the variation can't be
        seen clearly. By finding a least-squares planar fit, then subtracting the plane's offset at each point,
        the variation of heights can be seen without interference from tilts.
        
        For the given array of heights, the `scipy.linalg.lstsq` function gets the least-squares minimised plane
        which represents the tilt in the data. The values of the tilt plane at each grid point are subtracted,
        and the array updated with the new values with tilt removed. Object attributes are updated for the new data.
        
        Parameters:
        
        'array' : numpy array, shape == (M,N), optional
            - array containing the surface measurements.
            - By default, uses the main array of the data, which is stored in the 'heights' attribute.
            
        Returns:
        
        'self' : `ZygoMap` object
            - The modified version of the current object.
        
        """
        
        #"array" argument left so normal or cropped maps can be used (i.e. self.heights vs self.cropped)
        #detect array=None to mean default self.heights
        if array is None:
            array = self.heights
        
        #fit plane to data with linear least squares
        dims = array.shape
        x,y = np.indices(dims)

        #reduce points to exclude nan's - scipy's lstsq can't handle them
        valid_mask = np.isfinite(array)

        #create a stack of [x,y,height] position vectors for the valid grid points
        vectarray = np.dstack([x[valid_mask], y[valid_mask], array[valid_mask]])
        vectarray = vectarray.reshape(vectarray.size//3, 3)

        #construct data to model with additional constant (1) for the regression line
        #will solve for x coefficients in matrix equation Ax = Z
        A = np.c_[vectarray[:,0], vectarray[:,1], np.ones(vectarray.shape[0])]

        #solve for coefficients of least squares minimised plane
        #i.e. there are three coefficients for 
        C, *_, = scipy.linalg.lstsq(A, vectarray[:,2])

        #recreate the fitted plane values
        Z = C[0] * x[valid_mask] + C[1] * y[valid_mask] + C[2]


        #return the array with the planar tilt heights taken away
        new_array = array.copy()
        new_array[valid_mask] = array[valid_mask] - Z
        
        #update parameters for flattened heights
        self.peak, self.valley = np.nanmax(new_array), np.nanmin(new_array) # - need to consider all rotated values, even if they are not going to be stored (outwith bounds)
        self.peakvalley = self.peak - self.valley
        self.rms = np.sqrt(np.nanmean(new_array**2))
        
        #update the stored heights array
        self.heights = new_array.copy()
        
        return self
    
    
    def plot(self, plot_type="2d"):
        """
        Create either a 2D or 3D plot representing the surface height values of the `ZygoMap`.
        
        Surface measurements are accessed via the 'heights' array attribute. This array is passed
        into `matplotlib` functions to generate an output figure. Plots will contain a brief title
        displaying basic information of the plotted object. 3D plots also mark the 'peak' and 'valley'
        positions.
        
        Parameters:
        
        'plot_type' : str, optional
            - String argument to select a plot type, either `plot_type="2d"` or `plot_type="3d"`. Default is "2d".
            
        Returns: None
        
        """

        current_cmap = cm.get_cmap("viridis")
        array = self.heights * 1e9

        if plot_type == "2d":
#             %matplotlib inline

            fig = plt.figure(figsize=(4,4), dpi=100)
            ax = plt.axes((0,0,1,1), fc="#FFE8F2")
            for side in ax.spines:
                ax.spines[side].set_edgecolor("gray")
                ax.spines[side].set_linewidth(0.5)
                #ax.spines[side].set_visible(False)

            plt.imshow(array, cmap=current_cmap, origin="lower")

            plt.xlabel("X", fontsize=14)
            plt.ylabel("Y", fontsize=14)
            for tick in ax.xaxis.get_ticklabels():
                tick.set_alpha(0.8)
                tick.set_fontsize(9)

            for tick in ax.yaxis.get_ticklabels():
                tick.set_alpha(0.8)
                tick.set_fontsize(9)

    #         cb_formatter = plt.ScalarFormatter(useMathText=True)
    #         cb_formatter.set_powerlimits((-9,-9))

            cb = plt.colorbar(pad=0.05, shrink=0.8, aspect=25, format=None)#, ticks=[valley, 0, peak])
            cb.ax.tick_params(color="b", length=5)
    #         cb.formatter.set_scientific(True)
    #         cb.formatter.set_powerlimits((-9,-9))
            cb.outline.set_edgecolor("gainsboro")
            cb.outline.set_linewidth(0.8)

    #         ax.xaxis.set_major_locator(plt.NullLocator())  # - removes all (or specifically major ?) axis ticks/labels
            ax.tick_params(axis="both", length=1, pad=4)
            
            if self.is_cropped:
                plt.title(("Surface Height Map for ZygoMap '{0}'"
                           "\ncropped to radius: {1}").format(self.mapname, self.radius), fontsize=10, pad=10)
            else:
                plt.title(("Surface Height Map for ZygoMap '{0}'").format(self.mapname), fontsize=10, pad=10)


        elif plot_type == "3d":
#             %matplotlib notebook

            fig = plt.figure(figsize=(9,9), dpi=100)
            ax = fig.add_subplot(projection="3d")
            x,y = np.indices(array.shape)
            peak,valley = np.nanmax(array), np.nanmin(array)
            peakxy,valleyxy = np.where(array == peak), np.where(array == valley)
            ax.set_zlim(valley,peak)

            current_cmap = cm.get_cmap("viridis")

            p = ax.plot_surface(x,y, array, vmin=valley, vmax=peak, cmap=current_cmap, ccount=10000, rcount=10000)
            ax.plot3D(*valleyxy, valley, "bx", zorder=10, ms=10)
            ax.plot3D(*peakxy, peak, "rx", zorder=10, ms=10)
            ax.text(valleyxy[0][0],valleyxy[1][0],valley, "  {0:.1f}nm".format(valley), zorder=10, size=9)
            ax.text(peakxy[0][0],peakxy[1][0],peak, "  {0:.1f}nm".format(peak), zorder=10, size=9)

            cb = plt.colorbar(p, pad=0.05, shrink=0.5, aspect=25)#, ticks=[valley, 0, peak])
            cb.ax.tick_params(color="b", length=5)
    #         cb.formatter.set_scientific(True)
    #         cb.formatter.set_powerlimits((-9,-9))
            cb.outline.set_edgecolor("b")
            cb.outline.set_linewidth(0.5)

    #         plt.ticklabel_format(axis="z", style="sci", scilimits=(-9,-9), useMathText=True)
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())

            plt.xlabel("Y")  # - since matrix i,j are used for axes, spatial x,y should swap
            plt.ylabel("X")
            ax.set_zlabel("Height (nm)" + "\n"*20)
            ax.zaxis._axinfo["label"]["ha"] = "right"
            ax.zaxis.set_rotate_label(False)
            ax.zaxis.label.set_rotation("-3")
            # ax.zaxis.set_label_coords(0,0)

            ax.view_init(elev=10, azim=-55)
            
            if self.is_cropped:
                plt.title(("3D Map of Surface Height Map for ZygoMap '{0}'"
                           "\ncropped to radius: {1}").format(self.mapname, self.radius), fontsize=10, pad=10)
            else:
                plt.title(("3D Map of Surface Height Map for ZygoMap '{0}'").format(self.mapname), fontsize=10, pad=10)


        plt.show()

        return
