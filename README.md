# P4Project - Analysing Zygo&reg; Measurement Data
## (Development Branch)
Developing Python programming tools to analyse surface maps from Zygo&reg; interferometers.

## `ZygoMap` Objects for Surface Analysis:
Using the `ZygoMap` class, users can store and represent surface measurement data captured via interferometry techniques alongside Zygo's proprietary software `MetroPro`.

`ZygoMap` objects are specifically designed to read-in data from `MetroPro` ASCII data files. They can also work with user-provided arrays, giving specific control to users over their analysis.

<p align="center">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_A1.png" width="300">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_A2.png" width="300">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_C1.png" width="316">
</p>
<p align="center">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_C2.png" width="300">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_M1.png" width="300">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_M2.png" width="300">
</p>

## Bond Interface Maps:
Create simulated bonds between two surfaces, and study maps of their bond interface. `ZygoMap` objects can be combined to simulate the face-to-face bond of a pair of maps.

A pair of maps can be passed to `combinemaps`, which will automatically simulate and optimise their bond, returning a new `ZygoMap` representing the height gaps of the bond interface.

With the `comparebonds` auto-comparison function, we can quickly compare surface profiles and predict the best pairs to bond.

<p align="center">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_c1m1_3d.png" width="900">
</p>

## `ZygoMap` functionality:
See `ZygoMap_demo.ipynb` for in-depth walkthrough of functionality and an example workflow.

#### Tilt Removal

<p align="center">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_C1_tilted.png" width="420">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_C1_untilted.png" width="400">
</p>

#### Cropping

<p align="center">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_C1_uncropped.png" width="400">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_C1_cropped.png" width="400">
</p>

#### Upscaling / Grid Interpolation

<p align="center">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_c1m1_2d.png" width="400">
  <img src="https://github.com/scottgallacher-3/P4Project/blob/main/examples/new_c1m1_2d_upscaled.png" width="400">
</p>
