# Create curvilinear coordinate systems for complex anatomy

<p align="center">
   <img width="400" src="https://github.com/mmiv-center/HeatEquation/blob/master/img/screenshot.png?raw=true">
   <img width="400" src="https://github.com/mmiv-center/HeatEquation/blob/master/img/snapshot_volume.png?raw=true">
</p>

This module creates curvilinear coordinates from volumetric label fields. It simulates the heat equation and exports the temperature (potential) field and the gradient of the potential field (tangent unit vector). The gradient field represents the directions of geodesic lines connecting the fixed temperature regions.

This module can be used to compute the shorted path between points of the ventricles and points of the white/gray matter border using structural information (white matter) only. It can also be used to sample the white matter at a given distance perpendicular to the gray/white matter border.

## Usage

The HeatEquation program can be called with the following arguments: 
```
 Command tags: 
   [ -t --temperature-label-pairs < temperature > ]
      = Specify the temperature per label as <N> [<label value> <temperature>], with N the number of label and temperature values such as in '-t 4 0 0.0 1 100.0'. A label that is not specified will be assumed to be variable and used for the computation.
   [ -i < iterations > ]
      = Specify the number of iterations (default 1) the code is run. Suggested is a large number like 2000.
   [ -s < supersample > ]
      = Specify the number up-sampling steps using nearest neighboor interpolation (0).
   [ -q < quantize > ]
      = Quantize the output into N different regions of equal volume.
   [ -n ]
      = Export the unit normal vector and the unit binormal vector per voxel (gradient is the tangent vector)
 Command fields: 
   < infile > 
      = Input mask (nifti or other file format understood by itk)
   < outdir > 
      = Output mask directory
```

In general the convergence of this code is very slow. Use plenty of iterations to fill in values. The default temperature value for regions to estimate is 0, therefore its advisable to select temperature values around 0 for the simulation of the heat transfer. Complex structures should have a sufficient high resolution. You can upsample the input data by '-s <factor>' to improve the result in narrow channels. The option '-q X' allows the volume of the free-temperature labels to be segmented into equal volume regions.
```
./HeatEquation -s 3 -n -i 1000 -q 3 data/data_1wm_2ven_3gm.nii /tmp/ -t 4 3 -100 2 100.99
```

The simulation could also be carried out by propagating front algorithms. Those can be implemented more efficiently and are faster converging and should produce similar results to the trivial implementation used in this module.

The number of fixed temperature zones is adjustable. One can therefore specify several 'intermediate' regions with intermediate temperatures.

## Output

The output folder will contain several volumes after the computation is finished. There is a temperature field exported as a floating point nifti file, a gradient field exported as an .nrrd volume and, if specified, a quantized label field (option -q) where each region has equal volume (indexed from low to high temperature).

If the '-n' option is specified two more vector fields are exported. The gradient field vector is assumed as the tangent vector of a surface at each voxel. The normal unit vector perpendicular to the tangent vector as well as the unit binormal vector create a local frame of reference at each voxel location.
