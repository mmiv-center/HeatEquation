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
   [ -f ]
      = Specify the file format for the vector fields. Can be -f nrrd (default) or -f vtk or another format supported by itk.      
 Command fields: 
   < infile > 
      = Input mask (nifti or other file format understood by itk)
   < outdir > 
      = Output mask directory
```

In general the convergence of this code is very slow. Use plenty of iterations to fill in values. The default temperature value for regions to estimate is 0, therefore its advisable to select temperature values around 0 for the simulation of the heat transfer. Complex structures should have a sufficient high resolution. You can upsample the input data by '-s <factor>' to improve the result in narrow channels. The option '-q X' allows the volume of the free-temperature labels to be segmented into equal volume regions.
```
./HeatEquation -s 1 -f vtk -n -i 1000 -q 3 data/data_1wm_2ven_3gm.nii /tmp/ -t 4 3 -100 2 100.99
```

The simulation could also be carried out by propagating front algorithms. Those can be implemented more efficiently and are faster converging and should produce similar results to the trivial implementation used in this module.

The number of fixed temperature zones is adjustable. One can therefore specify several 'intermediate' regions with intermediate temperatures.

## Output

Given a call such as:
```
./HeatEquation -s 3 -n -i 10 -q 3 data/test.nii data/output/ -t 4 1 -100 3 100
```
the resulting directory structure after the call is:
```
.
├── output
│   ├── test_gradient.nrrd
│   ├── test_gradient_binormal.nrrd
│   ├── test_gradient_normal.nrrd
│   ├── test_temperature.json
│   ├── test_temperature.nii
│   └── test_temperature_quantized.nii
└── test.nii
```

The output folder will contain several volumes after the computation is finished. There is a temperature field exported as a floating point nifti file, a gradient field exported as an .nrrd volume and, if specified, a quantized label field (option -q) where each region has equal volume (indexed from low to high temperature).

If the '-n' option is specified two more vector fields are exported. The gradient field vector is assumed as the tangent vector of a surface at each voxel. The normal unit vector perpendicular to the tangent vector as well as the unit binormal vector create a local frame of reference at each voxel location.

Together with the output fields the program also exports a json file which contains some statistics about the run for provenance:
```
{
    "OutputSize": [
        384,
        384,
        261
    ],
    "OutputSpacing": [
        0.5228762626647949,
        0.5228762626647949,
        0.5279547373453776
    ],
    "SuperSamplingFactor": 3.0,
    "command_line": [
        "./HeatEquation",
        "-s",
        "3",
        "-i",
        "5",
        "-q",
        "3",
        "-c",
        "data/output/test.nii_temperature.nii",
        "data/test.nii.gz",
        "data/output/",
        "-t",
        "4",
        "1",
        "1",
        "3",
        "2"
    ],
    "output_gradient": "data/output//test.nii_gradient.nrrd",
    "output_temperature": "data/output//test.nii_temperature.nii",
    "output_temperature_quantized": "data/output//test.nii_temperature_quantized.nii",
    "output_temperature_quantized_thresholds": [
        1.2814070351758793,
        1.5326633165829144
    ],
    "temperature_range_specified": [
        1.0,
        2.0
    ],
    "temperatures": [
        {
            "label": 1,
            "temperature": 1.0
        },
        {
            "label": 3,
            "temperature": 2.0
        }
    ]
}
```

## Multi-stage processing for speed improvement

In order to speed up the convergence the program can be called in multi-resolution stages. The low resolution steps converge fast and are used for the larger resolution steps as initial temperature fields. Here an example that first calculates the temperature field at a quarter of the original resolution. The following steps use the previous temperature field as initialization until a field is reached that has 3 times the resolution of the input field.
```
./HeatEquation -s 0.25 -i 5000 data/test.nii.gz data/output/ -t 4 1 1 3 2
./HeatEquation -s 0.5 -i 2000 -c data/output/test.nii_temperature.nii data/test.nii.gz data/output/ -t 4 1 1 3 2
./HeatEquation -s 1 -i 500 -c data/output/test.nii_temperature.nii data/test.nii.gz data/output/ -t 4 1 1 3 2
./HeatEquation -s 3 -i 500 -c data/output/test.nii_temperature.nii -n data/test.nii.gz data/output/ -t 4 1 1 3 2
```

## Build

Have ITK installed and cmake (tested with itk 5.0.0, cmake 3.13) and:
```
cmake .
make
```
to get the executable.

Alternatively use the provided docker file to build itk and the module:
```
docker build  -t heatequation -f Dockerfile .
docker run --rm -it heatequation -h
```

## Todo

As we can now measure distances in space we could measure the distances between every pair of regions of interest. This would result in an explosion of potential fields but those could be managed using a lower number of weighted basis sets (PCA, but wavelets have local support). It would be interesting to compare those directions (tangent unit vectors of the potential field) with actual fiber tracts in the brain as calculated from DTI. For some pairs of regions this could fit well, for other it might not - which onces? Using all the different tangent fields we would be able to recover an upper bounds on the number of crossing fibers in the brain (given a fixed number of regions of interest).

This model assumes an infinite load of the fibers per voxel, or at least a load that grows with the cube of the number of regions of interest. What if the fiber bundle density of a voxel is limited? In that world the potential fields would have to interact. A voxel would have to hinder the distribution of heat as its load increases. The load would be the proportion of tangent fields in its volume occupied given all the region to region connections.

A reason for such a load could be that support cells can only keep a given number of axons happy per volume element (thickness of the myelin sheet). Upper bound might be in the corpus callosum. Question, what is the fiber density as measured by histology in the different regions of the white matter?

We can add a sparsness constrain on the potential field per pair-wise connection. Most of the time the gradient of the potential field should be 0. This would emulate a preference for tight tracks. It would be good to know if this is really required.

