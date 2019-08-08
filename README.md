# Create curvilinear coordinate systems for complex anatomy

This module creates curvilinear coordinates from volumetric label fields. It simulates the heat equation and exports the temperature potential field and the gradient of the potential field. The gradient field represents the directions of the geodesic lines for each volume element.

This module can be used to compute the shorted path between points of the ventricles and points of the white/gray matter border using structural information (white matter) only.

## Usage

```
./HeatEquation -i 10 data/data_1wm_2ven_3gm.nii /tmp/ -t 4 3 0 2 100.99
```