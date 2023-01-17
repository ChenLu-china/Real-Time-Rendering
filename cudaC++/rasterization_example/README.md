
# 3D Space Rasterization:

1. We input Cartesian Coordinate (X, Y, Z) to Unit Cube Coordinate (X_obj, Y_obj, Z_obj).  

2. Change cube center to left-buttom-back corner.  

3. Inference that the offset is 0.5 * (1 - center/ radius), the scaling is 0.5 / radius.  

4. As we pre-set grid size to change offset and scaling to rasteriza coordinate.  

<div>
  <image>
</div>
