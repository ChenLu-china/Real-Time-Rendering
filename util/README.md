## Camera
<div>
    <ul>
        <li>from to camera
    </ul>
</div>

```bash
```

## Intersection
<div>
    <ul>
        <li>Nearnest Interpolate Method
        <li>Linear Interpolate Method
        <li>Bilinear Interpolate Method
        <li>Trilinear Interpolate Method
    </ul>
<div>

```bash
```

## Lighting
<div>
    <ul>
        <li>Absorption：Some of it will be absorbed by volume as it pass through it.
        <li>Internal Transmittance: The amount of light being absorbed by the volume as it travels through it. The value from 0(the volume block all the light) to 1(well, it is a vacuum so all light is transmitted).
        <li>Beer Lambert Law: The concept of density is expressed in terms of absorbed coefficient. The denser the volume, the higher absorption coefficient. The equation expressed in T = exp(-distance * sigma_a).
        <li>Mean Free Path: 
        <li>Scatteting: In normal, each volume can also emit light which we are just mentioning for the sake of completeness, just like volumetric object "reflecting" light.
    </ul>
</div>

```bash
Vector background_color = {r, g, b};
float sigma_a = 0.1;
float distance = 10;
float T = exp(-distance * sigma_a);
Vector volume_color = {vr, vg, vb};
Vector final_color = T * background_color + (1 - T) * volme_color;
```  