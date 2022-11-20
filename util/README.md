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
        <li>Scatterting: In normal, each volume can also emit light which we are just mentioning for the sake of completeness, just like volumetric object "reflecting" light. In code, it is represented as a rgb color.
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

<div>
    <ul>
        <li>In-Scattering：Imagine that we have a light emitted by a light source traveling through the volume that must happen absorption follow beer law, we need to know how many intensity remain after absorption coefficient.
    </ul>
</div>

Supplementry:
The content I mentioned above that some keywords followed the color finally obtain as a viewer, somehow, the energy of light has decrease when it travels to volume, which include out-scattering. Out-Scattering: This causes light that's not traveling towards the eye.

```bash
float light_intensity = 10;
float T = exp(-distance * volume->absorption_coefficient);
light_intensity_attenuation = T * light_intensity;
```

For a color we receviced along the particular eye/camera ray that is a combination of light coming from the background and light coming from the light source scattered towards the eye due to the in-scattering.

## Ray Marching Algorithm

A. algorithm intergrates income light along the ray due to in-scattering.
<div>
    Work Flow:
    <ul>
        <li>1. Find the value for t0 and t1, the points where the camera/eye ray enters and leaves the volume object
        <li>2. Divide the segment defined by t0-t1 into X number of smaller segments of identical size. Generally we do so by choosing what we call a step size
        <li>3. What you do next is "march" along the camera ray X times, starting from either t0 or t1 
        <li>4. Each time we take a step, we shoot a "light ray" starting from the middle of the step (our sample point) to the light. 是的
    </ul>
    Simply pseudocode shows in ray_march.cpp
</div>

```bash
L(x) = exp(-t1 * volume->absorption_coefficient) * step_size * light_color  
```

In real scene, the forward ray marching is good than backward ray marching, because we can know when the transimssion be 0 that mean there is no light pass through the current volume.

```bash

```
<<<<<<< HEAD

B. Choosing the Step Size
=======
    
## The Phase Function
<div align=center>    
$$L_i = \sigma_s \int_{S^2}p(x, w, w^{'})L(x, w^{'})\, dw^{'}$$
</div>
    
Where $L_i$ is the in-scattering contribution, $\sigma_s$ is the scattering coefficience, $x$ the sample position and $w$ the view direction, $w^{'}$ donates the light direction. $S^2$ which can also write as $\Omega_{4\pi}$ means that the in-scattering contribution can be calculated by light coming from all directions over the entire sphere. BRDFs that gather light over the hemisphere of directions instead.

n summary, the phase function tells you how much light is likely to be scattered towards the viewer (ω) for any particular incoming light direction (ω′)

Isotropic and Anisotropic

>>>>>>> 520ffad7294cf281698731d4ab0a5523137923da
