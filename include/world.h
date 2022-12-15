#include "geometry.h" 

// we define a 3D Grid by res ** 3 
class SphereGrid
{

public:
    SphereGrid(const int reso): reso(reso), gsz(reso, reso, reso){}

private:
    int reso;
    Vec3i gsz;
    Vec3f center;
    Vec3f radius;
    Vec3f _offset;
    Vec3f _scaling;
};