#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <random>

#include "geometry.h"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);

class Ray
{
public:
    Ray(const Vec3f &orig, const Vec3f &dir): orig(orig), dir(dir) {
        invdir = 1 / dir;
        sign[0] = dir.x;
        sign[1] = dir.y;
        sign[2] = dir.z;
    }

    Vec3f orig, dir;
    Vec3f invdir;
    int sign[3];
};

class AABBbox
{
public:
    AABBbox(const Vec3f &b0, const Vec3f &b1){bounds[0] = b0, bounds[1] = b1;}

    bool intersect(const Ray &r, float &t) const
    {
        float tmin, tmax, tymin, tymax, tzmin, tzmax;

        tmin = (bounds[0].x - r.orig.x) * r.invdir.x;
        tmax = (bounds[1].x - r.orig.x) * r.invdir.x;
        tymin = (bounds[0].y - r.orig.y) * r.invdir.y;
        tymax = (bounds[1].y - r.orig.y) * r.invdir.y;
        
        if((tmin > tymax) || (tymin > tmax))
            return false;
        
        if (tmin < tymin) tmin = tymin;
        if (tmax > tymax) tmax = tymax; 

        tzmin = (bounds[0].z - r.orig.z) * r.invdir.z;
        tzmax = (bounds[1].z - r.orig.z) * r.invdir.z;
        
        if((tmin > tzmax) || (tzmin > tmax))
            return false;

        if(tmin < tzmin) tmin = tzmin;
        if(tmax > tzmax) tmax = tzmax;

        t = tmin;

        if(t < 0){
            t = tmax;
            if(t < 0 ) return false;
        }

        return true;
    }

    Vec3f bounds[2];
};


int main(int argc, char **argv)
{
    AABBbox box(Vec3f(-1), Vec3f(1));
    gen.seed(0);
    for(u_int32_t i = 0; i < 16; ++i){
        Vec3f randDir(2 * dis(gen) - 1, 2 * dis(gen) - 1, 2 * dis(gen) - 1);
        Ray ray(Vec3f(0), randDir);
        float t;
        if (box.intersect(ray, t))
        {
            Vec3f Phit = ray.orig + ray.dir * t;
            std::cerr << ray.orig << " " << Phit << std::endl;
        }
    }
    return 0;
}
