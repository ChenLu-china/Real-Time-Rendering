#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <memory>
#include <fstream>
#include <algorithm>

#include "geometry.h"

struct Point
{   Point() : x(0), y(0), z(0) {}
    Point(const float& value): x(value), y(value), z(value){}
    Point(const float& xx, const float& yy, const float& zz) : x(xx), y(yy), z(zz){}
    Point operator * (const Matrix44f &m) const
    {
        Point p;
       
         
        p.x     = m[0][0] * x + m[1][0] * y + m[2][0] * z + m[3][0];
        p.y     = m[0][1] * x + m[1][1] * y + m[2][1] * z + m[3][1];
        p.z     = m[0][2] * x + m[1][2] * y + m[2][2] * z + m[3][2];
        float w = m[0][3] * x + m[1][3] * y + m[2][3] * z + m[3][3];

        if (w != 1) {
            p.x /= w;
            p.y /= w;
            p.z /= w;
        }

        return p;
    }
    Point operator * (const Point& p) const
    { return Point(x * p.x, y * p.y, z * p.z); }

    Point operator + (const Vec3f& v) const
    { return Point(x + v.x, y + v.y, z + v.z); }

    Vec3f operator - (const Point& p) const
    { return Vec3f(x - p.x, y - p.y, z - p.z); }

    Point operator / (const Point& p) const
    { return Point(x / p.x, y / p.y, z / p.z); }

    float x, y, z;
};

struct Color
{
    Color(): r(0), g(0), b(0){}
    Color(float value) : r(value), g(value), b(value) {}
    Color(const float& rval, const float& gval, const float& bval) : r(rval), g(gval), b(bval) {}
    Color& operator += (const Color& c)
    { r += c.r, g += c.g, b += c.b; return *this; }
    Color operator * (const float& value) const
    { return Color(r * value, g * value, b * value); }
    Color operator + (const Color& c)
    { return Color(r + c.r, g + c.g, b + c.b); }

    float r, g, b;

};

struct Grid
{
    size_t baseResolution = 128;
    std::unique_ptr<float[]> densityData;
    Point bounds[2] {Point(-30), Point(30)};
    float operator ()(const int& xi, const int& yi, const int& zi) const
    {
        if(xi < 0 || xi > baseResolution - 1 ||
           yi < 0 || yi > baseResolution - 1 ||
           zi < 0 || zi > baseResolution - 1)
            return 0;
        return densityData[(zi * baseResolution + yi) * baseResolution + xi];
    }
};

struct Ray
{   
    Ray(const Point& orig, const Vec3f& dir): orig(orig), dir(dir) 
    {
        invdir = 1 / dir;


        sign[0] = (invdir.x < 0);
        sign[1] = (invdir.y < 0);
        sign[2] = (invdir.z < 0);
    }
    Point operator ()(const float& t)
    {
        return orig + t * dir;
    }

    Point orig;
    Vec3f dir, invdir;
    bool sign[3];
};

struct RenderContext
{
    float fov{ 45 };
    size_t width{ 640 }, height{ 480 };
    float frameAspectRatio;
    float focal;
    float pixelWidth;
    Color backgroundColor{ 0.572f, 0.772f, 0.921f };
};


void initialRenderContext(RenderContext& rc)
{
    rc.frameAspectRatio = rc.width / float(rc.height);
    rc.focal = tanf(M_PI / 180 * rc.fov * 0.5f);
    rc.pixelWidth = rc.focal / rc.width;
}

bool raybox(const Ray & ray, const Point bounds[2], float &tmin, float &tmax)
{    

    // calculate intersections between the orignial of rays and AABB Bounds.
    float txmin, txmax, tymin, tymax, tzmin, tzmax;
    txmin = bounds[ray.sign[0]].x - ray.orig.x;
    txmax = bounds[1 - ray.sign[0]].x - ray.orig.x;
    tymin = bounds[ray.sign[1]].y - ray.orig.y;
    tymax = bounds[1 - ray.sign[1]].y - ray.orig.y;

    txmin = txmin == 0 ? 0 : txmin * ray.invdir.x;
    txmax = txmax == 0 ? 0 : txmax * ray.invdir.x;
    tymin = tymin == 0 ? 0 : tymin * ray.invdir.y;
    tymax = tymax == 0 ? 0 : tymax * ray.invdir.y;

    if((txmin > tymax) || (tymin > txmax))  return false;

    tmin = tymin > txmin ? tymin : txmin;
    tmax = tymax > txmax ? txmax : tymax;

    // tmin = std::max(txmin, tymin);
    // tmax = std::min(txmax, tymax);

    tzmin = (bounds[ray.sign[0]].z - ray.orig.z);
    tzmax = (bounds[1 - ray.sign[1]].z - ray.orig.z);

    tzmin = tzmin == 0 ? 0 : tzmin * ray.invdir.z;
    tzmax = tzmax == 0 ? 0 : tzmax * ray.invdir.z;

    if((tmin < tzmax) || (tzmin > tmax)) return false;

    tmin = std::max(tzmin, tmin);
    tmax = std:: min(tzmax, tmax);

    return true;
}

float phaseHG()
{

}


//Function where the coordinate of the sample points are converted from
//world space to voxel space. We can use the these coordinates
float lookup(const Grid& grid, const Point& p)
{
    Vec3f grid_size = grid.bounds[1] - grid.bounds[0];
    Vec3f pLocal = p - grid.bounds[0];
    pLocal = pLocal / grid_size;
    Vec3f pVoxel = pLocal * grid.baseResolution;

    Vec3f pLattice(pVoxel.x - 0.5, pVoxel.y - 0.5, pVoxel.z - 0.5);

    int xi = static_cast<int>(std::floor(pLattice.x));
    int yi = static_cast<int>(std::floor(pLattice.y));
    int zi = static_cast<int>(std::floor(pLattice.z));

    //nearest neighbor seach 
    // return grid(xi, yi, zi);

    //trilinear interpolate
    float weight[3];
    float value = 0;

    for(int i=0; i < 2; ++i){
        weight[0] = 1 - std::abs(pLattice.x - (xi + i));
        for(int j = 0; j < 2; ++j){
            weight[1] = 1 - std::abs(pLattice.y - (yi + j));
            for (int k = 0; k < 2; ++k)
            {
                weight[2] = 1 - std::abs(pLattice.z - (zi + k));
                value += weight[0] * weight[1] * weight[2] * grid(xi + i, yi + j, zi + k);
            }
             
        }
    }
    return value;
}

void integrate(
    const Ray &ray,
    const float &tMin, const float &tMax,
    Color &L,
    float &T,
    const Grid& grid
)
{
    const float stepSize = 0.05;
    float sigma_a = 0.5;
    float sigma_s = 0.5;
    float sigma_t = sigma_a + sigma_s;
}

void trace(Ray &ray, 
           Color &L, 
           float& transmittance, 
           const RenderContext &rc, 
           Grid &grid)
{
    float tmin, tmax;
    if( raybox(ray, grid.bounds, tmin, tmax)){
        integrate(ray, tmin, tmax, L, transmittance, grid);
    }
}

void render(const size_t& frame)
{
    fprintf(stderr, "Rendering frame: %zu\n", frame);
}
