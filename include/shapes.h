#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>

#include "geometry.h"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);

class Object
{
public:
    Object(): color(dis(gen), dis(gen), dis(gen)){}
    virtual ~Object(){}
    virtual bool intersect(const Vec3f &, const Vec3f &, float &) const = 0;
    virtual void getSurfaceData(const Vec3f &, Vec3f &, Vec2f &) const = 0; 
    Vec3f color;
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

class Sphere : public Object
{
private:
    /* data */
public:
    
    Sphere(const Vec3f &c, const float &r): raidus(r), radius2(r * r), center(raidus){}
    bool intersect(const Vec3f &orig, const Vec3f &dir, float &t)
    {
        float t0, t1;
        Vec3f L = center - orig;

        float tca = L.dotProduct(dir);
        if (tca < 0 ) return false;
        float d2 = L.dotProduct(L) - tca * tca;
        t0 = tca - sqrt(radius2 - d2);
        t1 = tca + sqrt(radius2 - d2);

        if(t0 > t1) std::swap(t0, t1);
        if(t0 < 0){
            t0 = t1;
            if(t0 < 0) return false;
        }

        t = t0;
        return true;
    }

    void getSurfaceDate(const Vec3f &Phit, Vec3f &Nhit, Vec2f &tex) const
    {
        Nhit = Phit - center;
        Nhit.normalize();

        tex.x = (1 + atan2(Nhit.z, Nhit.x) / M_PI) * 0.5;
        tex.y = acosf(Nhit.y) / M_PI;
    }
    float radius2;
    float raidus;
    Vec3f center;
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
    Point operator ()(const float& t)const
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