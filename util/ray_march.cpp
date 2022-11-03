
#include "include/geometry.h"
#include "include/shapes.h"

Color background{0.572f, 0.772f, 0.921f};

Color back_integrate(const Ray& ray, float& size_step, const Grid& grid)
{
    float sigma_a = 0.1;
    float step_size = 0.2;

    float Transmission = 1;
    Color result;
    const Object* hit_object = nullptr;

    float tMin, tMax;
    for(const auto& object : objects){
        if(object->intersect(ray, tMin, tMax)){
            hit_object = object.get();
        }
    }

    if(!hit_object)
    {
        return background;
    }

    int numStep = std::ceil((tMax - tMin) / step_size);
    step_size = (tMax - tMin) / numStep;

    Vec3f dirLight{ 0, 1, 0 };
    Color colorLight{1.3, 0.3, 0.9};

    for(u_int8_t i=0; i < numStep; ++i){
        float t = tMax - step_size * (i + 0.5);
        Point sample_pose = ray(t);

        float step_transmission = exp(-step_size * sigma_a);
        Transmission *= step_transmission;

        float lt;
        Ray lray(sample_pose, dirLight);
        if(hit_object->intersect(ray, lt)){
            float light_atteunation = exp(-lt * sigma_a);
            result += colorLight * light_atteunation * step_size; 
        }

        result *= step_transmission;
    }

    result = background * Transmission + result;
    return result;
}

Color forward_integrate(const Ray& ray, const Grid& grid)
{
    //...

    float step_size = 0.2;
    float sigma_a = 0.5;

    float tMin, tMax;
    const Object* hit_object = nullptr;

    float transparency = 1;
    Color result;

    int numStep = std::ceil((tMax - tMin) / step_size);
    step_size = (tMax - tMin) / numStep;

    Vec3f dirLight{ 0, 1, 0 };
    Color colorLight{1.3, 0.3, 0.9};

    for(uint8_t i = 0; i < numStep; ++i){
        float t = tMin + step_size * (i + 0.5);
        Point samplePose = ray(t);

        transparency *= exp(-step_size * sigma_a);

        float lt;
        if(hit_object->intersect(ray, lt)){
            float light_attenuation = exp(-lt * sigma_a);
            result += colorLight * light_attenuation * transparency;
        }
    }
    return background * transparency + result;
}