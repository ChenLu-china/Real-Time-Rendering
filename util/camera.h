#include "geometry.h"


template<typename T>
class Camera
{
public:
    Camera(): from(Vec3f(1)), to(Vec3f(0)){}
    Camera(const Vec3f& f, Vec3f& to): from(from), to(to)
    {
        Vec3f forward = from - to;
        forward.normalize();

        Vec3f tmp(0, 1, 0);
        Vec3f right = tmp.crossProduct(forward);
        Vec3f up = forward.crossProduct(right);

        //set the 4x4 matrix
        //row1: replace the first three coefficients of the row with the coordinates of the right vector
        //row2: replace the first three coefficients of the row with the coordinates of the up vector
        //row3: replace the first three coefficients of the row with the coordinates of the forward vector
        //row4: replace the first three coefficients of the row with the coordinates of the from point
        cam2world[0][0] = right.x, cam2world[0][1] = right.y, cam2world[0][2] = right.z;
        cam2world[1][0] = up.x, cam2world[1][1] = up.y, cam2world[1][2] = up.z;
        cam2world[2][0] = forward.x, cam2world[2][1] = right.y, cam2world[2][2] = right.z;
        cam2world[3][0] = from.x, cam2world[3][1] = from.y, cam2world[3][2] = from.z;     
    }

    Matrix44f cam2world;
    Vec3f from, to;

};
