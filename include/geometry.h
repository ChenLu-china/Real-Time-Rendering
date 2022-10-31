#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cmath>

template<typename T>
class Vec2
{
public:
    Vec2(): x(T(0)), y(T(0)){}
    Vec2(T xx) : x(xx), y(xx){}
    Vec2(T xx, T yy) : x(xx), y(yy){}
    Vec2 operator + (const Vec2 &v) const 
    { return Vec2(x + v.x, y + v.y); }
    Vec2 operator - (const Vec2 &v) const 
    { return Vec2(x - v.x, y - v.y); }
    Vec2 operator - () const 
    { return Vec2(-x, -y); }
    Vec2 operator * (const T &r) const 
    { return Vec2(x * r, y * r); }
    Vec2 operator / (const T &r) const 
    { return Vec2(x / r, y / r); } 
    Vec2& operator /= (const T &r) 
    { x /= r, y /= r; return *this; }
    Vec2& operator *= (const T &r) 
    { x *= r, y *= r; return *this; }

    friend std::ostream& operator << (std::ostream &s, const Vec2<T> &v) 
    { 
        return s << '[' << v.x << ' ' << v.y << ']'; 
    } 
    friend Vec2 operator * (const T &r, const Vec2<T> &v) 
    { return Vec2(v.x * r, v.y * r); } 
    T x, y;
};

typedef Vec2<float> Vec2f;

template<typename T>
class Vec3
{
private:
    /* data */
public:

    Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
    Vec3(T xx) : x(xx), y(xx), z(xx) {}
    Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz){}
    Vec3 operator + (const Vec3 &v) const 
    { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator - (const Vec3 &v) const 
    { return Vec3(x - v.x, y - v.y, z - v.z); } 
    Vec3 operator - () const 
    { return Vec3(-x, -y, -z); }
    Vec3 operator * (const T &r) const 
    { return Vec3(x * r, y * r, z * r); }
    T dotProduct(const Vec3<T> &v) const 
    { return x * v.x + y * v.y + z * v.z; }
    Vec3 operator / (const Vec3& v)
    { return Vec3<T>(x / v.x, y / v.y, z / v.z); }
    Vec3& operator /= (const T &r) 
    { x /= r, y /= r, z /= r; return *this; }
    Vec3& operator *= (const T &r) 
    { x *= r, y *= r, z *= r; return *this; }
    Vec3 crossProduct(const Vec3<T> &v) const 
    { return Vec3<T>(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    T norm() const 
    { return x * x + y * y + z * z; }
    T length() const 
    { return sqrt(norm()); }
    
    const T& operator [] (uint8_t i) const { return (&x)[i]; }
    T& operator [] (uint8_t i) { return (&x)[i]; } 
    
    Vec3& normalize(){
        T n = norm();
        if(n > 0){
            T factor = 1 / sqrt(n);
            x *= factor, y *= factor, z *= factor;
        }
        return *this;
    }

    friend Vec3 operator * (const T &r, const Vec3 &v) 
    { return Vec3<T>(v.x * r, v.y * r, v.z * r); } 
    friend Vec3 operator / (const T &r, const Vec3 &v) 
    { return Vec3<T>(r / v.x, r / v.y, r / v.z); } 
    
    friend std::ostream& operator << (std::ostream &s, const Vec3<T> &v) 
    { 
        return s << '[' << v.x << ' ' << v.y << ' ' << v.z << ']'; 
    }

    T x, y, z;
};
//声明类的使用
typedef Vec3<float> Vec3f;
typedef Vec3<int> Vec3i;
 

template<typename T>
class Matrix44
{
public:
    T x[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};

    Matrix44() {}

    Matrix44 (T a, T b, T c, T d, T e, T f, T g, T h, 
              T i, T j, T k, T l, T m, T n, T o, T p) 
    { 
        x[0][0] = a; 
        x[0][1] = b; 
        x[0][2] = c; 
        x[0][3] = d; 
        x[1][0] = e; 
        x[1][1] = f; 
        x[1][2] = g; 
        x[1][3] = h; 
        x[2][0] = i; 
        x[2][1] = j; 
        x[2][2] = k; 
        x[2][3] = l; 
        x[3][0] = m; 
        x[3][1] = n; 
        x[3][2] = o; 
        x[3][3] = p; 
    }
    const T* operator [] (uint8_t i) const { return x[i]; } 
    T* operator [] (uint8_t i) { return x[i]; }

    Matrix44 operator * (const Matrix44& v) const
    {
        Matrix44 tmp;
        multiply(*this, v, tmp);

    }

    static void multiply(const Matrix44<T> &a, const Matrix44 &b, Matrix44 &c)
    {   
        // for (uint8_t i=0; i < 4; ++i){
        //     for(u_int8_t, j=0; j < 4; ++j){
        //         c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + 
        //             a[i][2] * b[2][j] + a[i][3] * b[3][j];
        //     }
        // }
        const T * __restrict ap = &a.x[0][0]; 
        const T * __restrict bp = &b.x[0][0]; 
        T * __restrict cp = &c.x[0][0]; 
 
        T a0, a1, a2, a3; 
 
        a0 = ap[0]; 
        a1 = ap[1]; 
        a2 = ap[2]; 
        a3 = ap[3]; 
 
        cp[0]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12]; 
        cp[1]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13]; 
        cp[2]  = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14]; 
        cp[3]  = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15]; 
 
        a0 = ap[4]; 
        a1 = ap[5]; 
        a2 = ap[6]; 
        a3 = ap[7]; 
 
        cp[4]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12]; 
        cp[5]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13]; 
        cp[6]  = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14]; 
        cp[7]  = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15]; 
 
        a0 = ap[8]; 
        a1 = ap[9]; 
        a2 = ap[10]; 
        a3 = ap[11]; 
 
        cp[8]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12]; 
        cp[9]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13]; 
        cp[10] = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14]; 
        cp[11] = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15]; 
 
        a0 = ap[12]; 
        a1 = ap[13]; 
        a2 = ap[14]; 
        a3 = ap[15]; 
 
        cp[12] = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12]; 
        cp[13] = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13]; 
        cp[14] = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14]; 
        cp[15] = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15]; 
    }

    Matrix44 transposed() const
    {   
        // Matrix44 t; 
        // for (uint8_t i = 0; i < 4; ++i) { 
        //     for (uint8_t j = 0; j < 4; ++j) { 
        //         t[i][j] = x[j][i]; 
        //     } 
        // } 
 
        // return t;

        return Matrix44(x[0][0], x[1][0], x[2][0], x[3][0], 
                        x[0][1], x[1][1], x[2][1], x[3][1], 
                        x[0][2], x[1][2], x[2][2], x[3][2], 
                        x[0][3], x[1][3], x[2][3], x[3][3]);
    }
    
    Matrix44& transpose () 
    { 
        Matrix44 tmp(x[0][0], 
                x[1][0], 
                x[2][0], 
                x[3][0], 
                x[0][1], 
                x[1][1], 
                x[2][1], 
                x[3][1], 
                x[0][2], 
                x[1][2], 
                x[2][2], 
                x[3][2], 
                x[0][3], 
                x[1][3], 
                x[2][3], 
                x[3][3]); 
        *this = tmp; 
 
        return *this; 
    }

    template<typename S> 
    void multVecMatrix(const Vec3<S> &src, Vec3<S> &dst) const 
    { 
        S a, b, c, w; 

        a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0] + x[3][0]; 
        b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1] + x[3][1]; 
        c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2] + x[3][2]; 
        w = src[0] * x[0][3] + src[1] * x[1][3] + src[2] * x[2][3] + x[3][3]; 

        dst.x = a / w; 
        dst.y = b / w; 
        dst.z = c / w; 
    } 

    template<typename S> 
    void multDirMatrix(const Vec3<S> &src, Vec3<S> &dst) const 
    { 
        S a, b, c; 
 
        a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0]; 
        b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1]; 
        c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2]; 
 
        dst.x = a; 
        dst.y = b; 
        dst.z = c; 
    }

    Matrix44 inverse() 
    { 
        int i, j, k; 
        Matrix44 s; 
        Matrix44 t (*this); 
 
        // Forward elimination
        for (i = 0; i < 3 ; i++) { 
            int pivot = i; 
 
            T pivotsize = t[i][i]; 
 
            if (pivotsize < 0) 
                pivotsize = -pivotsize; 
 
                for (j = i + 1; j < 4; j++) { 
                    T tmp = t[j][i]; 
 
                    if (tmp < 0) 
                        tmp = -tmp; 
 
                        if (tmp > pivotsize) { 
                            pivot = j; 
                            pivotsize = tmp; 
                        } 
                } 
 
            if (pivotsize == 0) { 
                // Cannot invert singular matrix
                return Matrix44(); 
            } 
 
            if (pivot != i) { 
                for (j = 0; j < 4; j++) { 
                    T tmp; 
 
                    tmp = t[i][j]; 
                    t[i][j] = t[pivot][j]; 
                    t[pivot][j] = tmp; 
 
                    tmp = s[i][j]; 
                    s[i][j] = s[pivot][j]; 
                    s[pivot][j] = tmp; 
                } 
            } 
 
            for (j = i + 1; j < 4; j++) { 
                T f = t[j][i] / t[i][i]; 
 
                for (k = 0; k < 4; k++) { 
                    t[j][k] -= f * t[i][k]; 
                    s[j][k] -= f * s[i][k]; 
                } 
            } 
        } 
 
        // Backward substitution
        for (i = 3; i >= 0; --i) { 
            T f; 
 
            if ((f = t[i][i]) == 0) { 
                // Cannot invert singular matrix
                return Matrix44(); 
            } 
 
            for (j = 0; j < 4; j++) { 
                t[i][j] /= f; 
                s[i][j] /= f; 
            } 
 
            for (j = 0; j < i; j++) { 
                f = t[j][i]; 
 
                for (k = 0; k < 4; k++) { 
                    t[j][k] -= f * t[i][k]; 
                    s[j][k] -= f * s[i][k]; 
                } 
            } 
        } 
 
        return s; 
    }
    
    const Matrix44<T>& invert() 
    { 
        *this = inverse(); 
        return *this; 
    }
    
    friend std::ostream& operator << (std::ostream &s, const Matrix44 &m) 
    { 
        std::ios_base::fmtflags oldFlags = s.flags(); 
        int width = 12;  //total with of the displayed number 
        s.precision(5);  //control the number of displayed decimals 
        s.setf (std::ios_base::fixed); 
 
        s << "(" << std::setw (width) << m[0][0] << 
             " " << std::setw (width) << m[0][1] << 
             " " << std::setw (width) << m[0][2] << 
             " " << std::setw (width) << m[0][3] << "\n" << 
 
             " " << std::setw (width) << m[1][0] << 
             " " << std::setw (width) << m[1][1] << 
             " " << std::setw (width) << m[1][2] << 
             " " << std::setw (width) << m[1][3] << "\n" << 
 
             " " << std::setw (width) << m[2][0] << 
             " " << std::setw (width) << m[2][1] << 
             " " << std::setw (width) << m[2][2] << 
             " " << std::setw (width) << m[2][3] << "\n" << 
 
             " " << std::setw (width) << m[3][0] << 
             " " << std::setw (width) << m[3][1] << 
             " " << std::setw (width) << m[3][2] << 
             " " << std::setw (width) << m[3][3] << ")\n"; 
 
        s.flags (oldFlags); 
        return s; 
    }
};

typedef Matrix44<float> Matrix44f;



