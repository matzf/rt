#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <xmmintrin.h>
#include <optional>
#include <algorithm>
#include <memory>
#include <vector>

static void writeImage(int *idImage, float *depthImage, int width, int height, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror(filename);
        exit(1);
    }

    const float max_depth = *std::max_element(depthImage, depthImage + width * height);

    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // use the bits from the object id of the hit object to make a
            // random color
            int id = idImage[y * width + x];
            unsigned char r = 0, g = 0, b = 0;

            for (int i = 0; i < 8; ++i) {
                // extract bit 3*i for red, 3*i+1 for green, 3*i+2 for blue
                int rbit = (id & (1 << (3 * i))) >> (3 * i);
                int gbit = (id & (1 << (3 * i + 1))) >> (3 * i + 1);
                int bbit = (id & (1 << (3 * i + 2))) >> (3 * i + 2);
                // and then set the bits of the colors starting from the
                // high bits...
                r |= rbit << (7 - i);
                g |= gbit << (7 - i);
                b |= bbit << (7 - i);
            }
            float d = depthImage[y * width + x] / max_depth;
            r *= d;
            g *= d;
            b *= d;
            fputc(r, f);
            fputc(g, f);
            fputc(b, f);
        }
    }
    fclose(f);
    printf("Wrote image file %s\n", filename);
}

static float rsqrt(float f) {
  return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_load_ss(&f)));
}

static float sqrt(float f) {
  return _mm_cvtss_f32(_mm_sqrt_ss(_mm_load_ss(&f)));
}

static float sq(float f) {
  return f*f;
}

struct alignas(16) Vec3f {
  float x,y,z;

  Vec3f normalized() const {
    float r = rsqrt(lengthSq());
    return Vec3f{x*r, y*r, z*r};
  }

  float lengthSq() const {
    return x*x + y*y + z*z;
  }
  
  float length() const {
    return sqrt(lengthSq());
  }

  friend Vec3f operator+(Vec3f lhs, const Vec3f &rhs) {
    return Vec3f{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
  }

  friend Vec3f operator-(Vec3f lhs, const Vec3f &rhs) {
    return Vec3f{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
  }
  
  friend Vec3f operator*(Vec3f lhs, float f) {
    return Vec3f{lhs.x * f, lhs.y * f, lhs.z * f};
  }
};

static float dot(Vec3f a, Vec3f b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__attribute__((unused))
static void println(Vec3f a) {
  printf("%f %f %f\n", a.x, a.y, a.z);
}

struct alignas(16) Sphere4 {
  float x[4];
  float y[4];
  float z[4];
  float rSq[4];
};

struct Light {
  Light(Vec3f p_, float intensity_) : p(p_), intensity(intensity_) {};
  Vec3f p;
  float intensity;
};

struct Scene {
  Sphere4 spheres;
  uint32_t numSpheres;
  std::vector<Light> lights;
};

struct Ray {
  Vec3f p, d;
};

struct Hit {
  Hit(Vec3f p_, float t_, Vec3f n_, uint32_t id_) : p(p_), t(t_), n(n_), id(id_) {}
  Vec3f p;
  float t;
  Vec3f n;
  uint32_t id;
};

static Ray generate_ray(float x, float y) {
  //   /|
  //  / |
  // .  |
  //  \ |
  //   \|

  Vec3f p{0.f, 0.f, 0.f};

  constexpr float fdist = 5.f;

  Vec3f d = Vec3f{x/fdist, y/fdist, 1.f}.normalized();
  return Ray{p,d};
}

static std::optional<Hit> hit(const Scene &scene, const Ray &ray) {
  const Sphere4 &spheres = scene.spheres;
  std::optional<Hit> hit = std::nullopt;
  for(uint32_t i = 0; i < scene.numSpheres; ++i) {
    Vec3f c{spheres.x[i], spheres.y[i], spheres.z[i]};
    float rSq = spheres.rSq[i];

    const Vec3f o = ray.p - c;

    /*
    d/dt (ox + t*dx) ^ 2 + (oy + t*dy) ^ 2 == 0
    d/dt ox^2 + 2ox*t*dx + t^2*dx^2 + ... == 0
    2 ox dx + 2 t dx^2 + 2 oy dy + 2 t dy^2 == 0
    ox dx + oy dy + t (dx^2 + dy^2) == 0
    t d.d == -o.d
    t == -o.d/d.d

    

    */

    const float tm = -dot(o, ray.d);
    const float dSq = (o + ray.d * tm).lengthSq(); 
    if(dSq <= rSq) {
      const float s = sqrt(rSq - dSq);
      float t;
      if(tm > s) {
        t = tm - s;
      } else {
        t = tm + s;
      }

      if(t > 0.f && (hit == std::nullopt || t < hit->t)) {
        Vec3f p = ray.p + ray.d * t;
        Vec3f n = (p - c).normalized();
        hit = std::make_optional<Hit>(p, t, n, i);
      }
    }
  }
  return hit;
}

static bool hit_any(const Scene &scene, const Ray &ray, float maxT)
{
  const Sphere4 &spheres = scene.spheres;
  for(uint32_t i = 0; i < scene.numSpheres; ++i) {
    Vec3f c{spheres.x[i], spheres.y[i], spheres.z[i]};
    float rSq = spheres.rSq[i];

    const Vec3f o = ray.p - c;
    const float tm = -dot(o, ray.d);
    const float dSq = (o + ray.d * tm).lengthSq(); 
    if(dSq <= rSq) {
      if(tm > 0.f && tm <= maxT) { // cheat ?
        return true;
      }
    }
  }
  return false;
}

static std::unique_ptr<Scene> make_scene() 
{

  auto scene = std::make_unique<Scene>();

  struct Sphere {
    Vec3f p;
    float r;
  };

  Sphere spheres[] = { { Vec3f{0.f, -1.f, 24.f}, 3.f },
                       { Vec3f{-1.f, 2.5f, 22.f}, 0.5f } };

  size_t i = 0;
  for(const Sphere &s : spheres) {
    scene->spheres.x[i] = s.p.x;
    scene->spheres.y[i] = s.p.y;
    scene->spheres.z[i] = s.p.z;
    scene->spheres.rSq[i] = sq(s.r);
    ++i;
  }
  scene->numSpheres = i;

  scene->lights.emplace_back(Vec3f{0.f, 10.f, 20.f}, 2.f);
  scene->lights.emplace_back(Vec3f{-10.f, 10.f, 20.f}, 1.f);
  return scene;
}

int main() {
  constexpr size_t width = 800;
  constexpr size_t height = 800;

  auto scene = make_scene();

  int *id = new int[width * height];
  float *depth = new float[width * height];

  for(size_t x = 0; x < width; x++) {
    for(size_t y = 0; y < height; y++) {
      Ray ray = generate_ray(2.f*(float)x/(width-1) - 1.f, -2.f*(float)y/(height-1) + 1.f);
      
      auto h = hit(*scene, ray);
      if(h) {
        float ld = 0.f;
        for(const auto &l : scene->lights) {
          const Vec3f lv = l.p - h->p;
          const Vec3f ldir = lv.normalized();
          const float ldist = lv.length();
          const float cosPhi = dot(h->n, ldir);
          if(cosPhi > 0.f && !hit_any(*scene, Ray{h->p + h->n * 0.01f, ldir}, ldist)) {
            ld += l.intensity * cosPhi / (ldist * ldist);
          }
        }
        id[y * width + x] = h->id + 10;
        depth[y * width + x] = ld;
      }
    }
  }
  writeImage(id, depth, width, height, "foo.ppm");
  delete id;
  delete depth;
}
