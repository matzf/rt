#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <xmmintrin.h>
#include <immintrin.h>
#include <optional>
#include <algorithm>
#include <memory>
#include <vector>

static constexpr float ray_mint = 1e-3f;

static void writeImage(int *idImage, float *depthImage, int width, int height, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror(filename);
        exit(1);
    }

    float max_depth = *std::max_element(depthImage, depthImage + width * height);
    max_depth = std::max(max_depth, 0.03f);
    printf("%f\n", max_depth);

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

  Vec3f operator-() const {
    return Vec3f{-x, -y, -z};
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

struct alignas(32) Sphere8 {
  float x[8];
  float y[8];
  float z[8];
  float rSq[8];
};

struct Light {
  Light(Vec3f p_, float intensity_) : p(p_), intensity(intensity_) {};
  Vec3f p;
  float intensity;
};

struct Scene {
  Sphere8 spheres;
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

  Vec3f p{x, y, 0.f};

  constexpr float fdist = 3.f;

  Vec3f d = Vec3f{x/fdist, y/fdist, 1.f}.normalized();
  return Ray{p,d};
}

struct Vec3f8 {
  __m256 x, y, z;

  static Vec3f8 splat(Vec3f v) {
    return Vec3f8{
      _mm256_broadcast_ss(&v.x),
      _mm256_broadcast_ss(&v.y),
      _mm256_broadcast_ss(&v.z)
    };
  }
  static Vec3f8 load(const float *x, const float *y, const float *z) {
    return Vec3f8{
      _mm256_load_ps(x),
      _mm256_load_ps(y),
      _mm256_load_ps(z)
    };
  }

  static __m256 dot(const Vec3f8 &a, const Vec3f8 &b) {
    return _mm256_add_ps(_mm256_mul_ps(a.x, b.x),
                         _mm256_add_ps(_mm256_mul_ps(a.y, b.y),
                                       _mm256_mul_ps(a.z, b.z)));

  }

  static Vec3f8 sub(const Vec3f8 &a, const Vec3f8 &b) {
    return Vec3f8{
      _mm256_sub_ps(a.x, b.x),
      _mm256_sub_ps(a.y, b.y),
      _mm256_sub_ps(a.z, b.z)
    };
  }
};


// Compute horizontal minimum. Does not consider NaNs (dont know what happens).
static void min_horiz(__m256 a, int *min_idx, float *min_val) {
  __m128 a_lo = _mm256_castps256_ps128(a);
  __m128 a_hi = _mm256_extractf128_ps(a, 1);
  __m128 b = _mm_min_ps(a_lo, a_hi); // 4

  __m128 b_lo = b;
  __m128 b_hi = _mm_permute_ps(b, 2 | 3<<2); // rest 2 entries ignored
  __m128 c = _mm_min_ps(b_lo, b_hi); // 2

  __m128 c_lo = c;
  __m128 c_hi = _mm_permute_ps(c, 1); // rest 3 entries ignored
  __m128 d = _mm_min_ps(c_lo, c_hi); // 1

  *min_val = _mm_cvtss_f32(d);

  // splat min accross.
  __m256 vmin = _mm256_permute_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(d), d, 1), 0);
  __m256 vcmp = _mm256_cmp_ps(a, vmin, _CMP_EQ_UQ);
  uint32_t mask = _mm256_movemask_ps(vcmp);
  *min_idx = __builtin_ctz(mask);
}

static std::optional<Hit> hit_spheres_f(const Sphere8 &spheres, size_t n, const Ray &ray)
{
  const __m256 inf = _mm256_set1_ps(INFINITY);
  const __m256 negHalf = _mm256_set1_ps(-0.5f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 ray_mint_ = _mm256_set1_ps(ray_mint);

  __m256 A = _mm256_set1_ps(dot(ray.d, ray.d));
  Vec3f8 c = Vec3f8::load(spheres.x, spheres.y, spheres.z);
  __m256 rSq = _mm256_load_ps(spheres.rSq);
  Vec3f8 p = Vec3f8::splat(ray.p);
  Vec3f8 d = Vec3f8::splat(ray.d);

  Vec3f8 o = Vec3f8::sub(p, c);
  __m256 B = _mm256_mul_ps(_mm256_set1_ps(2.f), Vec3f8::dot(o, d));
  __m256 C = _mm256_sub_ps(Vec3f8::dot(o, o), rSq);

  __m256 AC4 = _mm256_mul_ps(_mm256_set1_ps(4.f),
                             _mm256_mul_ps(A, C));
  __m256 discrim = _mm256_sub_ps(_mm256_mul_ps(B, B), AC4);

  __m256 discrim_nonneg = _mm256_cmp_ps(discrim, zero, _CMP_GE_OQ);
  if(_mm256_movemask_ps(discrim_nonneg) == 0) {
    return std::nullopt;
  }

  __m256 rootDiscrim = _mm256_sqrt_ps(discrim);
  __m256 qPos = _mm256_mul_ps(negHalf, _mm256_sub_ps(B, rootDiscrim));
  __m256 qNeg = _mm256_mul_ps(negHalf, _mm256_add_ps(B, rootDiscrim));
  __m256 q = _mm256_blendv_ps(qPos, qNeg, _mm256_cmp_ps(B, zero, _CMP_LT_OQ));
  __m256 t0 = _mm256_div_ps(q, A);
  __m256 t1 = _mm256_div_ps(C, q);

  // Set to INF if below mint and also turn NaNs to INF.
  __m256 t0_or_inf = _mm256_blendv_ps(inf, t0, _mm256_cmp_ps(t0, ray_mint_, _CMP_GT_OQ));
  __m256 t1_or_inf = _mm256_blendv_ps(inf, t1, _mm256_cmp_ps(t1, ray_mint_, _CMP_GT_OQ));
  __m256 t = _mm256_min_ps(t0_or_inf, t1_or_inf);

  float tmin;
  int id;
  min_horiz(t, &id, &tmin);

  if(tmin < INFINITY) {
    Vec3f p = ray.p + ray.d * tmin;
    Vec3f n = (p - Vec3f{spheres.x[id], spheres.y[id], spheres.z[id]}).normalized();
    return std::make_optional<Hit>(p, tmin, n, id);
  } else {
    return std::nullopt;
  }
}

static bool quadratic(float A, float B, float C, float *t0, float *t1)
{
  float discrim = B * B - 4.f * A * C;
  if (discrim < 0.f) return false;
  float rootDiscrim = sqrt(discrim);
  float q;
  if (B < 0.f) q = -.5f * (B - rootDiscrim);
  else         q = -.5f * (B + rootDiscrim);
  *t0 = q / A;
  *t1 = C / q;
  if(*t0 > *t1) std::swap(*t0, *t1);
  return true;
}

static std::optional<Hit> hit_spheres(const Sphere8 &spheres, size_t n, const Ray &ray)
{
  std::optional<Hit> hit = std::nullopt;

  const Vec3f d = ray.d;
  const float A = dot(d, d);

  for(uint32_t i = 0; i < n; ++i) {
    Vec3f c{spheres.x[i], spheres.y[i], spheres.z[i]};
    float rSq = spheres.rSq[i];

    const Vec3f o = ray.p - c;
    const float B = 2.f * dot(o, d);
    const float C = dot(o, o) - rSq;
    float t0, t1;
    if(quadratic(A, B, C, &t0, &t1)) {
      if(t1 < ray_mint)
        continue;
      float t;
      if(t0 > ray_mint) t = t0;
      else              t = t1;

      if(!hit || t < hit->t) {
        Vec3f p = ray.p + d * t;
        Vec3f n = (p - c).normalized();
        hit = std::make_optional<Hit>(p, t, n, i);
      }
    }
  }
  return hit;
}

static std::optional<Hit> hit(const Scene &scene, const Ray &ray) {
  return hit_spheres(scene.spheres, scene.numSpheres, ray);
}

static bool hit_any(const Scene &scene, const Ray &ray, float maxt)
{
  auto h = hit(scene, ray);
  if(h && h->t < maxt) {
    return true;
  }
  return false;

  const Sphere8 &spheres = scene.spheres;
  for(uint32_t i = 0; i < scene.numSpheres; ++i) {
    Vec3f c{spheres.x[i], spheres.y[i], spheres.z[i]};
    float rSq = spheres.rSq[i];

    const Vec3f o = ray.p - c;
    const float tm = -dot(o, ray.d);
    const float dSq = (o + ray.d * tm).lengthSq();
    if(dSq <= rSq) {
      if(tm > ray_mint && tm <= maxt) { // cheat ?
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

  /*
  Sphere spheres[] = { { Vec3f{0.f, 3.f, 25.f}, 3.f },
                       { Vec3f{0.f, -2.5f, 25.f}, 2.5f },
                       { Vec3f{0.f, 1.5f, 21.8f}, 0.15f },
                       { Vec3f{1.f, 0.5f, 21.4f}, 0.1f },
                       { Vec3f{-1.f, -0.5f, 20.f}, 0.3f },
                       { Vec3f{-1.5f, 0.f, 21.f}, 0.6f },
                       { Vec3f{5.f, 0.5f, 24.f}, 1.5f },
                       { Vec3f{.5f, -4.f, 15.f}, 0.8f },
                     };
  */
  Sphere spheres[] = {
                       { Vec3f{0.f, 3.f, 25.f}, 3.f },
                       { Vec3f{0.f, -2.5f, 25.f}, 2.5f },
                       { Vec3f{0.f, 1.5f, 21.8f}, 0.15f },
                       { Vec3f{1.f, 0.5f, 21.4f}, 0.1f },
                       { Vec3f{-1.f, -0.5f, 20.f}, 0.3f },
                       { Vec3f{-1.5f, 0.f, 21.f}, 0.6f },
                       { Vec3f{5.f, 0.5f, 24.f}, 1.5f },
                       { Vec3f{.5f, -4.f, 15.f}, 0.8f },
                     };

  size_t i = 0;
  for(const Sphere &s : spheres) {
    scene->spheres.x[i] = s.p.x;
    scene->spheres.y[i] = s.p.y;
    scene->spheres.z[i] = s.p.z;
    scene->spheres.rSq[i] = sq(s.r);
    ++i;
  }
  scene->numSpheres = i;

  scene->lights.emplace_back(Vec3f{0.f, 10.f, 20.f}, 10.f);
  scene->lights.emplace_back(Vec3f{-10.f, 1.f, 20.f}, 10.f);
  return scene;
}

static float frand() {
  return drand48();
}

static float direct(const Scene &scene, Vec3f p, Vec3f n) {
  float l = 0.f;
  for(const auto &light : scene.lights) {
    const Vec3f lv = light.p - p;
    const Vec3f ldir = lv.normalized();
    const float ldist = lv.length();
    const float cosPhi = dot(n, ldir);
    if(cosPhi > 0.f && !hit_any(scene, Ray{p, ldir}, ldist)) {
      l += light.intensity * cosPhi / (ldist * ldist);
    }
  }
  return l;
}

static Vec3f sample_hemisphere(Vec3f n) {
  // sample hemisphere
  float a = 2.f*frand()-1.f;
  float b = 2.f*frand()-1.f;
  float xSq = abs(a);
  float x = copysignf(sqrt(xSq), a);
  float ySq = abs(b) * (1.f - xSq);
  float y = copysign(sqrt(ySq), b);
  float z = sqrt(1 - xSq - ySq);
  Vec3f r{x,y,z};
  if(dot(n, r) >= 0.f) {
    return r;
  } else {
    return -r;
  }
}

static float indirect(const Scene &scene, Vec3f p, Vec3f n) {
  constexpr size_t numSamples = 4;
  constexpr float albedo = 0.75f;
  float l = 0.f;
  for(size_t i = 0; i < numSamples; ++i) {
    Vec3f dir = sample_hemisphere(n);
    auto h = hit(scene, Ray{p, dir});
    if(h) {
      l += albedo * dot(dir, n) * direct(scene, h->p + h->n, h->n); // numBounces: + indirect(scene, h->p, h-n);
    }
  }
  l /= (float)numSamples;
  return l;
}

int main() {
  constexpr size_t width = 5000;
  constexpr size_t height = 5000;

  auto scene = make_scene();

  int *id = new int[width * height];
  float *depth = new float[width * height];

  for(size_t x = 0; x < width; x++) {
    for(size_t y = 0; y < height; y++) {
      Ray ray = generate_ray(2.f*(float)x/(width-1) - 1.f, -2.f*(float)y/(height-1) + 1.f);

      auto h = hit(*scene, ray);
      if(h) {
        float l = direct(*scene, h->p, h->n) + indirect(*scene, h->p, h->n);
        //float l = indirect(*scene, h->p, h->n);
        id[y * width + x] = h->id + 10;
        depth[y * width + x] = l;
      }
    }
  }
  writeImage(id, depth, width, height, "foo.ppm");
  delete id;
  delete depth;
}
