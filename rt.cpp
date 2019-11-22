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

static void writeImage(int *idImage, float *image, int width, int height, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror(filename);
        exit(1);
    }

    float max_val = *std::max_element(image, image + width * height);
    max_val = std::max(max_val, 0.03f);
    printf("%f\n", max_val);

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
            float d = image[y * width + x] / max_val;
            /*
            r *= d;
            g *= d;
            b *= d;
            */
            r = g = b = 255 * d;
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

namespace {
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




static __m256 copysign_ps(__m256 mag, __m256 sig) {
  auto const sig_mask = _mm256_set1_ps(-0.f);
  return _mm256_or_ps(_mm256_and_ps(sig_mask, sig), _mm256_andnot_ps(sig_mask, mag)); // (mask & sig) | (~mask & mag)
}

static __m256 abs_ps(__m256 x) {
  auto const sig_mask = _mm256_set1_ps(-0.f);
  return _mm256_andnot_ps(sig_mask, x);
}

static __m256 sigbit_ps(__m256 x) {
  auto const sig_mask = _mm256_set1_ps(-0.f);
  return _mm256_and_ps(sig_mask, x);
}

static __m256 sq_ps(__m256 x) {
  return _mm256_mul_ps(x, x);
}

static float hsum_ps(__m256 a) {
  __m128 a_lo = _mm256_castps256_ps128(a);
  __m128 a_hi = _mm256_extractf128_ps(a, 1);
  __m128 b = _mm_add_ps(a_lo, a_hi); // 4

  __m128 b_lo = b;
  __m128 b_hi = _mm_permute_ps(b, 2 | 3<<2); // rest 2 entries ignored
  __m128 c = _mm_add_ps(b_lo, b_hi); // 2

  __m128 c_lo = c;
  __m128 c_hi = _mm_permute_ps(c, 1); // rest 3 entries ignored
  __m128 d = _mm_add_ps(c_lo, c_hi); // 1

  return _mm_cvtss_f32(d);
}

__attribute__((unused))
static void println_ps(__m256 x) {
  for(size_t i = 0; i < 7; ++i) {
    printf("%f ", x[i]);
  }
  printf("%f\n", x[7]);
}


struct Vec3f8 {
  __m256 x, y, z;

  static Vec3f8 splat(Vec3f v) {
    return Vec3f8{
      _mm256_set1_ps(v.x),
      _mm256_set1_ps(v.y),
      _mm256_set1_ps(v.z)
    };
  }

  static Vec3f8 load(const float *x, const float *y, const float *z) {
    return Vec3f8{
      _mm256_load_ps(x),
      _mm256_load_ps(y),
      _mm256_load_ps(z)
    };
  }

  static Vec3f at(const Vec3f8 &a, size_t i) {
    return Vec3f {
      a.x[i],
      a.y[i],
      a.z[i],
    };
  }

  static Vec3f8 add(const Vec3f8 &a, const Vec3f8 &b) {
    return Vec3f8{
      _mm256_add_ps(a.x, b.x),
      _mm256_add_ps(a.y, b.y),
      _mm256_add_ps(a.z, b.z)
    };
  }

  static Vec3f8 sub(const Vec3f8 &a, const Vec3f8 &b) {
    return Vec3f8{
      _mm256_sub_ps(a.x, b.x),
      _mm256_sub_ps(a.y, b.y),
      _mm256_sub_ps(a.z, b.z)
    };
  }

  static __m256 dot(const Vec3f8 &a, const Vec3f8 &b) {
    return _mm256_add_ps(_mm256_mul_ps(a.x, b.x),
                         _mm256_add_ps(_mm256_mul_ps(a.y, b.y),
                                       _mm256_mul_ps(a.z, b.z)));

  }

  static Vec3f8 scale(const Vec3f8 &a, const __m256 &s) {
    return Vec3f8{
      _mm256_mul_ps(a.x, s),
      _mm256_mul_ps(a.y, s),
      _mm256_mul_ps(a.z, s)
    };
  }

  static __m256 rlength(const Vec3f8 &a) {
    return _mm256_rsqrt_ps(dot(a, a));
  }

  static Vec3f8 normalized(const Vec3f8 &a) {
    return scale(a, rlength(a));
  }

  static void println(Vec3f8 &a) {
    for(size_t i = 0; i < 7; ++i) {
      printf("[%f %f %f] ", a.x[i], a.y[i], a.z[i]);
    }
    printf("[%f %f %f]\n", a.x[7], a.y[7], a.z[7]);
  }
};

struct Ray {
  Vec3f p, d;
};

struct Ray8 {
  Vec3f8 p, d;
};

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

template<typename T>
struct vec : public std::vector<T> {
  size_t num; // num _valid_ entries in this vector
};

struct Scene {
  vec<Sphere8> spheres;
  std::vector<Light> lights;
};


struct Hit {
  Hit(Vec3f p_, float t_, Vec3f n_, uint32_t id_) : p(p_), t(t_), n(n_), id(id_) {}
  Vec3f p;
  float t;
  Vec3f n;
  uint32_t id;
};
} // namespace

// Compute horizontal minimum. Does not consider NaNs (dont know what happens).
static void argmin_horiz(__m256 a, int *min_idx, float *min_val) {
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

// Solve quadratic equation, and return smaller solution >= ray_mint
static __m256 quadratic_ps(__m256 A, __m256 B, __m256 C) {
  const __m256 inf = _mm256_set1_ps(INFINITY);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 negHalf = _mm256_set1_ps(-0.5f);
  const __m256 ray_mint_ = _mm256_set1_ps(ray_mint);

  __m256 AC4 = _mm256_mul_ps(_mm256_set1_ps(4.f),
                             _mm256_mul_ps(A, C));
  __m256 discrim = _mm256_sub_ps(_mm256_mul_ps(B, B), AC4);

  __m256 discrim_nonneg = _mm256_cmp_ps(discrim, zero, _CMP_GE_OQ);
  if(_mm256_movemask_ps(discrim_nonneg) == 0) {
    return inf;
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
  return _mm256_min_ps(t0_or_inf, t1_or_inf);
}

static __m256 hit_sphere_ray8(const Vec3f8 &c, const __m256 rSq, const Ray8 &rays)
{
  __m256 A = Vec3f8::dot(rays.d, rays.d);
  Vec3f8 o = Vec3f8::sub(rays.p, c);
  __m256 B = _mm256_mul_ps(_mm256_set1_ps(2.f), Vec3f8::dot(o, rays.d));
  __m256 C = _mm256_sub_ps(Vec3f8::dot(o, o), rSq);

  return quadratic_ps(A, B, C);
}


struct Hit8 {
  Vec3f8 p;
  Vec3f8 n;
  __m256 t;
};

static Hit8 hit_ray8(const Scene &scene, const Ray8 &rays)
{
  __m256 tmin = _mm256_set1_ps(INFINITY);
  Vec3f8 cmin;
  for(const Sphere8 &spheres : scene.spheres) {
    for(size_t i = 0; i < 8; ++i) {
      Vec3f8 c = Vec3f8::splat(Vec3f{spheres.x[i], spheres.y[i], spheres.z[i]});
      __m256 rSq = _mm256_set1_ps(spheres.rSq[i]);
      __m256 t = hit_sphere_ray8(c, rSq, rays);
      __m256 m = _mm256_cmp_ps(t, tmin, _CMP_LT_OQ);
      if(_mm256_movemask_ps(m) == 0) {
        continue;
      }
      tmin = _mm256_blendv_ps(tmin, t, m);
      cmin.x = _mm256_blendv_ps(cmin.x, c.x, m);
      cmin.y = _mm256_blendv_ps(cmin.y, c.y, m);
      cmin.z = _mm256_blendv_ps(cmin.z, c.z, m);
    }
  }

  Vec3f8 p = Vec3f8::add(rays.p, Vec3f8::scale(rays.d, tmin));
  Vec3f8 n = Vec3f8::normalized(Vec3f8::sub(p, cmin));
  return Hit8{p, n, tmin};
}


static std::optional<Hit> hit_spheres_f(const Sphere8 &spheres, size_t n, const Ray &ray)
{
  (void)n;

  __m256 A = _mm256_set1_ps(dot(ray.d, ray.d));
  Vec3f8 c = Vec3f8::load(spheres.x, spheres.y, spheres.z);
  __m256 rSq = _mm256_load_ps(spheres.rSq);
  Vec3f8 p = Vec3f8::splat(ray.p);
  Vec3f8 d = Vec3f8::splat(ray.d);

  Vec3f8 o = Vec3f8::sub(p, c);
  __m256 B = _mm256_mul_ps(_mm256_set1_ps(2.f), Vec3f8::dot(o, d));
  __m256 C = _mm256_sub_ps(Vec3f8::dot(o, o), rSq);

  __m256 t = quadratic_ps(A, B, C);

  float tmin;
  int id;
  argmin_horiz(t, &id, &tmin);

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
  std::optional<Hit> closest_hit;
  for(size_t i = 0, n = scene.spheres.size(); i < n; ++i) {
    size_t num8 = (i < n-1) ? 8 : scene.spheres.num % 8;
    std::optional h = hit_spheres_f(scene.spheres[i], num8, ray);
    if(!closest_hit || h->t < closest_hit->t) {
      closest_hit = h;
    }
  }
  return closest_hit;
}

static bool hit_any(const Scene &scene, const Ray &ray, float maxt)
{
  auto h = hit(scene, ray);
  if(h && h->t < maxt) {
    return true;
  }
  return false;

  /*
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
  */
}

static std::unique_ptr<Scene> make_scene()
{

  auto scene = std::make_unique<Scene>();

  struct Sphere {
    Vec3f p;
    float r;
  };

  std::vector<Sphere> spheres = {
    { Vec3f{0.f, 3.f, 25.f}, 3.f },
    { Vec3f{0.f, -2.5f, 25.f}, 2.5f },
    { Vec3f{0.f, 1.5f, 21.8f}, 0.15f },
    { Vec3f{1.f, 0.5f, 21.4f}, 0.1f },
    { Vec3f{-1.f, -0.5f, 20.f}, 0.3f },
    { Vec3f{-1.5f, 0.f, 21.f}, 0.6f },
    { Vec3f{0.f, 0.f, 150.f}, 100.f },
    { Vec3f{0.f, 115.f, 0.f}, 100.f },
    { Vec3f{-5.f, 1.f, 18.5f}, 2.f },
    { Vec3f{-4.f, 3.f, 23.f}, 1.5f },
  };


  const Vec3f gridO{-7.5f, -5.f, 1.f};
  const size_t gridN = 10;
  const float gridRadius = 0.1f;
  const Vec3f gridRig = Vec3f{1.0f, 0.0f, 0.0f};
  const Vec3f gridUp  = Vec3f{0.5f, 1.0f, 0.f};
  const Vec3f gridFwd = Vec3f{0.0f, 0.0f, 3.f};
  for(size_t x = 0; x < gridN; ++x) {
    for(size_t y = 0; y < gridN; ++y) {
      for(size_t z = 0; z < gridN+1; ++z) {
        Vec3f pos = gridO
                    + gridRig * x
                    + gridUp * y
                    + gridFwd * z;
        spheres.push_back(Sphere{pos, gridRadius});
      }
    }
  }

  printf("%zu\n", spheres.size());

  // XXX This is ugly as fuck
  auto it = std::begin(spheres);
  size_t numSpheres = std::distance(it, std::end(spheres));
  scene->spheres.resize((numSpheres + 8 - 1) / 8);
  for(Sphere8 &s8 : scene->spheres) {
    for(size_t i = 0; i < 8 && it != std::end(spheres); ++i) {
      const Sphere& s = *it++;
      s8.x[i] = s.p.x;
      s8.y[i] = s.p.y;
      s8.z[i] = s.p.z;
      s8.rSq[i] = sq(s.r);
    }
  }
  if(numSpheres % 8 != 0) {
    for(size_t i = numSpheres % 8; i < 8; ++i) { // fill remaining entries
      scene->spheres.back().x[i] = INFINITY;
      scene->spheres.back().y[i] = INFINITY;
      scene->spheres.back().z[i] = INFINITY;
      scene->spheres.back().rSq[i] = 0.f;
    }
  }
  scene->spheres.num = numSpheres;

  scene->lights.emplace_back(Vec3f{0.f, 10.f, 20.f}, 10.f);
  scene->lights.emplace_back(Vec3f{-10.f, 10.f, 20.f}, 10.f);
  return scene;
}

static __m256 direct8(const Scene &scene, Vec3f8 p, Vec3f8 n) {
  __m256 le = _mm256_setzero_ps();
  for(const auto &light : scene.lights) {
    Vec3f8 lp = Vec3f8::splat(light.p);
    const Vec3f8 lv = Vec3f8::sub(lp, p);
    const __m256 ldistSq = Vec3f8::dot(lv, lv);
    const Vec3f8 ldir = Vec3f8::scale(lv, _mm256_rsqrt_ps(ldistSq));
    const __m256 cosPhi = Vec3f8::dot(n, ldir);

    Hit8 hits = hit_ray8(scene, Ray8{p, ldir}); // XXX skip comp for cosPhi <= 0

    __m256 lel = _mm256_div_ps(cosPhi, ldistSq);
    lel = _mm256_and_ps(lel, _mm256_cmp_ps(cosPhi, _mm256_setzero_ps(), _CMP_GT_OQ));
    lel = _mm256_and_ps(lel, _mm256_cmp_ps(sq_ps(hits.t), ldistSq, _CMP_GT_OQ));
    le = _mm256_fmadd_ps(lel, _mm256_set1_ps(light.intensity), le);
  }
  return le;
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

static Ray generate_ray(float x, float y) {
  constexpr float fdist = 3.f;
  Vec3f p{x, y, 0.f};
  Vec3f d = Vec3f{x/fdist, y/fdist, 1.f}.normalized();
  return Ray{p,d};
}

static float frand() {
  return drand48();
}

__attribute__((unused))
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

struct xorshift32_state8 {
  __m256i a;
};

static xorshift32_state8 seed_xorshift328(uint32_t s) {
  // Avoid 0.
  if (s + 7 < s) {
    s += 8;
  }
  return xorshift32_state8{
    _mm256_set_epi32(s, s+1, s+2, s+3, s+4, s+5, s+6, s+7)
  };
}

static __m256i xorshift328(xorshift32_state8 *state) {
  __m256i x = state->a;
  //x ^= x << 13;
  //x ^= x >> 17;
  //x ^= x << 5;
  x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
  x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
  x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
  state->a = x;
  return x;
}

__attribute__((unused))
static __m256 frand8(xorshift32_state8 *rng_state) {
  __m256i r = xorshift328(rng_state);
  // shift right 9, leaves 23 random bits for mantissa
  __m256i m = _mm256_srli_epi32(r, 9);

  __m256 one = _mm256_set1_ps(1.f);
  // bitwise or with 1.f sets the exponent bits so that the value of the float is in [1.f, 2.f)
  __m256 r1 = _mm256_or_ps(_mm256_castsi256_ps(m), one);
  // subtracting one gives a range [0.f, 1.f).
  return _mm256_sub_ps(r1, one);
}

static __m256 frand_posneg8(xorshift32_state8 *rng_state) {
  __m256i r = xorshift328(rng_state);
  // shift left 31 to get random sign bit
  __m256i s = _mm256_slli_epi32(r, 31);
  // shift right 9, leaves 23 random bits for mantissa
  __m256i m = _mm256_srli_epi32(r, 9);

  __m256 one = _mm256_set1_ps(1.f);
  // bitwise or of sign bit with 1.f, gives +/-1.f
  __m256 one_s = _mm256_or_ps(one, _mm256_castsi256_ps(s));
  // bitwise or with +/-1.f sets the sign and exponent bits so that the value of the float is in +/-[1.f, 2.f).
  __m256 r1 = _mm256_or_ps(_mm256_castsi256_ps(m), one_s);
  // for positive r1: subtract 1.f
  // for negative r1: subtract -1.f
  // gives a range (-1.f, 1.f).
  return _mm256_sub_ps(r1, one_s);
}


static xorshift32_state8 rng_state = seed_xorshift328(1);


static Vec3f8 sample_hemisphere8(Vec3f8 n) {
  __m256 one = _mm256_set1_ps(1.f);
  __m256 a = frand_posneg8(&rng_state);
  __m256 b = frand_posneg8(&rng_state);
  __m256 xSq = abs_ps(a);
  __m256 x = copysign_ps(_mm256_sqrt_ps(xSq), a);
  __m256 ySq = _mm256_mul_ps(abs_ps(b), _mm256_sub_ps(one, xSq));
  __m256 y = copysign_ps(_mm256_sqrt_ps(ySq), b);
  __m256 z = _mm256_sqrt_ps(_mm256_sub_ps(_mm256_sub_ps(one, xSq), ySq));
  Vec3f8 r{x,y,z};
  __m256 d = Vec3f8::dot(n, r);
  __m256 dsigbit = sigbit_ps(d);
  // flip sign of all coordinates if d is negative
  r.x = _mm256_xor_ps(r.x, dsigbit);
  r.y = _mm256_xor_ps(r.y, dsigbit);
  r.z = _mm256_xor_ps(r.z, dsigbit);
  return r;
}

static float indirect(const Scene &scene, Vec3f p, Vec3f n) {
  constexpr size_t numSampleRounds = 1;
  constexpr size_t numSamples = numSampleRounds * 8;
  constexpr float albedo = 0.75f;

#if 0
  const __m256 albedo8 = _mm256_set1_ps(albedo);
  const Vec3f8 p8 = Vec3f8::splat(p);
  const Vec3f8 n8 = Vec3f8::splat(n);
  __m256 l = _mm256_setzero_ps();

  for(size_t k = 0; k < numSampleRounds; ++k)
  {
    Vec3f8 dir8 = sample_hemisphere8(n8);
    Hit8 hits = hit_ray8(scene, Ray8{p8, dir8});

    __m256 ishit = _mm256_cmp_ps(hits.t, _mm256_set1_ps(INFINITY), _CMP_LT_OQ);
    if(_mm256_movemask_ps(ishit) == 0) {
      continue;
    }
    // XXX: coalesce only hits???
    __m256 ldirect = direct8(scene, hits.p, hits.n);

    __m256 w = _mm256_mul_ps(Vec3f8::dot(dir8, n8), albedo8);
    w = _mm256_and_ps(w, ishit);
    l = _mm256_fmadd_ps(w, ldirect, l);
  }
  return hsum_ps(l) / numSamples;
#else
  float l = 0.f;
  const Vec3f8 p8 = Vec3f8::splat(p);
  for(size_t k = 0; k < numSampleRounds; ++k) {
#if 0
    Vec3f8 dir8 = sample_hemisphere8(Vec3f8::splat(n));
    Hit8 hits = hit_ray8(scene, Ray8{p8, dir8});
    for(size_t i = 0; i < 8; ++i) {
      Vec3f dir = Vec3f8::at(dir8, i);
      auto h = hit(scene, Ray{p, dir});
      if(h) {
        l += albedo * dot(dir, n) * direct(scene, h->p, h->n); // numBounces: + indirect(scene, h->p, h-n);
      }
    }
#else
    for(size_t i = 0; i < 8; ++i) {
      Vec3f dir = sample_hemisphere(n);
      auto h = hit(scene, Ray{p, dir});
      if(h) {
        l += albedo * dot(dir, n) * direct(scene, h->p, h->n); // numBounces: + indirect(scene, h->p, h-n);
      }
    }
#endif
  }
  l /= (float)numSamples;
  return l;
#endif
}

int main() {
  constexpr size_t width = 1000;
  constexpr size_t height = 1000;

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
