// A simple proof of concept to make sure the semantics of VNNI.
// The official doc is incorrect!

#include <x86intrin.h>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sys/time.h>

uint8_t a[64];
uint8_t b[64];
uint8_t c[64];
uint32_t d[8][16];

struct timeval tv0, tv1;

void begin_roi() {
  gettimeofday(&tv0, nullptr);
}

#define TV_TO_SEC(tv) (tv.tv_sec * 1000000 + tv.tv_usec)

void end_roi() {
  gettimeofday(&tv1, nullptr);
  std::cout << TV_TO_SEC(tv1) - TV_TO_SEC(tv0) << std::endl;
}

void kernel() {
  __m512i _a, _b, _c, _d;
  _a = _mm512_load_si512(a);
  _b = _mm512_load_si512(b);
  _c = _mm512_load_si512(c);
  for (int i = 0; i < 1024 * 1024 * 1024; ++i) {
    _d = _mm512_load_si512(d[i % 8]);
    _d = _mm512_dpbusd_epi32(_d, _a, _b);
    _mm512_store_epi32(d[i % 8], _d);
  }
}

int main() {

  for (int i = 0; i < 64; ++i) {
    a[i] = i + 1;
    b[i] = 64 - i;
  }

  kernel();

  begin_roi();
  kernel();
  end_roi();

  for (int i = 0; i < 16; ++i) {
    std::cout << d[i] << std::endl;
  }

  return 0;

}
