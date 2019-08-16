// A simple proof of concept to make sure the semantics of VNNI.
// The official doc is incorrect!

#include <x86intrin.h>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "util.h"

uint8_t a[64];
uint8_t b[64];
uint8_t c[64];
uint32_t d[8][16];

void kernel() {
  __m512i _a, _b, _c, _d0, _d1, _d2, _d3, _d4, _d5, _d6, _d7;
  _a = _mm512_load_si512(a);
  _b = _mm512_load_si512(b);
  _c = _mm512_load_si512(c);

  _d0 = _mm512_load_si512(d[0]);
  _d1 = _mm512_load_si512(d[1]);
  _d2 = _mm512_load_si512(d[2]);
  _d3 = _mm512_load_si512(d[3]);
  _d4 = _mm512_load_si512(d[4]);
  _d5 = _mm512_load_si512(d[5]);
  _d6 = _mm512_load_si512(d[6]);
  _d7 = _mm512_load_si512(d[7]);

  int64_t n = 5ll * 1024 * 1024 * 1024;
  for (int64_t i = 0; i < n; i += 8) {
    _d0 = _mm512_dpbusd_epi32(_d0, _a, _b);
    _d1 = _mm512_dpbusd_epi32(_d1, _a, _b);
    _d2 = _mm512_dpbusd_epi32(_d2, _a, _b);
    _d3 = _mm512_dpbusd_epi32(_d3, _a, _b);
    _d4 = _mm512_dpbusd_epi32(_d4, _a, _b);
    _d5 = _mm512_dpbusd_epi32(_d5, _a, _b);
    _d6 = _mm512_dpbusd_epi32(_d6, _a, _b);
    _d7 = _mm512_dpbusd_epi32(_d7, _a, _b);
  }

  _mm512_store_epi32(d[0], _d0);
  _mm512_store_epi32(d[1], _d1);
  _mm512_store_epi32(d[2], _d2);
  _mm512_store_epi32(d[3], _d3);
  _mm512_store_epi32(d[4], _d4);
  _mm512_store_epi32(d[5], _d5);
  _mm512_store_epi32(d[6], _d6);
  _mm512_store_epi32(d[7], _d7);
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
