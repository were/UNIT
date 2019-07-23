#include <x86intrin.h>
#include <cassert>
#include <cstdint>
#include <iostream>

uint8_t a[64];
uint8_t b[64];
uint8_t c[64];
uint32_t d[16];

int main() {
  __m512i _a, _b, _c, _d;

  for (int i = 0; i < 64; ++i) {
    a[i] = i + 1;
    b[i] = 64 - i;
  }

  _a = _mm512_load_si512(a);
  _b = _mm512_load_si512(b);
  _c = _mm512_load_si512(c);
  _d = _mm512_dpbusd_epi32(_c, _a, _b);
  _mm512_store_epi32(d, _d);

  for (int i = 0; i < 16; ++i)
    std::cout << d[i] << " ";
  std::cout << std::endl;

  for (int i = 0; i < 16; ++i) {
    uint32_t sum = 0;
    for (int j = 0; j < 4; ++j)
      sum += (uint16_t) a[i * 4 + j] * (uint16_t) b[i * 4 + j];
    std::cout << sum << " ";
    assert(sum == d[i]);
  }
  std::cout << std::endl;
}
