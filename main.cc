#include <cassert>
#include <iostream>
#include <sys/time.h>
#include <x86intrin.h>
#include <cstdint>

#define ALIGN(x, y) (x + (y - x % y))

#define N 1
#define C 4
#define H_IN 127
#define W_IN 127

#define N_KER 32
#define H_KER 64
#define W_KER 64

#define H_OUT (H_IN - H_KER + 1)
#define W_OUT (W_IN - W_KER + 1)

uint8_t a[N][C][H_IN][ALIGN(W_IN, 64)] __attribute__((aligned(512)));
uint8_t b[N_KER][C][H_KER][ALIGN(W_KER, 64)] __attribute__((aligned(512)));
uint32_t c[N][N_KER][H_OUT][ALIGN(W_OUT, 16)] __attribute__((aligned(512)));
uint32_t d[N][N_KER][H_OUT][ALIGN(W_OUT, 16)] __attribute__((aligned(512)));
uint32_t e[N][N_KER][H_OUT][ALIGN(W_OUT, 16)] __attribute__((aligned(512)));

struct timeval tv0, tv1;

#define TV_TO_USEC(tv) (tv.tv_sec * 1000000 + tv.tv_usec)

void begin_roi() {
  gettimeofday(&tv0, nullptr);
}

void end_roi() {
  gettimeofday(&tv1, nullptr);
  std::cout << "Duration: " << TV_TO_USEC(tv1) - TV_TO_USEC(tv0) << std::endl;
}

void vanilla(uint8_t a[][C][H_IN][ALIGN(W_IN, 64)],
             uint8_t b[][C][H_KER][ALIGN(W_KER, 64)],
             uint32_t c[][N_KER][H_OUT][ALIGN(W_OUT, 16)]) {

  for (int batch = 0; batch < N; ++batch)
    for (int filter = 0; filter < N_KER; ++filter)
      for (int channel = 0; channel < C; ++channel)
        for (int i = 0; i < H_OUT; ++i)
          for (int j = 0; j < W_OUT; ++j) {
            uint32_t &sum = c[batch][filter][i][j];
            sum = 0;
            for (int x = 0; x < H_KER; ++x)
              for (int y = 0; y < W_KER; ++y) {
                sum += a[batch][channel][i + x][j + y] * b[filter][channel][x][y];
              }
          }
}

void tiled(uint8_t a[][C][H_IN][ALIGN(W_IN, 64)],
           uint8_t b[][C][H_KER][ALIGN(W_KER, 64)],
           uint32_t c[][N_KER][H_OUT][ALIGN(W_OUT, 16)]) {

#define check_addr(a, i, j, k, l)       \
    do {                                \
      void *addr = &a[i][j][k][l];      \
      uint64_t addr_ = (uint64_t) addr; \
      if (addr_ % 64 != 0) {            \
        std::cout << addr_ % 64         \
                  << std::endl;         \
        std::cout << addr << " "        \
                  << i << " "           \
                  << j << " " << k      \
                  << " " << l           \
                  << ":" << addr_       \
                  << std::endl;         \
        std::cout.flush();              \
        assert(false);                  \
      }                                 \
    } while (false) 


  for (int batch = 0; batch < N; ++batch)
    for (int filter = 0; filter < N_KER; ++filter)
      for (int channel = 0; channel < C; ++channel)
        for (int i = 0; i < H_OUT; ++i)
          for (int j = 0; j < W_OUT; ++j) {
            uint32_t &sum = c[batch][filter][i][j];
            sum = 0;
            for (int x = 0; x < H_KER; ++x)
              for (int y = 0; y < W_KER; ++y) {
                sum += a[batch][channel][i + x][j + y] * b[filter][channel][x][y];
              }
          }
}

int main() {

#define init(a, N, C, H, W)            \
  for (int i = 0; i < N; ++i)          \
    for (int j = 0; j < C; ++j)        \
      for (int k = 0; k < H; ++k)      \
        for (int l = 0; l < W; ++l)    \
          a[i][j][k][l] = rand() % 128;

  init(a, N, C, W_IN, H_IN);
  init(b, N_KER, C, W_KER, H_KER);

  vanilla(a, b, e);
  begin_roi();
  vanilla(a, b, c);
  end_roi();

  tiled(a, b, e);
  begin_roi();
  tiled(a, b, d);
  end_roi();

  for (int batch = 0; batch < N; ++batch) {
    for (int filter = 0; filter < N_KER; ++filter) {
      for (int x = 0; x < H_OUT; ++x) {
        for (int y = 0; y < W_OUT; ++y) {
          if (c[batch][filter][x][y] != d[batch][filter][x][y]) {
            std::cout << batch << ", " << filter << ", " << x << ", " << y << ": "
                      << c[batch][filter][x][y] << ", " << d[batch][filter][x][y]
                      << std::endl;
            assert(false);
          }
        }
      }
    }
  }

  return 0;
}

//_mm512_broadcastd_epi32
