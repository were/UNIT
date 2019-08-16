#include <sys/time.h>
#include <iostream>

struct timeval tv0, tv1;

void begin_roi() {
  gettimeofday(&tv0, nullptr);
}

#define TV_TO_SEC(tv) (tv.tv_sec * 1000000 + tv.tv_usec)

void end_roi() {
  gettimeofday(&tv1, nullptr);
  std::cout << TV_TO_SEC(tv1) - TV_TO_SEC(tv0) << std::endl;
}
