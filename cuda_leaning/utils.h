#include <time.h>
#include <sys/time.h>

double get_now() {
    timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec * 1000 + (double)time.tv_usec * 0.001;
}