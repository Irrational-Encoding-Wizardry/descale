#ifndef DESCALE_COMMON_H
#define DESCALE_COMMON_H


static inline int ceil_n(int x, int n)
{
    return (x + (n - 1)) & ~(n - 1);
}


static inline int floor_n(int x, int n)
{
    return x & ~(n - 1);
}


#endif  // DESCALE_COMMON_H
