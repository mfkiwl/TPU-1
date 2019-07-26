#ifndef GPUVECTORKERNELS_H
#define GPUVECTORKERNELS_H

__global__ void _InitI(int *res,int val,int columns);
__global__ void _TruncI(int *res,int *a,int start,int columns);

#endif
