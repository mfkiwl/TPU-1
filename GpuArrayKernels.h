#ifndef GPUARRAYKERNELS_H
#define GPUARRAYKERNELS_H

__global__ void _InitI(int *res,int val,int rows,int columns);
__global__ void _AddI(int *res,int *arr,int arrRows,int arrColumns,int rows,int columns);
__global__ void _LAddI(int *res,int *arr,int lamda,int arrRows,int arrColumns,int rows,int columns);
__global__ void _MulI(int *res,int *a,int *b,int cDim,int rows,int columns);

__global__ void _InitD(double *res,double val,int rows,int columns);
__global__ void _AddD(double *res,double *arr,int arrRows,int arrColumns,int rows,int columns);
__global__ void _2AddD(double *res,double *a,double *b,int bRows,int bColumns,int rows,int columns);
__global__ void _LAddD(double *res,double *arr,double lamda,int arrRows,int arrColumns,int rows,int columns);
__global__ void _MulD(double *res,double *a,double *b,int cDim,int rows,int columns);
__global__ void _GetDiagonalD(double *res,int rows,int columns);
__global__ void _SMulD(double *res,double val,int rows,int columns);
__global__ void _2SMulD(double *res,double *a,double val,int rows,int columns);
__global__ void _DivD(double *res,double val,int rows,int columns);
__global__ void _TransposeD(double *res,double *a,int rows,int columns);
__global__ void _DotD(double *res,double *a,double *b,int bRows,int bColumns,int rows,int columns);
__global__ void _ColAddD(double *res,int col,double val,int rows,int columns);
#endif
