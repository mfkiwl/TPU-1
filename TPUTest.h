#ifndef TPUTEST_H 
#define TPUTEST_H

#include"GpuArray.h"
#include"GpuTensor.h"
#include"GpuVector.h"
#include"TPU.h"

class TPUTest{
public:
	GpuArray<double> *SystemE(GpuArray<double> &a);
	void Test();
};

#endif
