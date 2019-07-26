#ifndef LIDAR_H
#define LIDAR_H

#include"GpuArray.h"
#include"GpuVector.h"
#include"GpuTensor.h"
#include"TPU.h"
#include<math.h>

class Lidar{
public:
	GpuArray<double> *SystemE(GpuArray<double> &t);
	void TPUCalcTest();
};

#endif
