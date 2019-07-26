#include "TPUTest.h"

GpuArray<double> *TPUTest::SystemE(GpuArray<double> &t){
	GpuArray<double> *res = new GpuArray<double>();
	*res = t;
	res->Mul(10.0);
	//
	return res;
}

void TPUTest::Test(){
	TPU<TPUTest> tpu;
	///
	GpuArray<double> inpt(10,3);
	inpt.RndInit(0.0,1.0);
	GpuVector<int> v(3);
	v.Set(0,10);
	v.Set(1,3);
	v.Set(2,3);
	GpuTensor<double> var(v);
	var.RndInit(0.0,1.0);
	var.SetDiagonal();
	GpuTensor<double> res;
	//
	tpu.TpuCalc(res,inpt,var);
	res.Prnt();
}
