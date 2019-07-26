#include "GpuTensorExample.h"

void GpuTensorExample::BasicUsage(){
	GpuVector<int> a(3);
	a.Set(0,2);
	a.Set(1,3);
	a.Set(2,4);
	GpuTensor<double> b(a);
	GpuVector<int> c(3);
	c.IncInit();
	int i = 0;
	while(c.IncContinue()){
		b.Set(c,i);
		c.Inc(a);
		i++;
	}
	b.Prnt();

	GpuVector<int> d0(3),d1(3),d2(3),d3(3),d4(3),d5(3);
	d0.Set(0,0);
	d0.Set(1,1);
	d0.Set(2,2);

	d1.Set(0,0);
	d1.Set(1,2);
	d1.Set(2,1);

	d2.Set(0,1);
	d2.Set(1,0);
	d2.Set(2,2);

	d3.Set(0,1);
	d3.Set(1,2);
	d3.Set(2,0);

	d4.Set(0,2);
	d4.Set(1,0);
	d4.Set(2,1);
	
	d5.Set(0,2);
	d5.Set(1,1);
	d5.Set(2,0);

	GpuTensor<double> t0,t1,t2,t3,t4,t5;
	b.Transpose(t0,d0);
	b.Transpose(t1,d1);
	b.Transpose(t2,d2);
	b.Transpose(t3,d3);
	b.Transpose(t4,d4);
	b.Transpose(t5,d5);
	
	t0.Prnt();
	t1.Prnt();
	t2.Prnt();
	t3.Prnt();
	t4.Prnt();
	t5.Prnt();
}

void GpuTensorExample::Set(){
	GpuVector<int> a(3);
	a = 3;
	GpuTensor<double> b(a);
	GpuArray<double> *t;
	GpuVector<int> s(1);
	t = new GpuArray<double>(3,3);
	*t = 0.0;
	s = 0;
	b.Set(s,t);
	t = new GpuArray<double>(3,3);
	*t = 1.0;
	s = 1;
	b.Set(s,t);
	t = new GpuArray<double>(3,3);
	*t = 2.0;
	s = 2;
	b.Set(s,t);
	b.Prnt();
}

void GpuTensorExample::Mul(){
	GpuVector<int> a(3),b(3);
	a.Set(0,2);
	a.Set(1,3);
	a.Set(2,4);
	b.Set(0,2);
	b.Set(1,4);
	b.Set(2,5);
	GpuTensor<double> t1(a),t2(b);
	t1 = 1.0;
	t2 = 1.0;
	GpuTensor<double> t3;
	t1.Mul(t3,t2);
	t3.Prnt();
}
