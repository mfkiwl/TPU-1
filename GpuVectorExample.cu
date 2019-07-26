#include "GpuVectorExample.h"

void GpuVectorExample::BasicUsage(){
	GpuVector<int> a(5);
	a = 3;
	a.Prnt();
	GpuVector<int> b;
	a.Trunc(b);
	b.Prnt();

}
