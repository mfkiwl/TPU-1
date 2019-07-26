#include "GpuArrayTest.h"

void GpuArrayTest::Example(){
	printf("-Create a 2*3 matrix and initializing it with 5.0\n");
	GpuArray<double> a(2,3);
	a = 5.0;
	printf("a = \n");
	a.Prnt();
	
	printf("-Create a 3*4 random matrix\n");
	GpuArray<double> b(3,4);
	b.RndInit(0.0,1.0);
	printf("b = \n");
	b.Prnt();

	printf("-Create c witch is a copy of a\n");
	GpuArray<double> c = a;
	printf("c = \n");
	c.Prnt();

	printf("-Create d witch is a*b\n");
	GpuArray<double> d;
	a.Mul(d,b);
	printf("d = \n");
	d.Prnt();

	printf("-Create e witch is the transpose of d\n");
	GpuArray<double> e;
	d.Transpose(e);
	printf("e = \n");
	e.Prnt();


	printf("-Create f witch is the dot product of a and c\n");
	GpuArray<double> f;
	a.Dot(f,c);
	printf("f = \n");
	f.Prnt();
}
