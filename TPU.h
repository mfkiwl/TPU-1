#ifndef TPU_H
#define TPU_H

#include"GpuVector.h"
#include"GpuArray.h"
#include"GpuTensor.h"

template<class T>
class TPU{
	public:
		static void TpuCalc(GpuTensor<double> &t,
				GpuArray<double> &inpt,
				GpuTensor<double> &var,double dt = 0.0001){
			T system;
			int k1 = inpt.GetColumns();
			int n = inpt.GetRows();
			GpuArray<double> *rs = system.SystemE(inpt);
			int k2 = rs->GetColumns();
			GpuVector<int> a(3);
			a.Set(0,k1);
			a.Set(1,n);
			a.Set(2,k2);
			GpuTensor<double> t1(a);
			//
			GpuArray<double> *nA;
			GpuVector<int> s(1);
			for (int i=0; i<k1; i++){
				GpuArray<double> tmp = inpt;
				tmp.ColumnAdd(i,dt);
				s.Set(0,i);
				nA = system.SystemE(tmp);
				nA->LAdd(-1.0,*rs);
				nA->Div(dt);
				t1.Set(s,nA);
			}
			//
			GpuVector<int> tr(3);
			tr.Set(0,1);
			tr.Set(1,2);
			tr.Set(2,0);
			GpuTensor<double> jac;
			t1.Transpose(jac,tr);
			//
			GpuVector<int> tr2(3);
			tr2.Set(0,0);
			tr2.Set(1,2);
			tr2.Set(2,1);
			GpuTensor<double> jacT;
			jac.Transpose(jacT,tr2);
			//
			GpuTensor<double> tmp;
			jac.Mul(tmp,var);
			tmp.Mul(t,jacT);
		}
};

#endif
