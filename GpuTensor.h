#ifndef GPUTENSOR_H
#define GPUTENSOR_H

#include<stdio.h>
#include<stdlib.h>
#include"GpuArray.h"
#include"GpuVector.h"

template<class T>
class GpuTensor{
	GpuVector<int> dim;
	void *p;
public:
	GpuTensor(){
		p = NULL;
	}

	GpuTensor(GpuVector<int> &t){
		dim = t;
		int hDim = dim.Get(0);
		//
		if (dim.GetColumns() == 3){
			GpuArray<T> **ptr;
			p = (void *)malloc(hDim * sizeof(GpuArray<T> *));
			ptr = (GpuArray<T> **)p;
			for (int i=0; i<hDim; i++){
				ptr[i] = new GpuArray<T>(dim.Get(1),dim.Get(2));
			}
		}
		else{
			GpuTensor<T> **ptr;
			GpuVector<int> sDim;
			dim.Trunc(sDim);
			p = (void *)malloc(hDim * sizeof(GpuTensor<T> *));
			ptr = (GpuTensor<T> **)p;
			for (int i=0; i<hDim; i++){
				ptr[i] = new GpuTensor<T>(sDim);
			}
		}
	}

	~GpuTensor(){
		this->Free();
	}

	void Free(){
		if (p != NULL){
			if (dim.GetColumns() == 3){
				GpuArray<T> **ptr;
				ptr = (GpuArray<T> **)p;
				for (int i=0; i<dim.Get(0); i++){
					delete ptr[i];
				}
				free(p);
				p = NULL;
			}
			else{
				GpuTensor<T> **ptr;
				ptr = (GpuTensor<T> **)p;
				for (int i=0; i<dim.Get(0); i++){
					delete ptr[i];
				}
				free(p);
				p = NULL;
			}
		}
	}
	
	void Allocate(GpuVector<int> &t){
		this->Free();
		dim = t;
		int hDim = dim.Get(0);
		//
		if (dim.GetColumns() == 3){
			GpuArray<T> **ptr;
			p = (void *)malloc(hDim * sizeof(GpuArray<T> *));
			ptr = (GpuArray<T> **)p;
			for (int i=0; i<hDim; i++){
				ptr[i] = new GpuArray<T>(dim.Get(1),dim.Get(2));
			}
		}
		else{
			GpuTensor<T> **ptr;
			GpuVector<int> sDim;
			dim.Trunc(sDim);
			p = (void *)malloc(hDim * sizeof(GpuTensor<T> *));
			ptr = (GpuTensor<T> **)p;
			for (int i=0; i<hDim; i++){
				ptr[i] = new GpuTensor<T>(sDim);
			}
		}
	}

	void Set(GpuVector<int> &t,T val){
		if (t.GetColumns() == 3){
			GpuArray<T> **ptr;
			ptr = (GpuArray<T> **)p;
			ptr[t.Get(0)]->Set(t.Get(1),t.Get(2),val);
		}
		else{
			GpuTensor<T> **ptr;
			GpuVector<int> s;
			t.Trunc(s);
			ptr = (GpuTensor<T> **)p;
			ptr[t.Get(0)]->Set(s,val);
		}
	}

	T Get(GpuVector<int> &t){
		if(t.GetColumns() == 3){
			GpuArray<T> **ptr;
			ptr = (GpuArray<T> **)p;
			return ptr[t.Get(0)]->Get(t.Get(1),t.Get(2));
		}
		else{
			GpuTensor<T> **ptr;
			GpuVector<int> s;
			t.Trunc(s);
			ptr = (GpuTensor<T> **)p;
			return ptr[t.Get(0)]->Get(s);
		}
	}

	GpuVector<int> *GetDim() {
		GpuVector<int> *t = &dim;
		return t;
	}

	void Set(GpuVector<int> &t,GpuArray<T> *a){
		if (dim.GetColumns() == 3){
			GpuArray<T> **ptr;
			ptr = (GpuArray<T> **)p;
			delete ptr[t.Get(0)];
			ptr[t.Get(0)] = a;
		}
		else{
			GpuTensor<T> **ptr;
			GpuVector<int> s;
			t.Trunc(s);
			ptr = (GpuTensor<T> **)p;
			ptr[t.Get(0)]->Set(s,a);
		}
	}

	void operator = (GpuTensor<T> &t){
		GpuVector<int> *dm = t.GetDim();
		this->Allocate(*dm);
		//
		if (dim.GetColumns() == 3){
			GpuArray<T> **ptrThis,**ptrT;
			//
			ptrThis = (GpuArray<T> **)p;
			ptrT = (GpuArray<T> **)t.GetP();
			//
			for (int i=0; i<dim.Get(0); i++){
				*ptrThis[i] = *ptrT[i];
			}
		}
		else{
			GpuTensor<T> **ptrThis,**ptrT;
			//
			ptrThis = (GpuTensor<T> **)p;
			ptrT = (GpuTensor<T> **)t.GetP();
			//
			for (int i=0; i<dim.Get(0); i++){
				*ptrThis[i] = *ptrT[i];
			}
		}
	}

	void Init(T val){
		if (dim.GetColumns() == 3){
			GpuArray<T> **t;
			t = (GpuArray<T> **)p;
			for (int i=0; i<dim.Get(0); i++){
				t[i]->Init(val);
			}
		}
		else{
			GpuTensor<T> **t;
			t = (GpuTensor<T> **)p;
			for (int i=0; i<dim.Get(0); i++){
				t[i]->Init(val);
			}
		}
	}

	void operator = (T val){
		this->Init(val);
	}

	void Prnt(){
		if (dim.GetColumns() == 3){
			GpuArray<T> **t;
			t = (GpuArray<T> **)p;
			for (int i=0; i<dim.Get(0); i++){
				t[i]->Prnt();
				printf("\n");
			}
		}
		else{
			GpuTensor<T> **t;
			t = (GpuTensor<T> **)p;
			for (int i=0; i<dim.Get(0); i++){
				printf("\n");
				t[i]->Prnt();
			}
		}
	}

	void *GetP(){
		return p;
	}

	void Add(GpuTensor<T> &a){
		if (dim.GetColumns() == 3){
			GpuArray<T> **t1,**t2;
			t1 = (GpuArray<T> **)p;
			t2 = (GpuArray<T> **)a.GetP();
			for (int i=0; i<dim.Get(0); i++){
				t1[i]->Add(*t2[i]);
			}
		}
		else{
			GpuTensor<T> **t1,**t2;
			t1 = (GpuTensor<T> **)p;
			t2 = (GpuTensor<T> **)a.GetP();
			for (int i=0; i<dim.Get(0); i++){
				t1[i]->Add(*t2[i]);
			}
		}
	}

	void LAdd(double lamda,GpuTensor<T> &t){
		if (dim.GetColumns() == 3){
			GpuArray<T> **t1,**t2;
			t1 = (GpuArray<T> **)p;
			t2 = (GpuArray<T> **)t.GetP();
			for (int i=0; i<dim.Get(0); i++){
				t1[i]->LAdd(lamda,*t2[i]);
			}
		}
		else{
			GpuTensor<T> **t1,**t2;
			t1 = (GpuTensor<T> **)p;
			t2 = (GpuTensor<T> **)t.GetP();
			for (int i=0; i<dim.Get(0); i++){
				t1[i]->LAdd(lamda,*t2[i]);
			}
		}
	}

	void Transpose(GpuTensor<T> &a,GpuVector<int> &t){
		GpuVector<int> aDim(t.GetColumns());
		dim.Shuffle(aDim,t);
		aDim.Prnt();
		
		a.Allocate(aDim);
		GpuVector<int> v1(t.GetColumns());
		GpuVector<int> v2(t.GetColumns());
		v1.IncInit();
		
		while(v1.IncContinue()){
			v1.Shuffle(v2,t);
			a.Set(v2,this->Get(v1));
			v1.Inc(dim);
		}
	}

	void MulH(GpuTensor<T> &a,GpuTensor<T> &t){
		if (dim.GetColumns() == 3){
			GpuArray<T> **ptr1,**ptr2,**ptr3;
			ptr1 = (GpuArray<T> **)p;
			ptr2 = (GpuArray<T> **)t.GetP();
			ptr3 = (GpuArray<T> **)a.GetP();
			for (int i=0; i<dim.Get(0); i++){
				ptr1[i]->Mul(*ptr3[i],*ptr2[i]);
			}
		}
		else{
			GpuTensor<T> **ptr1,**ptr2,**ptr3;
			ptr1 = (GpuTensor<T> **)p;
			ptr2 = (GpuTensor<T> **)t.GetP();
			ptr3 = (GpuTensor<T> **)a.GetP();
			for (int i=0; i<dim.Get(0); i++){
				ptr1[i]->MulH(*ptr3[i],*ptr2[i]);
			}
		}
	}

	void Mul(GpuTensor<T> &a,GpuTensor<T> &t){
		GpuVector<int> *tDim = t.GetDim();
		GpuVector<int> aDim = dim;
		aDim.SetLast(tDim->GetLast());
		a.Allocate(aDim);
		this->MulH(a,t);
	}
};

template<>
class GpuTensor<double>{
	GpuVector<int> dim;
	void *p;
public:
	GpuTensor(){
		p = NULL;
	}

	GpuTensor(GpuVector<int> &t){
		dim = t;
		int hDim = dim.Get(0);
		//
		if (dim.GetColumns() == 3){
			GpuArray<double> **ptr;
			p = (void *)malloc(hDim * sizeof(GpuArray<double> *));
			ptr = (GpuArray<double> **)p;
			for (int i=0; i<hDim; i++){
				ptr[i] = new GpuArray<double>(dim.Get(1),dim.Get(2));
			}
		}
		else{
			GpuTensor<double> **ptr;
			GpuVector<int> sDim;
			dim.Trunc(sDim);
			p = (void *)malloc(hDim * sizeof(GpuTensor<double> *));
			ptr = (GpuTensor<double> **)p;
			for (int i=0; i<hDim; i++){
				ptr[i] = new GpuTensor<double>(sDim);
			}
		}
	}

	~GpuTensor(){
		this->Free();
	}

	void Free(){
		if (p != NULL){
			if (dim.GetColumns() == 3){
				GpuArray<double> **ptr;
				ptr = (GpuArray<double> **)p;
				for (int i=0; i<dim.Get(0); i++){
					delete ptr[i];
				}
				free(p);
				p = NULL;
			}
			else{
				GpuTensor<double> **ptr;
				ptr = (GpuTensor<double> **)p;
				for (int i=0; i<dim.Get(0); i++){
					delete ptr[i];
				}
				free(p);
				p = NULL;
			}
		}
	}
	
	void Allocate(GpuVector<int> &t){
		this->Free();
		dim = t;
		int hDim = dim.Get(0);
		//
		if (dim.GetColumns() == 3){
			GpuArray<double> **ptr;
			p = (void *)malloc(hDim * sizeof(GpuArray<double> *));
			ptr = (GpuArray<double> **)p;
			for (int i=0; i<hDim; i++){
				ptr[i] = new GpuArray<double>(dim.Get(1),dim.Get(2));
			}
		}
		else{
			GpuTensor<double> **ptr;
			GpuVector<int> sDim;
			dim.Trunc(sDim);
			p = (void *)malloc(hDim * sizeof(GpuTensor<double> *));
			ptr = (GpuTensor<double> **)p;
			for (int i=0; i<hDim; i++){
				ptr[i] = new GpuTensor<double>(sDim);
			}
		}
	}

	void Set(GpuVector<int> &t,double val){
		if (t.GetColumns() == 3){
			GpuArray<double> **ptr;
			ptr = (GpuArray<double> **)p;
			ptr[t.Get(0)]->Set(t.Get(1),t.Get(2),val);
		}
		else{
			GpuTensor<double> **ptr;
			GpuVector<int> s;
			t.Trunc(s);
			ptr = (GpuTensor<double> **)p;
			ptr[t.Get(0)]->Set(s,val);
		}
	}

	double Get(GpuVector<int> &t){
		if(t.GetColumns() == 3){
			GpuArray<double> **ptr;
			ptr = (GpuArray<double> **)p;
			return ptr[t.Get(0)]->Get(t.Get(1),t.Get(2));
		}
		else{
			GpuTensor<double> **ptr;
			GpuVector<int> s;
			t.Trunc(s);
			ptr = (GpuTensor<double> **)p;
			return ptr[t.Get(0)]->Get(s);
		}
	}

	GpuVector<int> *GetDim() {
		GpuVector<int> *t = &dim;
		return t;
	}

	void Set(GpuVector<int> &t,GpuArray<double> *a){
		if (dim.GetColumns() == 3){
			GpuArray<double> **ptr;
			ptr = (GpuArray<double> **)p;
			delete ptr[t.Get(0)];
			ptr[t.Get(0)] = a;
		}
		else{
			GpuTensor<double> **ptr;
			GpuVector<int> s;
			t.Trunc(s);
			ptr = (GpuTensor<double> **)p;
			ptr[t.Get(0)]->Set(s,a);
		}
	}

	void operator = (GpuTensor<double> &t){
		GpuVector<int> *dm = t.GetDim();
		this->Allocate(*dm);
		//
		if (dim.GetColumns() == 3){
			GpuArray<double> **ptrThis,**ptrT;
			//
			ptrThis = (GpuArray<double> **)p;
			ptrT = (GpuArray<double> **)t.GetP();
			//
			for (int i=0; i<dim.Get(0); i++){
				*ptrThis[i] = *ptrT[i];
			}
		}
		else{
			GpuTensor<double> **ptrThis,**ptrT;
			//
			ptrThis = (GpuTensor<double> **)p;
			ptrT = (GpuTensor<double> **)t.GetP();
			//
			for (int i=0; i<dim.Get(0); i++){
				*ptrThis[i] = *ptrT[i];
			}
		}
	}

	void Init(double val){
		if (dim.GetColumns() == 3){
			GpuArray<double> **t;
			t = (GpuArray<double> **)p;
			for (int i=0; i<dim.Get(0); i++){
				t[i]->Init(val);
			}
		}
		else{
			GpuTensor<double> **t;
			t = (GpuTensor<double> **)p;
			for (int i=0; i<dim.Get(0); i++){
				t[i]->Init(val);
			}
		}
	}

	void RndInit(double from,double to){
		if (dim.GetColumns() == 3){
			GpuArray<double> **ptr;
			//
			ptr = (GpuArray<double> **)p;
			//
			for (int i=0; i<dim.Get(0); i++){
				ptr[i]->RndInit(from,to);
			}
		}
		else{
			GpuTensor<double> **ptr;
			//
			ptr = (GpuTensor<double> **)p;
			//
			for (int i=0; i<dim.Get(0); i++){
				ptr[i]->RndInit(from,to);
			}
		}
	}

	void SetDiagonal(){
		if (dim.GetColumns() == 3){
			GpuArray<double> **ptr;
			//
			ptr = (GpuArray<double> **)p;
			//
			for (int i=0; i<dim.Get(0); i++){
				ptr[i]->SetDiagonal();
			}
		}
		else{
			GpuTensor<double> **ptr;
			//
			ptr = (GpuTensor<double> **)p;
			//
			for (int i=0; i<dim.Get(0); i++){
				ptr[i]->SetDiagonal();
			}
		}
	}

	void operator = (double val){
		this->Init(val);
	}

	void Prnt(){
		if (dim.GetColumns() == 3){
			GpuArray<double> **t;
			t = (GpuArray<double> **)p;
			for (int i=0; i<dim.Get(0); i++){
				t[i]->Prnt();
				printf("\n");
			}
		}
		else{
			GpuTensor<double> **t;
			t = (GpuTensor<double> **)p;
			for (int i=0; i<dim.Get(0); i++){
				printf("\n");
				t[i]->Prnt();
			}
		}
	}

	void *GetP(){
		return p;
	}

	void Add(GpuTensor<double> &a){
		if (dim.GetColumns() == 3){
			GpuArray<double> **t1,**t2;
			t1 = (GpuArray<double> **)p;
			t2 = (GpuArray<double> **)a.GetP();
			for (int i=0; i<dim.Get(0); i++){
				t1[i]->Add(*t2[i]);
			}
		}
		else{
			GpuTensor<double> **t1,**t2;
			t1 = (GpuTensor<double> **)p;
			t2 = (GpuTensor<double> **)a.GetP();
			for (int i=0; i<dim.Get(0); i++){
				t1[i]->Add(*t2[i]);
			}
		}
	}

	void LAdd(double lamda,GpuTensor<double> &t){
		if (dim.GetColumns() == 3){
			GpuArray<double> **t1,**t2;
			t1 = (GpuArray<double> **)p;
			t2 = (GpuArray<double> **)t.GetP();
			for (int i=0; i<dim.Get(0); i++){
				t1[i]->LAdd(lamda,*t2[i]);
			}
		}
		else{
			GpuTensor<double> **t1,**t2;
			t1 = (GpuTensor<double> **)p;
			t2 = (GpuTensor<double> **)t.GetP();
			for (int i=0; i<dim.Get(0); i++){
				t1[i]->LAdd(lamda,*t2[i]);
			}
		}
	}

	void Transpose(GpuTensor<double> &a,GpuVector<int> &t){
		GpuVector<int> aDim(t.GetColumns());
		dim.Shuffle(aDim,t);
		aDim.Prnt();
		
		a.Allocate(aDim);
		GpuVector<int> v1(t.GetColumns());
		GpuVector<int> v2(t.GetColumns());
		v1.IncInit();
		
		while(v1.IncContinue()){
			v1.Shuffle(v2,t);
			a.Set(v2,this->Get(v1));
			v1.Inc(dim);
		}
	}

	void MulH(GpuTensor<double> &a,GpuTensor<double> &t){
		if (dim.GetColumns() == 3){
			GpuArray<double> **ptr1,**ptr2,**ptr3;
			ptr1 = (GpuArray<double> **)p;
			ptr2 = (GpuArray<double> **)t.GetP();
			ptr3 = (GpuArray<double> **)a.GetP();
			for (int i=0; i<dim.Get(0); i++){
				ptr1[i]->Mul(*ptr3[i],*ptr2[i]);
			}
		}
		else{
			GpuTensor<double> **ptr1,**ptr2,**ptr3;
			ptr1 = (GpuTensor<double> **)p;
			ptr2 = (GpuTensor<double> **)t.GetP();
			ptr3 = (GpuTensor<double> **)a.GetP();
			for (int i=0; i<dim.Get(0); i++){
				ptr1[i]->MulH(*ptr3[i],*ptr2[i]);
			}
		}
	}

	void Mul(GpuTensor<double> &a,GpuTensor<double> &t){
		GpuVector<int> *tDim = t.GetDim();
		GpuVector<int> aDim = dim;
		aDim.SetLast(tDim->GetLast());
		a.Allocate(aDim);
		this->MulH(a,t);
	}
};

#endif
