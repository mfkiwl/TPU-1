#ifndef GPUVECTOR_H
#define GPUVECTOR_H

#include<stdlib.h>
#include<stdio.h>
#include"GpuVectorKernels.h"

template<class T>
class GpuVector{
	int columns;
	T *p;
public:
	GpuVector(){
		p = NULL;
	}
	~GpuVector(){
		free(p);
	}
	GpuVector(int tColumns){
		columns = tColumns;
		p = (T *)malloc(columns * sizeof(T));
	}
	void Allocate(int tColumns){
		free(p);
		columns = tColumns;
		p = (T *)malloc(columns * sizeof(T));
	}
};

template<>
class GpuVector<int>{
	int columns;
	int *p;
	int incCycles;
public:
	GpuVector(){
		p = NULL;
		incCycles = 0;
	}

	GpuVector(const GpuVector<int> &t){
		columns = t.GetColumns();
		p = (int *)malloc(columns * sizeof(int));
		cudaMemcpy(p,t.GetP(),columns*sizeof(int),cudaMemcpyHostToHost);
		incCycles = 0;
	}

	~GpuVector(){
		free(p);
	}

	GpuVector(int tColumns){
		columns = tColumns;
		p = (int *)malloc(columns * sizeof(int));
		incCycles = 0;
	}

	void Allocate(int tColumns){
		free(p);
		columns = tColumns;
		p = (int *)malloc(columns * sizeof(int));
		incCycles = 0;
	}

	int * GetP() const{
		return p;
	}

	int GetColumns() const{
		return columns;
	}

	int Get(int i) const{
		return p[i];
	}

	void Set(int i,int val){
		p[i] = val;
	}

	void Prnt(){
		for (int i=0; i<columns; i++){
			printf("%d ",p[i]);
		}
		printf("\n");
	}

	void Init(int val){
		int *gpuP;
		int size = columns * sizeof(int);
		//Allocate memory in the gpu
		cudaMalloc((void **)&gpuP,size);
		//Prepare block and grid dimensions
		int dimX = 32;
		dim3 block(dimX);
		dim3 grid((columns + block.x - 1)/block.x);
		//Fire the Kernel
		_InitI<<<grid,block>>>(gpuP,val,columns);
		//Copy Result Back to cpu
		cudaMemcpy(p,gpuP,size,cudaMemcpyDeviceToHost);
		//Free mem in the gpu
		cudaFree(gpuP);
	}

	void operator = (GpuVector<int> const &t){
		free(p);
		columns = t.GetColumns();
		p = (int *)malloc(columns * sizeof(int));
		cudaMemcpy(p,t.GetP(),columns * sizeof(int),cudaMemcpyHostToHost);
	}

	void operator = (int val){
		this->Init(val);
	}

	bool operator == (const GpuVector<int> &t){
		if (columns == t.GetColumns()){
			for (int i=0; i<columns; i++){
				if (this->Get(i) != t.Get(i)){
					return false;
				}
			}
			return true;
		}

		return false;
	}

	bool operator != (const GpuVector<int> &t){
		return (!(*this == t));
	}

	//this = [1 2 3 4] -> t = [2 3 4]
	void Trunc(GpuVector<int> &t,int start = 1){
		t.Allocate(columns - start);
		int sizeP = columns * sizeof(int);
		int col = columns - start;
		int sizeT = col * sizeof(int);
		int *gpuP,*gpuT;
		//Allocate memmory in the gpu
		cudaMalloc((void **)&gpuP,sizeP);
		cudaMalloc((void **)&gpuT,sizeT);
		//Copy Data from cpu to gpu
		cudaMemcpy(gpuP,p,sizeP,cudaMemcpyHostToDevice);
		//Prepare block and grid dimensions
		int dimX = 32;
		dim3 block(dimX);
		dim3 grid((col + block.x - 1)/block.x);
		//Fire the Kernel
		_TruncI<<<grid,block>>>(gpuT,gpuP,start,col);
		//Copy result back to cpu
		cudaMemcpy(t.GetP(),gpuT,sizeT,cudaMemcpyDeviceToHost);
		//Free mem in the gpu
		cudaFree(gpuP);
		cudaFree(gpuT);
	}

	bool IsZero(){
		bool res = true;
		//
		for (int i=0; i<columns; i++){
			res = (this->Get(i) == 0);
			if (!res){
				break;
			}
		}
		return res;
	}
	
	//this = [a b c d e] start = 2 -> [a b 0 0 0 0]
	void SetZero(int start = 0){
		for (int i=start; i<columns; i++){
			this->Set(i,0);
		}
	}

	void IncInit(){
		this->SetZero();
		incCycles = 0;
	}

	void IncH(GpuVector<int> &t,int i){
		if (i == -1){
			this->SetZero();
			incCycles++;
		}
		else{
			int val = this->Get(i);
			val++;
			//
			if (val == t.Get(i)){
				this->SetZero(i);
				this->IncH(t,i-1);
			}
			else{
				this->Set(i,val);
			}
		}
	}
	
	void Inc(GpuVector<int> &t){
		this->IncH(t,columns - 1);
	}

	bool IncContinue(){
		return (incCycles == 0);
	}

	//this = [3 4 2 9] shuffle = [3 0 2 1] -> [9 3 2 4]
	void Shuffle(GpuVector<int> &a,GpuVector<int> &shuffle){
		for (int i=0; i<columns; i++){
			a.Set(i,this->Get(shuffle.Get(i)));
		}
	}

	int GetFirst(){
		return p[0];
	}

	int GetLast(){
		return p[columns - 1];
	}

	void SetLast(int val){
		p[columns - 1] = val;
	}
};

#endif
