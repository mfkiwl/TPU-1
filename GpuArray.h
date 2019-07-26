#ifndef GPUARRAY_H
#define GPUARRAY_H

#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include"GpuArrayKernels.h"
#include<math.h>

template<class T>
class GpuArray{
	T *p;
	int rows;
	int columns;
	public:
	GpuArray(){
		p = NULL;
	}
	GpuArray(int tRows,int tColumns){
		rows = tRows;
		columns = tColumns;
		p = (T *)malloc(rows*columns*sizeof(T));
	}
	~GpuArray(){
		free(p);
	}
	void Allocate(int tRows,int tColumns){
		rows = tRows;
		columns = tColumns;
		p = (T *)malloc(rows*columns*sizeof(T));
	}
	int GetRows(){
		return rows;
	}
	int GetColumns(){
		return columns;
	}
	T *GetP(){
		return p;
	}
	void Set(int i,int j,T val){
		p[i*columns + j] = val;
	}
	T Get(int i,int j){
		return p[i*columns + j];
	}
};



template<>
class GpuArray<int>{
	int *p;
	int rows;
	int columns;
	public:
	GpuArray(){
		p = NULL;
	}
	GpuArray(int tRows,int tColumns){
		rows = tRows;
		columns = tColumns;
		p = (int *)malloc(rows*columns*sizeof(int));
	}
	~GpuArray(){
		free(p);
	}
	void Allocate(int tRows,int tColumns){
		rows = tRows;
		columns = tColumns;
		p = (int *)malloc(rows*columns*sizeof(int));
	}
	int GetRows(){
		return rows;
	}
	int GetColumns(){
		return columns;
	}
	int *GetP(){
		return p;
	}
	void Set(int i,int j,int val){
		p[i*columns + j] = val;
	}
	int Get(int i,int j){
		return p[i*columns + j];
	}
	void Prnt(){
		for (int i=0; i<rows; i++){
			for (int j=0; j<columns; j++){
				printf("%d ",this->Get(i,j));
			}
			printf("\n");
		}
	}
	void Init(int val){
		int *gpuP;
		int size = rows * columns * sizeof(int);
		cudaMalloc((void **)&gpuP,size);
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);

		_InitI<<<grid,block>>>(gpuP,val,rows,columns);
		
		cudaMemcpy(p,gpuP,size,cudaMemcpyDeviceToHost);
		cudaFree(gpuP);
	}
	void Add(GpuArray<int> &t){
		int *gpuP,*gpuT;
		int size1 = rows * columns * sizeof(int);
		int tRows = t.GetRows();
		int tColumns = t.GetColumns();
		int size2 = tRows * tColumns * sizeof(int);

		cudaMalloc((void **)&gpuP,size1);
		cudaMalloc((void **)&gpuT,size2);

		cudaMemcpy(gpuP,p,size1,cudaMemcpyHostToDevice);
		cudaMemcpy(gpuT,t.GetP(),size2,cudaMemcpyHostToDevice);

		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);

		_AddI<<<grid,block>>>(gpuP,gpuT,tRows,tColumns,rows,columns);

		cudaMemcpy(p,gpuP,size1,cudaMemcpyDeviceToHost);

		cudaFree(gpuP);
		cudaFree(gpuT);
	}
	void LAdd(int lamda,GpuArray<int> &t){
		int *gpuP,*gpuT;
		int size1 = rows * columns * sizeof(int);
		int tRows = t.GetRows();
		int tColumns = t.GetColumns();
		int size2 = tRows * tColumns * sizeof(int);

		cudaMalloc((void **)&gpuP,size1);
		cudaMalloc((void **)&gpuT,size2);

		cudaMemcpy(gpuP,p,size1,cudaMemcpyHostToDevice);
		cudaMemcpy(gpuT,t.GetP(),size2,cudaMemcpyHostToDevice);

		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);

		_LAddI<<<grid,block>>>(gpuP,gpuT,lamda,tRows,tColumns,rows,columns);

		cudaMemcpy(p,gpuP,size1,cudaMemcpyDeviceToHost);

		cudaFree(gpuP);
		cudaFree(gpuT);
	}
	void Mul(GpuArray<int> &a,GpuArray<int> &b){
		int cDim = columns;
		if (cDim != b.GetRows()){
			printf("GpuArray::Mul->Dimensions\n");
			exit(1);
		}
		int col = b.GetColumns();
		a.Allocate(rows,col);
		int *gpuP,*gpuB,*gpuA;

		int sizeP = rows * cDim * sizeof(int);
		int sizeB = cDim * col * sizeof(int);
		int sizeA = rows * col * sizeof(int);
		
		cudaMalloc((void **)&gpuP,sizeP);
		cudaMalloc((void **)&gpuB,sizeB);
		cudaMalloc((void **)&gpuA,sizeA);

		cudaMemcpy(gpuP,p,sizeP,cudaMemcpyHostToDevice);
		cudaMemcpy(gpuB,b.GetP(),sizeB,cudaMemcpyHostToDevice);

		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(col + block.y - 1)/block.y);
		
		_MulI<<<grid,block>>>(gpuA,gpuP,gpuB,cDim,rows,col);

		cudaMemcpy(a.GetP(),gpuA,sizeA,cudaMemcpyDeviceToHost);

		cudaFree(gpuP);
		cudaFree(gpuB);
		cudaFree(gpuA);
	}
};

template<>
class GpuArray<double>{
	double *p;
	int rows;
	int columns;
	public:

	GpuArray(){
		p = NULL;
	}

	GpuArray(int tRows,int tColumns){
		rows = tRows;
		columns = tColumns;
		p = (double *)malloc(rows*columns*sizeof(double));
	}

	GpuArray(const GpuArray<double> &t){
		rows = t.GetRows();
		columns = t.GetColumns();
		p = (double *)malloc(rows*columns*sizeof(double));
		cudaMemcpy(p,t.GetP(),rows*columns*sizeof(double),cudaMemcpyHostToHost);
	}

	~GpuArray(){
		free(p);
	}

	void Allocate(int tRows,int tColumns){
		free(p);
		rows = tRows;
		columns = tColumns;
		p = (double *)malloc(rows*columns*sizeof(double));
	}

	int GetRows() const{
		return rows;
	}

	int GetColumns() const{
		return columns;
	}

	double *GetP() const{
		return p;
	}

	void Set(int i,int j,double val){
		p[i*columns + j] = val;
	}

	double Get(int i,int j) const{
		return p[i*columns + j];
	}

	void Prnt(){
		for (int i=0; i<rows; i++){
			for (int j=0; j<columns; j++){
				printf("%f ",this->Get(i,j));
			}
			printf("\n");
		}
	}

	void Init(double val){
		double *gpuP;
		int size = rows * columns * sizeof(double);
		//Alocate memmory in the gpu
		cudaMalloc((void **)&gpuP,size);
		//Prepare the grid and block dimensions
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);
		//Launch the kernel
		_InitD<<<grid,block>>>(gpuP,val,rows,columns);
		//Copy the result back to cpu
		cudaMemcpy(p,gpuP,size,cudaMemcpyDeviceToHost);
		//free gpu memmory
		cudaFree(gpuP);
	}

	void RndInit(double from,double to){
		for (int i=0; i<rows*columns; i++){
			p[i] = from + ((double)rand()/RAND_MAX)*(to - from);
		}
	}

	void GetDiagonal(GpuArray<double> &t){
		t.Allocate(rows,columns);
		int size = rows * columns * sizeof(double);
		double *gpuP;
		//
		cudaMalloc((void **)&gpuP,size);
		//
		cudaMemcpy(gpuP,p,size,cudaMemcpyHostToDevice);
		//
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);
		//
		_GetDiagonalD<<<grid,block>>>(gpuP,rows,columns);
		//
		cudaMemcpy(t.GetP(),gpuP,size,cudaMemcpyDeviceToHost);
		//
		cudaFree(gpuP);
	}

	void SetDiagonal(){
		int size = rows * columns * sizeof(double);
		double *gpuP;
		//
		cudaMalloc((void **)&gpuP,size);
		//
		cudaMemcpy(gpuP,p,size,cudaMemcpyHostToDevice);
		//
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);
		//
		_GetDiagonalD<<<grid,block>>>(gpuP,rows,columns);
		//
		cudaMemcpy(p,gpuP,size,cudaMemcpyDeviceToHost);
		//
		cudaFree(gpuP);

	}
	
	void Add(GpuArray<double> const &t){
		double *gpuP,*gpuT;
		//
		int size1 = rows * columns * sizeof(double);
		int tRows = t.GetRows();
		int tColumns = t.GetColumns();
		int size2 = tRows * tColumns * sizeof(double);
		//Allocate memmory in the gpu
		cudaMalloc((void **)&gpuP,size1);
		cudaMalloc((void **)&gpuT,size2);
		//Copy data to the gpu
		cudaMemcpy(gpuP,p,size1,cudaMemcpyHostToDevice);
		cudaMemcpy(gpuT,t.GetP(),size2,cudaMemcpyHostToDevice);
		//Prepare the grid and block dimensions
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);
		//Launch the kernel
		_AddD<<<grid,block>>>(gpuP,gpuT,tRows,tColumns,rows,columns);
		//Copy the result back to the cpu
		cudaMemcpy(p,gpuP,size1,cudaMemcpyDeviceToHost);
		//Free gpu memmory
		cudaFree(gpuP);
		cudaFree(gpuT);
	}
	
	//a = this + b
	void Add(GpuArray<double> &a,GpuArray<double> const &b){
		a.Allocate(rows,columns);
		int bRows = b.GetRows();
		int bColumns = b.GetColumns();
		int sizeA = rows*columns*sizeof(double);
		int sizeB = bRows*bColumns*sizeof(double);
		double *gpuP,*gpuA,*gpuB;
		//Allocate memmory in the gpu
		cudaMalloc((void **)&gpuP,sizeA);
		cudaMalloc((void **)&gpuA,sizeA);
		cudaMalloc((void **)&gpuB,sizeB);
		//Copy data to the gpu
		cudaMemcpy(gpuP,p,sizeA,cudaMemcpyHostToDevice);
		cudaMemcpy(gpuB,b.GetP(),sizeB,cudaMemcpyHostToDevice);
		//Prepare the grid and block dimensions
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);
		//Fire the kernel
		_2AddD<<<grid,block>>>(gpuA,gpuP,gpuB,bRows,bColumns,rows,columns);
		//Copy the result back to cpu
		cudaMemcpy(a.GetP(),gpuA,sizeA,cudaMemcpyDeviceToHost);
		//Free gpu memmory
		cudaFree(gpuP);
		cudaFree(gpuA);
		cudaFree(gpuB);
	}
	
	void LAdd(double lamda,GpuArray<double> &t){
		double *gpuP,*gpuT;
		int size1 = rows * columns * sizeof(double);
		int tRows = t.GetRows();
		int tColumns = t.GetColumns();
		int size2 = tRows * tColumns * sizeof(double);
		//Allocate memmory in the gpu
		cudaMalloc((void **)&gpuP,size1);
		cudaMalloc((void **)&gpuT,size2);
		//Copy data from cpu to gpu
		cudaMemcpy(gpuP,p,size1,cudaMemcpyHostToDevice);
		cudaMemcpy(gpuT,t.GetP(),size2,cudaMemcpyHostToDevice);
		//Define grid and block dimensions
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);
		//
		_LAddD<<<grid,block>>>(gpuP,gpuT,lamda,tRows,tColumns,rows,columns);
		//Copy the result bach to cpu
		cudaMemcpy(p,gpuP,size1,cudaMemcpyDeviceToHost);
		//Free gpu space
		cudaFree(gpuP);
		cudaFree(gpuT);
	}

	void operator = (GpuArray<double> const &t){
		free(p);
		rows = t.GetRows();
		columns = t.GetColumns();
		p = (double *)malloc(rows*columns*sizeof(double));
		cudaMemcpy(p,t.GetP(),rows*columns*sizeof(double),cudaMemcpyHostToHost);
	}

	void operator = (double val){
		this->Init(val);
	}

	void Mul(GpuArray<double> &a,GpuArray<double> &b){
		int cDim = columns;
		//Check dimensions
		if (cDim != b.GetRows()){
			printf("GpuArray::Mul->Dimensions\n");
			exit(1);
		}
		int col = b.GetColumns();
		double *gpuP,*gpuB,*gpuA;
		//Allocate the result
		a.Allocate(rows,col);
		//Sizes of this,b and a
		int sizeP = rows * cDim * sizeof(double);
		int sizeB = cDim * col * sizeof(double);
		int sizeA = rows * col * sizeof(double);
		//Allocate memmory in the gpu
		cudaMalloc((void **)&gpuP,sizeP);
		cudaMalloc((void **)&gpuB,sizeB);
		cudaMalloc((void **)&gpuA,sizeA);
		//CopyData from cpu to gpu
		cudaMemcpy(gpuP,p,sizeP,cudaMemcpyHostToDevice);
		cudaMemcpy(gpuB,b.GetP(),sizeB,cudaMemcpyHostToDevice);
		//Define the block and grid dimensions
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(col + block.y - 1)/block.y);
		//
		_MulD<<<grid,block>>>(gpuA,gpuP,gpuB,cDim,rows,col);
		//Copy the result back to cpu	
		cudaMemcpy(a.GetP(),gpuA,sizeA,cudaMemcpyDeviceToHost);
		//Free gpu Memmory
		cudaFree(gpuP);
		cudaFree(gpuB);
		cudaFree(gpuA);
	}

	void Mul(double val){
		double *gpuP;
		int size = rows * columns * sizeof(double);
		//Allocate memory in the gpu
		cudaMalloc((void **)&gpuP,size);
		//Copy Data from cpu to gpu
		cudaMemcpy(gpuP,p,size,cudaMemcpyHostToDevice);
		//Define the block and grid size
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);
		//launch the kernel
		_SMulD<<<grid,block>>>(gpuP,val,rows,columns);
		//Copy the result bach to cpu
		cudaMemcpy(p,gpuP,size,cudaMemcpyDeviceToHost);
		//free gpu Memmory
		cudaFree(gpuP);
	}

	void Div(double val){
		double *gpuP;
		int size = rows * columns * sizeof(double);
		//Allocate memory in the gpu
		cudaMalloc((void **)&gpuP,size);
		//Copy Data from cpu to gpu
		cudaMemcpy(gpuP,p,size,cudaMemcpyHostToDevice);
		//Define the block and grid size
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);
		//launch the kernel
		_DivD<<<grid,block>>>(gpuP,val,rows,columns);
		//Copy the result bach to cpu
		cudaMemcpy(p,gpuP,size,cudaMemcpyDeviceToHost);
		//free gpu Memmory
		cudaFree(gpuP);
	}

	void Transpose(GpuArray<double> &a){
		a.Allocate(columns,rows);
		int size = rows * columns * sizeof(double);
		double *gpuP,*gpuA;
		//Allocate memmory in the gpu
		cudaMalloc((void **)&gpuP,size);
		cudaMalloc((void **)&gpuA,size);
		//Copy Data from the cpu to the gpu
		cudaMemcpy(gpuP,p,size,cudaMemcpyHostToDevice);
		//Define the block and grid size
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((columns + block.x - 1)/block.x,(rows + block.y - 1)/block.y);
		//Fire the kernel
		_TransposeD<<<grid,block>>>(gpuA,gpuP,columns,rows);
		//Copy result back to cpu
		cudaMemcpy(a.GetP(),gpuA,size,cudaMemcpyDeviceToHost);
		//Free gpu Mem
		cudaFree(gpuP);
		cudaFree(gpuA);
	}

	void Dot(GpuArray<double> &a,GpuArray<double> &b){
		int sizeA = rows * columns * sizeof(double);
		int bRows = b.GetRows();
		int bColumns = b.GetColumns();
		int sizeB = bRows * bColumns * sizeof(double);
		//Allocate a
		a.Allocate(rows,columns);
		//
		double *gpuP,*gpuA,*gpuB;
		//Allocate memmory in the gpu
		cudaMalloc((void **)&gpuP,sizeA);
		cudaMalloc((void **)&gpuA,sizeA);
		cudaMalloc((void **)&gpuB,sizeB);
		//Copy Data from the cpu to the gpu
		cudaMemcpy(gpuP,p,sizeA,cudaMemcpyHostToDevice);
		cudaMemcpy(gpuB,b.GetP(),sizeB,cudaMemcpyHostToDevice);
		//Define the block and grid size
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);
		//Fire the kernel
		_DotD<<<grid,block>>>(gpuA,gpuP,gpuB,bRows,bColumns,columns,rows);
		//Copy result back to cpu
		cudaMemcpy(a.GetP(),gpuA,sizeA,cudaMemcpyDeviceToHost);
		//Free gpu Mem
		cudaFree(gpuP);
		cudaFree(gpuA);
		cudaFree(gpuB);
	}

	void  ColumnAdd(int i,double val){
		int size = rows * columns * sizeof (double);
		double *gpuP;
		//
		cudaMalloc((void **)&gpuP,size);
		//
		cudaMemcpy(gpuP,p,size,cudaMemcpyHostToDevice);
		//
		int dimX = 32;
		int dimY = 32;
		dim3 block(dimX,dimY);
		dim3 grid((rows + block.x - 1)/block.x,(columns + block.y - 1)/block.y);
		//
		_ColAddD<<<grid,block>>>(gpuP,i,val,rows,columns);
		//
		cudaMemcpy(p,gpuP,size,cudaMemcpyDeviceToHost);
		//
		cudaFree(gpuP);
	}
	
	void RotX(double theta){
		this->Allocate(3,3);
		this->Set(0,0,1.0);
		this->Set(0,1,0.0);
		this->Set(0,2,0.0);
		this->Set(1,0,0.0);
		this->Set(1,1,cos(theta));
		this->Set(1,2,sin(theta));
		this->Set(2,0,0.0);
		this->Set(2,1,-sin(theta));
		this->Set(2,2,cos(theta));
	}

	void RotY(double theta){
		this->Allocate(3,3);
		this->Set(0,0,cos(theta));
		this->Set(0,1,0.0);
		this->Set(0,2,-sin(theta));
		this->Set(1,0,0.0);
		this->Set(1,1,1.0);
		this->Set(1,2,0.0);
		this->Set(2,0,sin(theta));
		this->Set(2,1,0.0);
		this->Set(2,2,cos(theta));
	}
	
	void RotZ(double theta){
		this->Allocate(3,3);
		this->Set(0,0,cos(theta));
		this->Set(0,1,sin(theta));
		this->Set(0,2,0.0);
		this->Set(1,0,-sin(theta));
		this->Set(1,1,cos(theta));
		this->Set(1,2,0.0);
		this->Set(2,0,0.0);
		this->Set(2,1,0.0);
		this->Set(2,2,1.0);
	}
};


#endif
