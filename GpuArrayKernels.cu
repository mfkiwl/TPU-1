__global__ void _InitI(int *res,int val,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < rows && y < columns){
		int pos = x*columns + y;
		res[pos] = val;
	}
}

__global__ void _AddI(int *res,int *arr,int arrRows,int arrColumns,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < rows && y < columns){
		int pos = x*columns + y;
		int _x = x % arrRows;
		int _y = y % arrColumns;
		int _pos = _x * arrColumns + _y;
		res[pos] += arr[_pos];
	}
}

__global__ void _LAddI(int *res,int *arr,int lamda,int arrRows,int arrColumns,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < rows && y < columns){
		int pos = x*columns + y;
		int _x = x % arrRows;
		int _y = y % arrColumns;
		int _pos = _x * arrColumns + _y;
		res[pos] += lamda*arr[_pos];
	}
}

__global__ void _MulI(int *res,int *a,int *b,int cDim,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < rows && y < columns){
		int sum = 0;
		for (int i=0; i<cDim; i++){
			sum += a[x*cDim + i] * b[i*cDim + y];
		}
		res[x*columns + y] = sum;
	}
}

__global__ void _InitD(double *res,double val,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		int pos = x*columns + y;
		res[pos] = val;
	}
}

__global__ void _AddD(double *res,double *arr,int arrRows,int arrColumns,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		int pos = x*columns + y;
		int _x = x % arrRows;
		int _y = y % arrColumns;
		int _pos = _x * arrColumns + _y;
		res[pos] += arr[_pos];
	}
}

__global__ void _2AddD(double *res,double *a,double *b,int bRows,int bColumns,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		int pos = x*columns + y;
		int _x = x % bRows;
		int _y = y % bColumns;
		int _pos = _x * bColumns + _y;
		res[pos] = a[pos] + b[_pos];
	}
}

__global__ void _LAddD(double *res,double *arr,double lamda,int arrRows,int arrColumns,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		int pos = x*columns + y;
		int _x = x % arrRows;
		int _y = y % arrColumns;
		int _pos = _x * arrColumns + _y;
		res[pos] += lamda*arr[_pos];
	}
}

__global__ void _MulD(double *res,double *a,double *b,int cDim,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		double sum = 0.0;
		for (int i=0; i<cDim; i++){
			sum += a[x*cDim + i] * b[i*columns + y];
		}
		res[x*columns + y] = sum;
	}
}

__global__ void _SMulD(double *res,double val,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		int pos = x * columns + y;
		res[pos] = res[pos]*val;
	}
}

__global__ void _2SMulD(double *res,double *a,double val,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		int pos = x * columns + y;
		res[pos] = a[pos]*val;
	}
}

__global__ void _DivD(double *res,double val,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		int pos = x * columns + y;
		res[pos] = res[pos]/val;
	}
}

__global__ void _TransposeD(double *res,double *a,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		int pos1 = x * columns + y;
		int pos2 = y * rows + x;
		res[pos1] = a[pos2];
	}
}

__global__ void _GetDiagonalD(double *res,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		if (x != y){
			res[x*columns + y] = 0.0;
		}
	}
}

__global__ void _DotD(double *res,double *a,double *b,int bRows,int bColumns,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		int pos = x * columns + y;
		int _x = x % bRows;
		int _y = y % bColumns;
		int _pos = _x * bColumns + _y;
		res[pos] = a[pos] * b[_pos];
	}
}

__global__ void _ColAddD(double *res,int col,double val,int rows,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	if (x < rows && y < columns){
		int pos = x*columns + y;
		if (y == col){
			res[pos] += val;
		}
	}
}
