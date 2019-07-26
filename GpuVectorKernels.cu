
__global__ void _InitI(int *res,int val,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < columns){
		res[x] = val;
	}
}

__global__ void _TruncI(int *res,int *a,int start,int columns){
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < columns){
		res[x] = a[start+x];
	}

}
