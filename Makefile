GpuArrayTest:MainGpuArrayTest.o GpuArrayTest.o GpuArrayKernels.o
	nvcc -o GpuArrayTest MainGpuArrayTest.o GpuArrayTest.o GpuArrayKernels.o
GpuVectorExample:MainGpuVectorExample.o GpuVectorExample.o GpuVectorKernels.o
	nvcc -o GpuVectorExample MainGpuVectorExample.o GpuVectorExample.o GpuVectorKernels.o
GpuTensorExample:MainGpuTensorExample.o GpuTensorExample.o GpuArrayKernels.o GpuVectorKernels.o
	nvcc -o GpuTensorExample MainGpuTensorExample.o GpuTensorExample.o GpuArrayKernels.o GpuVectorKernels.o
TPUTest:MainTpuTest.o GpuArrayKernels.o GpuVectorKernels.o TPUTest.o
	nvcc -o TPUTest MainTpuTest.o GpuArrayKernels.o GpuVectorKernels.o TPUTest.o
LidarTPU:MainLidar.o Lidar.o GpuArrayKernels.o GpuVectorKernels.o 
	nvcc -o LidarTPU MainLidar.o Lidar.o GpuArrayKernels.o GpuVectorKernels.o
MainGpuArrayTest.o:MainGpuArrayTest.cu GpuArray.h
	nvcc -c MainGpuArrayTest.cu
GpuArrayTest.o:GpuArrayTest.cu GpuArray.h
	nvcc -c GpuArrayTest.cu
GpuArrayKernels.o:GpuArrayKernels.cu GpuArray.h
	nvcc -c GpuArrayKernels.cu
GpuVectorExample.o:GpuVectorExample.cu GpuVector.h
	nvcc -c GpuVectorExample.cu
MainGpuVectorExample.o:MainGpuVectorExample.cu GpuVector.h
	nvcc -c MainGpuVectorExample.cu
GpuVectorKernels.o:GpuVectorKernels.cu GpuVector.h
	nvcc -c GpuVectorKernels.cu
MainGpuTensorExample.o:MainGpuTensorExample.cu GpuTensor.h
	nvcc -c MainGpuTensorExample.cu
GpuTensorExample.o:GpuTensorExample.cu GpuTensor.h
	nvcc -c GpuTensorExample.cu
MainTpuTest.o:MainTpuTest.cu GpuArray.h GpuVector.h GpuTensor.h
	nvcc -c MainTpuTest.cu
TPUTest.o:TPUTest.cu GpuArray.h GpuVector.h GpuTensor.h
	nvcc -c TPUTest.cu
MainLidar.o:MainLidar.cu
	nvcc -c MainLidar.cu
Lidar.o:Lidar.cu
	nvcc -c Lidar.cu
clean:
	rm *.o || true
	rm GpuArrayTest || true
	rm GpuVectorExample || true
	rm GpuTensorExample || true
	rm TPUTest || true
	rm LidarTPU || true
