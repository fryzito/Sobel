#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "mypgm.h"

#define SIZE 4096
#define N 16384
#define L 1024
#define INF (1<<30)
#define MIN_(a,b) (((a)<(b))?(a):(b))
#define MAX_(a,b) (((a)>(b))?(a):(b))
#define sqr(x) (x)*(x)

typedef float uc;
float minimo =  INF;
float maximo = -INF;

__global__ void kernel(uc *img1, uc *img2,float mini,float maxi){
	int offset = (blockIdx.x * L) + threadIdx.x;

	int left = offset - 1;
	int right = offset + 1;

	int top = offset - N;
	int bottom = offset + N;

	// Esquinas
	int sqrA = top-1;
	int sqrB = top+1;
	int sqrC = bottom-1;
	int sqrD = bottom+1;

	// Mascara Horizontal
	float p = 0.0;  
	p += -1 * img1[sqrA];
	p += 0  * img1[left];
	p += 1  * img1[sqrC];
	p += -2 * img1[top];
	p += 0  * img1[offset];
	p += 2  * img1[bottom];
	p += -1 * img1[sqrB];
	p += 0  * img1[right];
	p += 1  * img1[sqrD];
	p = (float)MAX_BRIGHTNESS * (float)(p - mini) / (float)(maxi - mini);

	// Mascara Vertical
	float q = 0.0;  
	q += -1 * img1[sqrA];
	q += -2 * img1[left];
	q += -1 * img1[sqrC];
	q += 0  * img1[top];
	q += 0  * img1[offset];
	q += 0  * img1[bottom];
	q += 1  * img1[sqrB];
	q += 2  * img1[right];
	q += 1  * img1[sqrD];

	q = (float)MAX_BRIGHTNESS * (float)(p - mini) / (float)(maxi - mini);

	img2[offset] = sqrt(sqr(q) + sqr(p));
}

uc* vectorize(){
	uc *img = new uc[SIZE*SIZE];
	for (size_t i = 0; i < SIZE; i++)
		for (size_t j = 0; j < SIZE; j++){
			size_t pos = (i*SIZE) + j;
			img[pos] = image1[i][j];
			//minimo = MIN_(minimo,img[pos]);
			//maximo = MAX_(maximo,img[pos]);
		}
	return img;
}

void vector_to_matrix( uc* img){
	for (size_t i = 0; i < SIZE; i++)
		for (size_t j = 0; j < SIZE; j++){
			size_t pos = SIZE*i + j;
			image2[i][j] = img[pos];
		}
}

//unsigned char image1[MAX_IMAGESIZE][MAX_IMAGESIZE]

int main(void)
{
	load_image_data();  
	uc *dev_img1;
	uc *dev_img2;

	uc *img1 = vectorize();	
	uc *img2 = new uc[SIZE*SIZE];
	
	cudaMalloc((void**)&dev_img1, SIZE*SIZE*sizeof(uc));
	cudaMalloc((void**)&dev_img2, SIZE*SIZE*sizeof(uc));

	cudaMemcpy( dev_img1, img1, SIZE*SIZE*sizeof(uc), cudaMemcpyHostToDevice);
	

	clock_t kerneltime=clock(); 
	// Haciendo el calculo de tiempo en el device
	kernel <<<N,L>>>(dev_img1, dev_img2,0.0,255.0);
	cudaThreadSynchronize();
	printf("\nsvm kernel time is:%f\n",((double)clock()-kerneltime)/CLOCKS_PER_SEC); 

	cudaMemcpy(img2, dev_img2, SIZE*SIZE*sizeof(uc), cudaMemcpyDeviceToHost);

	vector_to_matrix(img2);

	x_size2 = SIZE;
	y_size2 = SIZE;
	
	save_image_data();

	return 0;
}
