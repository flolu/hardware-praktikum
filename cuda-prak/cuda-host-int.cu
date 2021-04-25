//Host-Code für Aufgabe 3.2: Host-Code für die Verwendung von unsigned int
#include "cuda.h"
#include "cuda-kernels-int.cu"

//Auskommentierter Code ist eine Kopie von cuda-host.cu
//Überarbeiten Sie ihn für die Verwendung mit unsigned int!

void copyImgCuda(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
   unsigned int *img_in_dev, *img_out_dev;
   int size=width*height;
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned int));
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
   copyImgKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
}

void mirrorCuda(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
/*    unsigned int *img_in_dev, *img_out_dev;
   int size=width*height;
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned int));
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
   mirrorKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   cudaFree(img_in_dev);
   cudaFree(img_out_dev); */
}


void linearTransformCuda(unsigned int* img_in, unsigned int* img_out, int width, int height,float alpha, float beta)
{
   unsigned int *img_in_dev, *img_out_dev;
   int size=width*height;
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned int));
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
   linearTransformKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height,alpha,beta);
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
}

void bwCuda(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
  /*  unsigned int *img_in_dev, *img_out_dev;
   int size=width*height*4;
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned int));
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
   bwKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   cudaFree(img_in_dev);
   cudaFree(img_out_dev); */
}

void sobelCuda(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
/*    unsigned int *img_in_dev, *img_bw_dev, *img_out_dev;
   int size=width*height*4;
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_bw_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned int));
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
   bwKernel<<<grid,threads>>>(img_in_dev,img_bw_dev,width,height);
   sobelKernel<<<grid,threads>>>(img_bw_dev,img_out_dev,width,height);
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
   cudaFree(img_bw_dev); */
}
