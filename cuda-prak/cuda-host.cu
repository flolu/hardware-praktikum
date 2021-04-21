#include "cuda.h"
#include "cuda-kernels.cu"

void copyImgCuda(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
   unsigned char *img_in_dev, *img_out_dev;
   int size=width*height*4;
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned char));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned char));
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned char),cudaMemcpyHostToDevice);
   copyImgKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
}

void mirrorCuda(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
   unsigned char *img_in_dev, *img_out_dev;
   int size=width*height*4;
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned char));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned char));
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned char),cudaMemcpyHostToDevice);
   mirrorKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
}


void linearTransformCuda(unsigned char* img_in, unsigned char* img_out, int width, int height,float alpha, float beta)
{
   unsigned char *img_in_dev, *img_out_dev;
   int size=width*height*4;
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned char));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned char));
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned char),cudaMemcpyHostToDevice);
   linearTransformKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height,alpha,beta);
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
}

void bwCuda(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
   unsigned char *img_in_dev, *img_out_dev;
   int size=width*height*4;
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned char));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned char));
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned char),cudaMemcpyHostToDevice);
   bwKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
}

void sobelCuda(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  unsigned char *img_in_dev, *img_bw_dev, *img_out_dev;
  int size=width*height*4;
  cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned char));
  cudaMalloc((void**)&img_bw_dev,size*sizeof(unsigned char));
  cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned char));
  dim3 threads(16,16);
  dim3 grid(width/threads.x+1,height/threads.y+1);
  cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned char),cudaMemcpyHostToDevice);
  bwKernel<<<grid,threads>>>(img_in_dev,img_bw_dev,width,height);
  sobelKernel<<<grid,threads>>>(img_bw_dev,img_out_dev,width,height);
  cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);
  cudaFree(img_in_dev);
  cudaFree(img_bw_dev);
  cudaFree(img_out_dev);
}

