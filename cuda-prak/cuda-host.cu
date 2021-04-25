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
   //TODO: Aufgabe 2.5 Kantendetektion Hostcode
   //0. bwCuda() Code kopieren und um Folgendes erweitern:

   //1. temporäres GPU Array definieren und mittels cudaMalloc anlegen (verwenden Sie als Namen z.B. img_bw_dev)

   //2. bwKernel ausführen und in img_bw_dev schreiben

   //3. sobelKernel ausführen und von img_bw_dev lesen, schreiben in img_out_dev

   //4. img_bw_dev Array wieder frei geben mit cudaFree

}

