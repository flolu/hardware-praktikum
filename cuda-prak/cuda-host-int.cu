#include "cuda.h"
#include "cuda-kernels-int.cu"

void copyImgCuda(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
   unsigned int *img_in_dev, *img_out_dev;
   int size=width*height;
   // Speicher auf GPU allokieren
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned int));
   // Konfiguration (Threads, Blöcke)
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   // Kopieren der Daten auf Device
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
   // Ausführung copyKernel
   copyImgKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
   // Kopieren der Daten auf Host
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   // Speicher auf GPU freigeben
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
}

void mirrorCuda(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
   unsigned int *img_in_dev, *img_out_dev;
   int size=width*height;
   // Speicher auf GPU allokieren
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned int));
   // Konfiguration (Threads, Blöcke)
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   // Kopieren der Daten auf Device
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
   // Ausführung mirrorKernel
   mirrorKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
   // Kopieren der Daten auf Host
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   // Speicher auf GPU freigeben
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
}


void linearTransformCuda(unsigned int* img_in, unsigned int* img_out, int width, int height,float alpha, float beta)
{
   unsigned int *img_in_dev, *img_out_dev;
   int size=width*height;
   // Speicher auf GPU allokieren
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned int));
   // Konfiguration (Threads, Blöcke)
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   // Kopieren der Daten auf Device
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
   // Ausführung linearKernel
   linearTransformKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height,alpha,beta);
   // Kopieren der Daten auf Host
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   // Speicher auf GPU freigeben
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
}

void bwCuda(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
   unsigned int *img_in_dev, *img_out_dev;
   int size=width*height;
   // Speicher auf GPU allokieren
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned int));
   // Konfiguration (Threads, Blöcke)
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   // Kopieren der Daten auf Device
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
   // Ausführung bwKernel
   bwKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
   // Kopieren der Daten auf Host
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   // Speicher auf GPU freigeben
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
}

void sobelCuda(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
   unsigned int *img_in_dev, *img_bw_dev, *img_out_dev;
   int size=width*height;
   // Speicher auf GPU allokieren (zusätzlich für bw-Bild)
   cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_bw_dev,size*sizeof(unsigned int));
   cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned int));
   // Konfiguration (Threads, Blöcke)
   dim3 threads(16,16);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   // Kopieren der Daten auf Device
   cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned int),cudaMemcpyHostToDevice);
   // Ausführung bwKernel, danach Ausführung sobelKernel
   bwKernel<<<grid,threads>>>(img_in_dev,img_bw_dev,width,height);
   sobelKernel<<<grid,threads>>>(img_bw_dev,img_out_dev,width,height);
   // Kopieren der Daten auf Host
   cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned int),cudaMemcpyDeviceToHost);
   // Speicher auf GPU freigeben (zusätzlich für bw-Bild)
   cudaFree(img_in_dev);
   cudaFree(img_out_dev);
   cudaFree(img_bw_dev);
}
