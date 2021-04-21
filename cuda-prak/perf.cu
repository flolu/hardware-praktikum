#include "cuda.h"
#include "helper.hpp"
#ifndef USEINT
#include "cuda-kernels.cu"
#else
#include "cuda-kernels-int.cu"
#endif

#ifndef USEINT 
#define ARRAY char
#else
#define ARRAY int
#endif

void runPerf(unsigned ARRAY* img_in,int width, int height,int r,int thr)
{
   unsigned ARRAY* img_in_dev,*img_out_dev,*img_bw_dev;
   unsigned ARRAY* img_out;
#ifndef USEINT
   int size=width*height*4;
#else
   int size=width*height;
#endif
   img_out=(unsigned ARRAY*)malloc(size*sizeof(unsigned ARRAY));
   cudaEvent_t start,end;
   cudaEventCreate(&start);
   cudaEventCreate(&end);
   float tsum[9],t;
   printf("===============\n");
   printf("Kernel Timings:\n");
   printf("===============\n");
   for (int k=0;k<thr;k++)
   {
   memset(tsum,0,9*4);
   dim3 threads(1<<k,1<<k);
   dim3 grid(width/threads.x+1,height/threads.y+1);
   for (int i=0;i<r;i++)
   {
      cudaEventRecord(start);
      cudaMalloc((void**)&img_in_dev,size*sizeof(unsigned ARRAY));
      cudaMalloc((void**)&img_out_dev,size*sizeof(unsigned ARRAY));
      cudaMalloc((void**)&img_bw_dev,size*sizeof(unsigned ARRAY));
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&t,start,end);
      tsum[0]+=t;

      cudaEventRecord(start);
      cudaMemcpy(img_in_dev,img_in,size*sizeof(unsigned ARRAY),cudaMemcpyHostToDevice);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&t,start,end);
      tsum[1]+=t;

      cudaEventRecord(start);
      copyImgKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&t,start,end);
      tsum[2]+=t;

      cudaEventRecord(start);
      mirrorKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&t,start,end);
      tsum[3]+=t;

      cudaEventRecord(start);
      linearTransformKernel<<<grid,threads>>>(img_in_dev,img_out_dev,width,height,1.5,100);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&t,start,end);
      tsum[4]+=t;

      cudaEventRecord(start);
      bwKernel<<<grid,threads>>>(img_in_dev,img_bw_dev,width,height);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&t,start,end);
      tsum[5]+=t;

      cudaEventRecord(start);
//      bwKernel<<<grid,threads>>>(img_in_dev,img_bw_dev,width,height);
      sobelKernel<<<grid,threads>>>(img_bw_dev,img_out_dev,width,height);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&t,start,end);
      tsum[6]+=t;

      cudaEventRecord(start);
      cudaMemcpy(img_out,img_out_dev,size*sizeof(unsigned ARRAY),cudaMemcpyDeviceToHost);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&t,start,end);
      tsum[7]+=t;

      cudaEventRecord(start);
      cudaFree(img_in_dev);
      cudaFree(img_out_dev);
      cudaFree(img_bw_dev);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&t,start,end);
      tsum[8]+=t;

      cudaEventSynchronize(end);

   }
   printf("\nThreadblock Configuration: (%i,%i) Threads\n",threads.x,threads.y);
   printf("-------------------------------------------\n");
   printf("copyKernel: %f ms\n",tsum[2]/r);
   printf("mirrorKernel: %f ms\n",tsum[3]/r);
   printf("linearTransformKernel: %f ms\n",tsum[4]/r);
   printf("bwKernel: %f ms\n",tsum[5]/r);
   printf("sobelKernel: %f ms\n",tsum[6]/r);
   }
   printf("\n===============\n");
   printf("Memory Timings:\n");
   printf("===============\n");
   printf("cudaMalloc: %f ms\n",tsum[0]/r);
   printf("cudaMemcpy H->D: %f ms\n",tsum[1]/r);
   printf("cudaMemcpy D->H: %f ms\n",tsum[7]/r);
   printf("cudaFree: %f ms\n",tsum[8]/r);
   cudaEventDestroy(start);
   cudaEventDestroy(end);
   free(img_out);
}


int main(int argc,char** argv)
{
   if (argc<2)
   {
      printf("Usage: %s img_in.png [runs]\n",argv[0]);
      exit(0);
   }
   int runs=1;
   int thr=6;
   if (argc>2)
      runs=atoi(argv[2]);
   PNG img_in(argv[1]);
   img_in.read_png_file();
   runPerf(img_in.getRawImg(),img_in.getWidth(),img_in.getHeight(),runs,thr);
   return 0;
}
