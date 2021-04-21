__global__ void copyImgKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
   int i = threadIdx.x+blockIdx.x*blockDim.x;
   int j = threadIdx.y+blockIdx.y*blockDim.y;

   if (i<width && j<height)
   {
      int adrIn=(i+j*width)*4;
      int adrOut=adrIn;
      unsigned char r,g,b,a;
      r = img_in[adrIn+0];
      g = img_in[adrIn+1];
      b = img_in[adrIn+2];
      a = img_in[adrIn+3];

      img_out[adrOut+0] = r;
      img_out[adrOut+1] = g;
      img_out[adrOut+2] = b;
      img_out[adrOut+3] = a;
   }
}

__device__ float clamp_color(float color){
  if (color > 255) return 255;
  if (color < 0) return 0;
  return color;
}

__global__ void linearTransformKernel(unsigned char* img_in, unsigned char* img_out, int width, int height, float alpha, float beta)
{
   int i = threadIdx.x+blockIdx.x*blockDim.x;
   int j = threadIdx.y+blockIdx.y*blockDim.y;

   if (i<width && j<height)
   {
      int adrIn=(i+j*width)*4;
      int adrOut=adrIn;
      unsigned char r,g,b,a;
      r = img_in[adrIn+0];
      g = img_in[adrIn+1];
      b = img_in[adrIn+2];
      a = img_in[adrIn+3];

      img_out[adrOut+0] = clamp_color(alpha * r + beta);
      img_out[adrOut+1] = clamp_color(alpha * g + beta);
      img_out[adrOut+2] = clamp_color(alpha * b + beta);
      img_out[adrOut+3] = a;
   }
}

__global__ void mirrorKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

  if (i<width && j<height)
  {
    int adrIn=(i+j*width)*4;
    int adrOut=adrIn;
    unsigned char r,g,b,a;
    r = img_in[adrIn+0];
    g = img_in[adrIn+1];
    b = img_in[adrIn+2];
    a = img_in[adrIn+3];

    int adrMirrored = ((width - i) + j * width) * 4;

    float r_new = r;
    float g_new = g;
    float b_new = b;
    float a_new = a;

    if (i >= width/2) {
      r_new = img_in[adrMirrored+0];
      g_new = img_in[adrMirrored+1];
      b_new = img_in[adrMirrored+2];
      a_new = img_in[adrMirrored+3];
    }

    img_out[adrOut+0] = r_new;
    img_out[adrOut+1] = g_new;
    img_out[adrOut+2] = b_new;
    img_out[adrOut+3] = a_new;
  }
}

__global__ void bwKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

  if (i<width && j<height)
  {
     int adrIn=(i+j*width)*4;
     int adrOut=adrIn;
     unsigned char r,g,b,a;
     r = img_in[adrIn+0];
     g = img_in[adrIn+1];
     b = img_in[adrIn+2];
     a = img_in[adrIn+3];

     float grey = (r+g+b) / 3;

     img_out[adrOut+0] = grey;
     img_out[adrOut+1] = grey;
     img_out[adrOut+2] = grey;
     img_out[adrOut+3] = a;
  }
}

__global__ void sobelKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{

   //TODO: Aufgabe 2.5 Kantendetektion mit Sobelfilter Kernel implementieren
   //Kommentieren Sie die folgenden Anweisungen aus um die SX und SY Arrays zu erhalten
   //const float SX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
   //const float SY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};

}
