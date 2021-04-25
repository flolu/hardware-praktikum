#define getR(img) (((unsigned int)img&0xFF000000)>>24)
#define getG(img) (((unsigned int)img&0x00FF0000)>>16)
#define getB(img) (((unsigned int)img&0x0000FF00)>>8)
#define getA(img) (((unsigned int)img&0x000000FF)>>0)
#define output(r,g,b,a) (((unsigned int)r<<24)+((unsigned int)g<<16)+((unsigned int)b<<8)+((unsigned int)a<<0))

__global__ void copyImgKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
   int i = threadIdx.x+blockIdx.x*blockDim.x;
   int j = threadIdx.y+blockIdx.y*blockDim.y;

   if (i<width && j<height)
   {
      int adrIn=i+j*width;
      int adrOut=adrIn;
      unsigned int color = img_in[adrIn];
      unsigned int r = getR(color);
      unsigned int g = getG(color);
      unsigned int b = getB(color);
      unsigned int a = getA(color);

      img_out[adrOut] = output(r,g,b,a);
   }
}

__device__ unsigned char checkOverflow(float value) {
  if (value > 255) return 255;
  return (unsigned char)value;
}

__global__ void linearTransformKernel(unsigned int* img_in, unsigned int* img_out, int width, int height, float alpha, float beta)
 {
   int i = threadIdx.x+blockIdx.x * blockDim.x;
   int j = threadIdx.y+blockIdx.y * blockDim.y;

   if (i<width && j<height)
   {
      int adrIn=i+j*width;
      int adrOut=adrIn;
      unsigned int color = img_in[adrIn];
      unsigned int r = checkOverflow(alpha * getR(color) + beta);
      unsigned int g = checkOverflow(alpha * getG(color) + beta);
      unsigned int b = checkOverflow(alpha * getB(color) + beta);
      unsigned int a = getA(color);

      img_out[adrOut] = output(r,g,b,a);
   }
 }

__global__ void mirrorKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

  if (i<width && j<height)
  {
    int adrIn=i+j*width;
    int adrOut=adrIn;

    unsigned int color;
    if (i >= width/2) {
      color = img_in[(width - i) + j * width];
    } else {
      color = img_in[adrIn];
    }

    unsigned int r = getR(color);
    unsigned int g = getG(color);
    unsigned int b = getB(color);
    unsigned int a = getA(color);

    img_out[adrOut] = output(r,g,b,a);
  }
}

__global__ void bwKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

  if (i<width && j<height)
  {
     int adrIn=i+j*width;
     int adrOut=adrIn;
     unsigned int color = img_in[adrIn];
     unsigned int a = getA(color);
     unsigned char grey = (getR(color) + getG(color) + getB(color)) / 3;
     img_out[adrOut] = output(grey, grey, grey, a);
  }
}

__global__ void sobelKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

  if (i<width && j<height)
  {
    int adrIn=i+j*width;
    int adrOut=adrIn;
    unsigned int color = img_in[adrIn];
    unsigned int a = getA(color);
    unsigned char color_byte = 0;

    if (i > 0 && i < width - 1 && j > 0 && j < height - 1) {
      const float SY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
      const float SX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};

      float horizontal = 0;
      float vertical = 0;
      for (int k = -1; k <= 1; k++) {
        for (int l = -1; l <= 1; l++) {
          int adr = i+k + (j+l)*width;
          int grey = getR(img_in[adr]);
          horizontal += SY[1+k][1+l] * grey;
          vertical += SX[1+k][1+l] * grey;
        }
      }

      float color = sqrt(horizontal*horizontal + vertical*vertical);
      if (color > 255) color = 255;
      color_byte = (unsigned char)color;
    }

    img_out[adrOut] = output(color_byte, color_byte, color_byte, a);
  }
}
