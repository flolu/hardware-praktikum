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
      int adrIn=(i+j*width);
      int adrOut=adrIn;
      unsigned int color = img_in[adrIn];
      unsigned int r = getR(color);
      unsigned int g = getG(color);
      unsigned int b = getB(color);
      unsigned int a = getA(color);

      img_out[adrOut] = output(r,g,b,a);
   }
}

__device__ unsigned int checkOverflow(float value) {
  unsigned char byte = (unsigned char) value;
  if (byte > 255) return 255;
  return (unsigned int)byte;
}

__global__ void linearTransformKernel(unsigned int* img_in, unsigned int* img_out, int width, int height, float alpha, float beta)
 {
   int i = threadIdx.x+blockIdx.x * blockDim.x;
   int j = threadIdx.y+blockIdx.y * blockDim.y;

   if (i<width && j<height)
   {
      int adrIn=(i+j*width);
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
   //TODO: Spiegeln
}

__global__ void bwKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
   //TODO: Graubild erstellen
}

__global__ void sobelKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{

   //TODO: Kantendetektion mit Sobelfilter Kernel implementieren
   //Kommentieren Sie die folgenden Anweisungen aus um die SX und SY Arrays zu erhalten
   //const float SX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
   //const float SY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};

}
