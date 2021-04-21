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

__global__ void linearTransformKernel(unsigned char* img_in, unsigned char* img_out, int width, int height, float alpha, float beta)
{
   //TODO: Aufgabe 2.2 Helligkeit und Kontrast
}

__global__ void mirrorKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
   //TODO: Aufgabe 2.3 Spiegeln
}

__global__ void bwKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
   //TODO: Aufgabe 2.4 Graubild erstellen
}

__global__ void sobelKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  
   //TODO: Aufgabe 2.5 Kantendetektion mit Sobelfilter Kernel implementieren
   //Kommentieren Sie die folgenden Anweisungen aus um die SX und SY Arrays zu erhalten
   //const float SX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
   //const float SY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};

}
