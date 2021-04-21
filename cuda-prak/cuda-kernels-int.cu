//CUDA Kernels für Verwendung von Integer in Aufgabe 3.2

#define getR(img) (((unsigned int)img&0xFF000000)>>24)
//TODO: weitere Makros definieren

__global__ void copyImgKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
   //TODO: Copy Kernel implementieren
}

__global__ void linearTransformKernel(unsigned int* img_in, unsigned int* img_out, int width, int height, float alpha, float beta)
{
   //TODO: Helligkeit und Kontrast
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
