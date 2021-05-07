#define getR(img) (((unsigned int)img&0xFF000000)>>24)
#define getG(img) (((unsigned int)img&0x00FF0000)>>16)
#define getB(img) (((unsigned int)img&0x0000FF00)>>8)
#define getA(img) (((unsigned int)img&0x000000FF)>>0)
#define output(r,g,b,a) (((unsigned int)r<<24)+((unsigned int)g<<16)+((unsigned int)b<<8)+((unsigned int)a<<0))

__global__ void copyImgKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
   // Stelle Pixel im Bild
   int i = threadIdx.x+blockIdx.x*blockDim.x;
   int j = threadIdx.y+blockIdx.y*blockDim.y;

   // durch Bedingung sichergestellt Bildgrenze nicht überschritten
   if (i<width && j<height)
   {
      int adrIn=i+j*width;
      int adrOut=adrIn;
      // Hilfsvariable für Farbwerte der Pixel (Kopie Eingabebild)
      unsigned int color = img_in[adrIn];

      // Ausgabe der Farbwerte für jeden Pixel (Ausgabebild)
      img_out[adrOut] = output(getR(color),getG(color),getB(color),getA(color));
   }
}

__device__ unsigned char checkOverflow(float value) {
  if (value > 255) return 255;
  return (unsigned char)value;
}

__global__ void linearTransformKernel(unsigned int* img_in, unsigned int* img_out, int width, int height, float alpha, float beta)
 {
   // Stelle Pixel im Bild
   int i = threadIdx.x+blockIdx.x * blockDim.x;
   int j = threadIdx.y+blockIdx.y * blockDim.y;

   // durch Bedingung sichergestellt Bildgrenze nicht überschritten
   if (i<width && j<height)
   {
      int adrIn=i+j*width;
      int adrOut=adrIn;
      // Hilfsvariable für Farbwerte der Pixel (Kopie Eingabebild)
      unsigned int color = img_in[adrIn];

      // Veränderung Kontrast, Helligkeit
      unsigned int r = checkOverflow(alpha * getR(color) + beta);
      unsigned int g = checkOverflow(alpha * getG(color) + beta);
      unsigned int b = checkOverflow(alpha * getB(color) + beta);

      // Ausgabe der Farbwerte für jeden Pixel (Ausgabebild)
      img_out[adrOut] = output(r,g,b,getA(color););
   }
 }

__global__ void mirrorKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
  // Stelle Pixel im Bild
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

   // durch Bedingung sichergestellt Bildgrenze nicht überschritten
  if (i<width && j<height)
  {
    int adrIn=i+j*width;
    int adrOut=adrIn;

    unsigned int color;
    if (i >= width/2) {
      // modifizierte Adresse für Pixel auf rechter Seite Bild
      color = img_in[(width - i) + j * width];
    } else {
      // Pixel auf linker Seite Bild
      color = img_in[adrIn];
    }

    // Ausgabe der Farbwerte für jeden Pixel (Ausgabebild)
    img_out[adrOut] = output(getR(color),getG(color),getB(color),getA(color));
  }
}

__global__ void bwKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
  // Stelle Pixel im Bild
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

   // durch Bedingung sichergestellt Bildgrenze nicht überschritten
  if (i<width && j<height)
  {
     int adrIn=i+j*width;
     int adrOut=adrIn;
     // Hilfsvariable für Farbwerte der Pixel (Kopie Eingabebild)
     unsigned int color = img_in[adrIn];

     // Grauer Farbwert
     unsigned char grey = (getR(color) + getG(color) + getB(color)) / 3;

     // Ausgabe der Farbwerte für jeden Pixel (Ausgabebild)
     img_out[adrOut] = output(grey, grey, grey, getA(color););
  }
}

__global__ void sobelKernel(unsigned int* img_in, unsigned int* img_out, int width, int height)
{
  // Stelle Pixel im Bild
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

   // durch Bedingung sichergestellt Bildgrenze nicht überschritten
  if (i<width && j<height)
  {
    int adrIn=i+j*width;
    int adrOut=adrIn;
    unsigned int color_byte = 0;

    // durch Bedingung sichergestellt, dass Kantendetektion nicht auf Randpixel ausgeführt wird
    if (i > 0 && i < width - 1 && j > 0 && j < height - 1) {
      // Definition der partiellen Ableitungen
      const float SY[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
      const float SX[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};

      // Definition Hilfsvariablen
      float horizontal = 0;
      float vertical = 0;
      // durch Schleifen sichergestellt, dass Kantendetektion auf mittleren und auf alle umliegende Pixel ausgeführt (insgesamt 9)
      for (int k = -1; k <= 1; k++) {
        for (int l = -1; l <= 1; l++) {
          // Definition modifizierte Adresse
          int adr = i+k + (j+l)*width;
          int grey = getR(img_in[adr]);
          // Berechnung der partiellen Ableitungen
          horizontal += SY[1+k][1+l] * grey;
          vertical += SX[1+k][1+l] * grey;
        }
      }

      // Berechnung euklidischer Betrag
      float color = sqrt(horizontal*horizontal + vertical*vertical);
			// Begrenzung Wertebereich Farbe
      if (color > 255) color = 255;
      // Ergebnis euklischer Betrag (float) als char gecastet
      color_byte = (unsigned int)color;
    }

    // Ausgabe der Hilfsvariable (color_byte) bei jeden Farbwert, dadurch sichergestellt Bild schwarz-weiß
    img_out[adrOut] = output(color_byte, color_byte, color_byte, getA(color););
  }
}
