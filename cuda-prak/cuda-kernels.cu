__global__ void copyImgKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
   // Stelle Pixel im Bild
   int i = threadIdx.x+blockIdx.x*blockDim.x;
   int j = threadIdx.y+blockIdx.y*blockDim.y;

   // durch Bedingung sichergestellt Bildgrenze nicht überschritten
   if (i<width && j<height)
   {
      int adrIn=(i+j*width)*4;
      int adrOut=adrIn;
      unsigned char r,g,b,a;
      // Hilfsvariablen für Farbwerte der Pixel (Kopie Eingabebild)
      r = img_in[adrIn+0];
      g = img_in[adrIn+1];
      b = img_in[adrIn+2];
      a = img_in[adrIn+3];

      // Ausgabe der Farbwerte für jeden Pixel (Ausgabebild)
      img_out[adrOut+0] = r;
      img_out[adrOut+1] = g;
      img_out[adrOut+2] = b;
      img_out[adrOut+3] = a;
   }
}

// Hilfsfunktion Begrenzung Wertebereich Farbe
__device__ unsigned char clamp_color(float color){
  if (color > 255) return 255;
  return (unsigned char)color;
}

__global__ void linearTransformKernel(unsigned char* img_in, unsigned char* img_out, int width, int height, float alpha, float beta)
{
   // Stelle Pixel im Bild
   int i = threadIdx.x+blockIdx.x*blockDim.x;
   int j = threadIdx.y+blockIdx.y*blockDim.y;

   // durch Bedingung sichergestellt Bildgrenze nicht überschritten
   if (i<width && j<height)
   {
      int adrIn=(i+j*width)*4;
      int adrOut=adrIn;
      // Hilfsvariablen für Farbwerte der Pixel (Kopie Eingabebild)
      unsigned char r,g,b,a;
      r = img_in[adrIn+0];
      g = img_in[adrIn+1];
      b = img_in[adrIn+2];
      a = img_in[adrIn+3];

      // Veränderung Kontrast, Helligkeit, Ausgabe der Farbwerte für jeden Pixel (Ausgabebild)
      img_out[adrOut+0] = clamp_color(alpha * r + beta);
      img_out[adrOut+1] = clamp_color(alpha * g + beta);
      img_out[adrOut+2] = clamp_color(alpha * b + beta);
      img_out[adrOut+3] = a;
   }
}

__global__ void mirrorKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  // Stelle Pixel im Bild
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

  // durch Bedingung sichergestellt Bildgrenze nicht überschritten
  if (i<width && j<height)
  {
    int adrIn=(i+j*width)*4;
    int adrOut=adrIn;
    unsigned char r, g, b, a;

    // Verwendung modifizierte Adresse, durch Bedingung sichergestellt: nur für Pixel auf rechter Seite Bild angewendet
    if (i >= width/2) {
      // Definition modifizierte Adresse für Pixel auf rechter Seite Bild
      int adrMirrored = ((width - i) + j * width) * 4;
      r = img_in[adrMirrored+0];
      g = img_in[adrMirrored+1];
      b = img_in[adrMirrored+2];
      a = img_in[adrMirrored+3];
    } else {
      // Pixel auf linker Seite Bild
      r = img_in[adrIn+0];
      g = img_in[adrIn+1];
      b = img_in[adrIn+2];
      a = img_in[adrIn+3];
    }

    // Ausgabe der Farbwerte für jeden Pixel (Ausgabebild)
    img_out[adrOut+0] = r;
    img_out[adrOut+1] = g;
    img_out[adrOut+2] = b;
    img_out[adrOut+3] = a;
  }
}

__global__ void bwKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  // Stelle Pixel im Bild
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

  // durch Bedingung sichergestellt Bildgrenze nicht überschritten
  if (i<width && j<height)
  {
     int adrIn=(i+j*width)*4;
     int adrOut=adrIn;
     unsigned char r,g,b,a;
     // Hilfsvariablen für Farbwerte der Pixel (Kopie Eingabebild)
     r = img_in[adrIn+0];
     g = img_in[adrIn+1];
     b = img_in[adrIn+2];
     a = img_in[adrIn+3];

     // Hilfsvariable beinhaltet Erstellung Graubild
     unsigned char grey = (r+g+b) / 3;

     // Ausgabe der Hilfsvariable bei jeden Farbwert, dadurch sichergestellt Bild schwarz-weiß
     img_out[adrOut+0] = grey;
     img_out[adrOut+1] = grey;
     img_out[adrOut+2] = grey;
     img_out[adrOut+3] = a;
  }
}

__global__ void sobelKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  // Stelle Pixel im Bild
  int i = threadIdx.x+blockIdx.x*blockDim.x;
  int j = threadIdx.y+blockIdx.y*blockDim.y;

  // durch Bedingung sichergestellt Bildgrenze nicht überschritten
  if (i<width && j<height)
  {
    int adrIn=(i+j*width)*4;
    int adrOut=adrIn;
    unsigned char a = img_in[adrIn+3];
    unsigned char color_byte = 0;

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
          int adr = (i+k + (j+l)*width) * 4;
          unsigned char c = img_in[adr];
          // Berechnung der partiellen Ableitungen
          horizontal += SY[1+k][1+l] * c;
          vertical += SX[1+k][1+l] * c;
        }
      }

      // Berechnung euklidischer Betrag
      float color = sqrt(horizontal*horizontal + vertical*vertical);
			// Begrenzung Wertebereich Farbe
      if (color > 255) color = 255;
      // Ergebnis euklischer Betrag (float) als char gecastet
      color_byte = (unsigned char)color;
    }

    // Ausgabe der Hilfsvariable (color_byte) bei jeden Farbwert, dadurch sichergestellt Bild schwarz-weiß
    img_out[adrOut+0] = color_byte;
    img_out[adrOut+1] = color_byte;
    img_out[adrOut+2] = color_byte;
    img_out[adrOut+3] = a;
  }
}
