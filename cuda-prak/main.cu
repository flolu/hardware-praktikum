#ifndef USEINT
#include "cuda-host.cu"
#else
#include "cuda-host-int.cu"
#endif
#include "helper.hpp"



int main(int argc, char **argv)
{
   int mode=0;
   float alpha,beta;
   if (argc<4)
   {
      printf("Usage: %s operation img_in.png img_out.png [alpha] [beta]\n",argv[0]);
      printf("Where operation is one of: copy,linear,mirror,bw,sobel\n");
      printf("alpha and beta are mandatory for linear transformation operation!\n");
      return 0;
   }
   char *operation=argv[1];
   if (strcmp(operation,"copy")==0)
   {
      mode=0;
   } 
   else if (strcmp(operation,"linear")==0)
   {
      mode=1;
      if (argc<6)
      {
         printf("Linear Transformation requested, please provide alpha and beta!\n");
         return 0;
      }
      alpha=atof(argv[4]);
      beta=atof(argv[5]);
   }
   else if (strcmp(operation,"mirror")==0)
   {
      mode=2;
   }
   else if (strcmp(operation,"bw")==0)
   {
      mode=3;
   }
   else if (strcmp(operation,"sobel")==0)
   {
      mode=4;
   }
   else
   {
         printf("not recognized operation %s, copy image instead\n",operation);
   }
   
   PNG img_in(argv[2]);
   img_in.read_png_file();
#ifndef USEINT
   unsigned char* out_img;
   out_img=(unsigned char*)malloc(img_in.getWidth()*img_in.getHeight()*4);
#else
   unsigned int* out_img;
   out_img=(unsigned int*)malloc(img_in.getWidth()*img_in.getHeight()*sizeof(unsigned int));
#endif
   switch(mode)
   {
      case 1:
         printf("Linear transformation\n");
         linearTransformCuda(img_in.getRawImg(),out_img,img_in.getWidth(),img_in.getHeight(),alpha,beta);
         checkErrors();
         break;
      case 2:
         printf("Mirroring Image\n");
         mirrorCuda(img_in.getRawImg(),out_img,img_in.getWidth(),img_in.getHeight());
         checkErrors();
         break;
      case 3:
         printf("black/white Image\n");
         bwCuda(img_in.getRawImg(),out_img,img_in.getWidth(),img_in.getHeight());
         checkErrors();
         break;
      case 4:
         printf("Sobel Filter\n");
         sobelCuda(img_in.getRawImg(),out_img,img_in.getWidth(),img_in.getHeight());
         checkErrors();
         break;
      default:
         printf("Copy Image\n");
         copyImgCuda(img_in.getRawImg(),out_img,img_in.getWidth(),img_in.getHeight());
         checkErrors();
         break;
   }
   PNG img_out(argv[3],img_in.getWidth(),img_in.getHeight(),out_img);
   img_out.write_png_file();
   return 0;
}
