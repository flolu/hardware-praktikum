#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#define PNG_DEBUG 3
#include <png.h>

#ifndef USEINT
#define RAW_ARRAY unsigned char
#else
#define RAW_ARRAY unsigned int
#endif

#define PNG_GETR(i) (((unsigned int)i&0xFF000000)>>24)
#define PNG_GETG(i) (((unsigned int)i&0x00FF0000)>>16)
#define PNG_GETB(i) (((unsigned int)i&0x0000FF00)>>8)
#define PNG_GETA(i) (((unsigned int)i&0x000000FF)>>0)

#define PNG_OUTPUT(r,g,b,a) (((unsigned int)r<<24)+((unsigned int)g<<16)+\
            ((unsigned int)b<<8)+((unsigned int)a<<0))


void checkErrors()
{
   cudaError_t err=cudaGetLastError();
   if (err!=cudaSuccess)
   {
      printf("ERROR in CUDA part: %s (%i)\n",cudaGetErrorString(err),err);
   }
   else
      printf("Test passed!\n");
}

void abort_(const char * s, ...)
{
   va_list args;
   va_start(args, s);
   vfprintf(stderr, s, args);
   fprintf(stderr, "\n");
   va_end(args);
   abort();
}

class PNG
{
   public:
   PNG(char* file, int _width=0, int _height=0,RAW_ARRAY* _raw_data=NULL)
   {
      file_name=file;
      width=_width;
      height=_height;
      raw_data=_raw_data;
      row_pointers=NULL;
      raw_initialized=false;
   }

   ~PNG()
   {
      if (row_pointers!=NULL)
      {
         /* cleanup heap allocation */
         for (y=0; y<height; y++)
            free(row_pointers[y]);   
         free(row_pointers);
      }
      if (raw_initialized)
      {
         free(raw_data);
      }
   }

   png_structp getImgStruct()
   {
      return png_ptr;
   }

   png_bytep* getImgPtr()
   {
      return row_pointers;
   }

   RAW_ARRAY* getRawImg()
   {
      return (RAW_ARRAY*) raw_data;
   }

   int getWidth()
   {
      return width;
   }

   int getHeight()
   {
      return height;
   }

   void setRawImg(RAW_ARRAY* _raw_data)
   {
      raw_data=_raw_data;
   }

   int getBitDepth()
   {
      return bit_depth;
   }

   void initRowPointers()
   {

      if (row_pointers==NULL)
      {
//         printf("Initializing Row Pointer with size %i %i\n",width,height);
         if (bit_depth == 16) 
         {
            abort_("16 Bit Color-Depth is not supported!");
         }
         else
            rowbytes = width*4;
      
         row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
         for (y=0; y<height; y++)
            row_pointers[y] = (png_byte*) malloc(rowbytes);
      }
   }

   void initRawData()
   {
      if (raw_data==NULL)
      {
         raw_data=(RAW_ARRAY*)malloc(width*height*4);
         raw_initialized=true;
      }

   }

   void read_png_file()
   {
           unsigned char header[8];    // 8 is the maximum size that can be checked
           size_t size;
           /* open file and test for it being a png */
           FILE *fp = fopen(file_name, "rb");
           if (!fp)
                   abort_("[read_png_file] File %s could not be opened for reading", file_name);
           size=fread(header, 1, 8, fp);
           if (png_sig_cmp(header, 0, 8) && size>0)
                   abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);


           /* initialize stuff */
           png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

           if (!png_ptr)
                   abort_("[read_png_file] png_create_read_struct failed");

           info_ptr = png_create_info_struct(png_ptr);
           if (!info_ptr)
                   abort_("[read_png_file] png_create_info_struct failed");

           if (setjmp(png_jmpbuf(png_ptr)))
                   abort_("[read_png_file] Error during init_io");

           png_init_io(png_ptr, fp);
           png_set_sig_bytes(png_ptr, 8);

           png_read_info(png_ptr, info_ptr);

           width = png_get_image_width(png_ptr, info_ptr);
           height = png_get_image_height(png_ptr, info_ptr);
           color_type = png_get_color_type(png_ptr, info_ptr);
           bit_depth = png_get_bit_depth(png_ptr, info_ptr);

           number_of_passes = png_set_interlace_handling(png_ptr);
           png_read_update_info(png_ptr, info_ptr);
          
           printf("Reading PNG File %s (%ix%i, COLOR_TYPE:%i)\n",file_name,width,height,color_type);
           printf("============================================================\n");
           initRowPointers();

           /* read file */
           if (setjmp(png_jmpbuf(png_ptr)))
                   abort_("[read_png_file] Error during read_image");


           png_read_image(png_ptr, row_pointers);

           toRaw();
           fclose(fp);
//           printf("Reading complete\n");
   }

   void toRaw()
   {
      initRawData();
      int i;
      for (y=0;y<height;y++)
      {
         i=0;
#ifndef USEINT
         for (x=0;x<width*4;x+=4)
         {
            int id=x+y*width*4;
            raw_data[id]=(unsigned char)row_pointers[y][i++];
            raw_data[id+1]=(unsigned char)row_pointers[y][i++];
            raw_data[id+2]=(unsigned char)row_pointers[y][i++];
            if (color_type!=PNG_COLOR_TYPE_RGB_ALPHA)
               raw_data[id+3]=(unsigned char)255;
            else
               raw_data[id+3]=(unsigned char)row_pointers[y][i++];
         }
#else
         for (x=0;x<width;x++)
         {
            int id=x+y*width;
            unsigned char r,g,b,a;
            r=(unsigned char)row_pointers[y][i++];
            g=(unsigned char)row_pointers[y][i++];
            b=(unsigned char)row_pointers[y][i++];
            if (color_type!=PNG_COLOR_TYPE_RGB_ALPHA)
               a=(unsigned char)255;
            else
               a=(unsigned char)row_pointers[y][i++];
            raw_data[id]=PNG_OUTPUT(r,g,b,a);
         }
#endif
      }
   }

   void fromRaw()
   {
      initRowPointers();
      for (y=0;y<height;y++)
      {
         
#ifndef USEINT
         for (x=0;x<width*4;x++)                     
         {
            row_pointers[y][x]= (png_byte)raw_data[x+y*width*4];
         }
#else
         int i=0;
         for (x=0;x<width;x++)                     
         {
            row_pointers[y][i++]= (png_byte)PNG_GETR(raw_data[x+y*width]);
            row_pointers[y][i++]= (png_byte)PNG_GETG(raw_data[x+y*width]);
            row_pointers[y][i++]= (png_byte)PNG_GETB(raw_data[x+y*width]);
            row_pointers[y][i++]= (png_byte)PNG_GETA(raw_data[x+y*width]);
         }  
#endif
      }
   }


   void write_png_file()
   {
      printf("Writing PNG File %s (%ix%i, COLOR_TYPE:%i)\n",file_name,width,height,PNG_COLOR_TYPE_RGB_ALPHA);
           /* create file */
           FILE *fp = fopen(file_name, "wb");
           if (!fp)
                   abort_("[write_png_file] File %s could not be opened for writing", file_name);


           /* initialize stuff */
           png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

           if (!png_ptr)
                   abort_("[write_png_file] png_create_write_struct failed");

           info_ptr = png_create_info_struct(png_ptr);
           if (!info_ptr)
                   abort_("[write_png_file] png_create_info_struct failed");

           if (setjmp(png_jmpbuf(png_ptr)))
                   abort_("[write_png_file] Error during init_io");

           png_init_io(png_ptr, fp);


           /* write header */
           if (setjmp(png_jmpbuf(png_ptr)))
                   abort_("[write_png_file] Error during writing header");

           png_set_IHDR(png_ptr, info_ptr, width, height,
                        8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
                        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

           png_write_info(png_ptr, info_ptr);

           fromRaw();


           /* write bytes */
           if (setjmp(png_jmpbuf(png_ptr)))
                   abort_("[write_png_file] Error during writing bytes");

           png_write_image(png_ptr, row_pointers);


           /* end write */
           if (setjmp(png_jmpbuf(png_ptr)))
                   abort_("[write_png_file] Error during end of write");

           png_write_end(png_ptr, NULL);


           fclose(fp);
           printf("Writing complete\n");
   }

   private:
   char *file_name;
   int x, y;

   int width, height, rowbytes;
   png_byte color_type;
   png_byte bit_depth;

   png_structp png_ptr;
   png_infop info_ptr;
   int number_of_passes;
   png_bytep * row_pointers;
   bool raw_initialized;
   RAW_ARRAY* raw_data;

};


