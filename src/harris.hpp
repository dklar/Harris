#include "hls_video.h"
#include <ap_fixed.h>
#include <stdint.h>
#include "hls_stream.h"

#define MAX_WIDTH  1920
#define MAX_HEIGHT 1080



namespace imgProc {

typedef hls::stream<ap_axiu<32,1,1,1> > AXI_STREAM;
typedef hls::Mat<MAX_HEIGHT,MAX_WIDTH,HLS_8UC3>RGB_IMAGE;

enum direction{
	grad0,grad45,grad90,grad135
};
enum type{
	flat,corner,edge
};

struct directedPixel{
	uint8_t pixel;
	direction dir;
};
struct weightPixel{
	uint16_t value;
	type t;
};

class imgFunctions {
public:
	template<int WIDTH, int HEIGHT>
	void Gauss3(uint8_t *imageIn, uint8_t *imageOut);
	template<int WIDTH, int HEIGHT>
	void Gauss5(uint8_t *imageIn, uint8_t *imageOut);
	template<int WIDTH, int HEIGHT>
	void SobelX(uint8_t *imageIn, uint8_t *imageOut);
	template<int WIDTH, int HEIGHT>
	void SobelY(uint8_t *imageIn, uint8_t *imageOut);
	template<int WIDTH, int HEIGHT>
	void Sobel(uint8_t *imageIn, directedPixel *imageOut);
	template<int WIDTH, int HEIGHT>
	void Mul(uint8_t *image1,uint8_t *image2, uint16_t *imageOut);
	template<int WIDTH, int HEIGHT>
	void Dublicate(uint8_t *imageIn, uint8_t *imageOut1, uint8_t *imageOut2);
	template<int WIDTH, int HEIGHT>
	void NonMaxSuppression(directedPixel* imageIn, uint8_t* imageOut);
	template<int WIDTH, int HEIGHT>
	void canny(RGB_IMAGE &src, RGB_IMAGE &dst,int hyst_thres);
	template<int WIDTH, int HEIGHT>
	void MatToGrayArray(RGB_IMAGE &in, uint8_t* out);
	template<int WIDTH, int HEIGHT>
	void ArrayToMat(uint8_t* in, RGB_IMAGE &out);
	template<int WIDTH, int HEIGHT>
	void NonMaxSurpression(weightPixel *imageIn,weightPixel *imageOut);
	template<int WIDTH, int HEIGHT>
	void harris(RGB_IMAGE &src, weightPixel *dst,int thresUp);

};

template<int WIDTH, int HEIGHT>
void MatToGrayArray(RGB_IMAGE &in, uint8_t* out) {
	hls::Scalar<3,uint8_t> pixel_value;
	loopPixel: for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
			in >> pixel_value;
			uint8_t red = (pixel_value.val[0] * 77) >> 8;			//*0.299
			uint8_t green = (pixel_value.val[1] * 150) >> 8;		//*0.587
			uint8_t blue = (pixel_value.val[2] * 28) >> 8;			//0.114
			out[x + y * WIDTH] = red + green + blue;
		}
	}
}

template<int WIDTH, int HEIGHT>
void ArrayToMat(uint8_t* in, RGB_IMAGE &out) {
	hls::Scalar<3,uint8_t> px1;

	backConvertLoop: for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
			px1.val[0] = in[x + y * WIDTH];
			px1.val[1] = in[x + y * WIDTH];
			px1.val[2] = in[x + y * WIDTH];
			out << px1;
		}
	}
}

template<int WIDTH, int HEIGHT>
void Gauss3(uint8_t *imageIn, uint8_t *imageOut) {

	const int K_SIZE = 3;
	uint8_t line_buf[K_SIZE][WIDTH];
	uint8_t window_buf[K_SIZE][K_SIZE];
	const int KERNEL[K_SIZE][K_SIZE] = { { 1, 2, 1 }, { 2, 4, 2 }, { 1, 2, 1 } };

#pragma HLS ARRAY_PARTITION variable=window_buf complete dim=0
#pragma HLS ARRAY_PARTITION variable=KERNEL complete dim=0
#pragma HLS ARRAY_RESHAPE variable=line_buf complete dim=1


	gaussLoop:
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
		#pragma HLS PIPELINE II=1
		#pragma HLS LOOP_FLATTEN off

			int pix_gauss = 0;
			for (int yl = 0; yl < K_SIZE - 1; yl++)
				line_buf[yl][x] = line_buf[yl + 1][x];


			line_buf[K_SIZE - 1][x] = imageIn[x + y * WIDTH];

			for (int yw = 0; yw < K_SIZE; yw++) {
				for (int xw = 0; xw < K_SIZE - 1; xw++) {
					window_buf[yw][xw] = window_buf[yw][xw + 1];
				}
			}

			for (int yw = 0; yw < K_SIZE; yw++)
				window_buf[yw][K_SIZE - 1] = line_buf[yw][x];


			for (int yw = 0; yw < K_SIZE; yw++) {
				for (int xw = 0; xw < K_SIZE; xw++) {
					pix_gauss += window_buf[yw][xw] * KERNEL[yw][xw];
				}
			}

			pix_gauss >>= 4;
			imageOut[x + y * WIDTH] = pix_gauss;
		}
	}
}

template<int WIDTH, int HEIGHT>
void Gauss5(uint8_t *imageIn, uint8_t *imageOut) {

	const int K_SIZE = 5;
	uint8_t line_buf[K_SIZE][WIDTH];
	uint8_t window_buf[K_SIZE][K_SIZE];
	const int KERNEL[K_SIZE][K_SIZE] = { { 1, 4, 6, 4, 1 }, { 4, 16, 24, 16, 4 }, { 6,
			24, 36, 24, 6 }, { 4, 16, 24, 16, 4 }, { 1, 4, 6, 4, 1 } };

#pragma HLS ARRAY_PARTITION variable=window_buf complete dim=0
#pragma HLS ARRAY_PARTITION variable=KERNEL complete dim=0
#pragma HLS ARRAY_RESHAPE variable=line_buf complete dim=1


	gaussLoop:
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
		#pragma HLS PIPELINE II=1
		#pragma HLS LOOP_FLATTEN off

			int pix_gauss = 0;

			for (int yl = 0; yl < K_SIZE - 1; yl++) {
				line_buf[yl][x] = line_buf[yl + 1][x];
			}

			line_buf[K_SIZE - 1][x] = imageIn[x + y * WIDTH];

			for (int yw = 0; yw < K_SIZE; yw++) {
				for (int xw = 0; xw < K_SIZE - 1; xw++) {
					window_buf[yw][xw] = window_buf[yw][xw + 1];
				}
			}

			// write to window buffer
			for (int yw = 0; yw < K_SIZE; yw++) {
				window_buf[yw][K_SIZE - 1] = line_buf[yw][x];
			}

			for (int yw = 0; yw < K_SIZE; yw++) {
				for (int xw = 0; xw < K_SIZE; xw++) {
					pix_gauss += window_buf[yw][xw] * KERNEL[yw][xw];
				}
			}

			pix_gauss >>= 8;
			imageOut[x + y * WIDTH] = pix_gauss;
		}
	}
}

template<int WIDTH, int HEIGHT>
void SobelX(uint8_t *imageIn, uint8_t *imageOut) {
	const int K_SIZE = 3;
	uint8_t line_buf[K_SIZE][WIDTH];
	uint8_t window_buf[K_SIZE][K_SIZE];
	const int KERNEL[K_SIZE][K_SIZE] = { { 1, 0, -1 }, { 2, 0, -2 },
			{ 1, 0, -1 } };

#pragma HLS ARRAY_PARTITION variable=window_buf complete dim=0
#pragma HLS ARRAY_PARTITION variable=KERNEL complete dim=0
#pragma HLS ARRAY_RESHAPE variable=line_buf complete dim=1

	SobelXLoop:
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_FLATTEN off

			int pixel = 0;

			for (int yl = 0; yl < K_SIZE - 1; yl++) {
				line_buf[yl][x] = line_buf[yl + 1][x];
			}

			line_buf[K_SIZE - 1][x] = imageIn[x + y * WIDTH];

			for (int yw = 0; yw < K_SIZE; yw++) {
				for (int xw = 0; xw < K_SIZE - 1; xw++) {
					window_buf[yw][xw] = window_buf[yw][xw + 1];
				}
			}

			for (int yw = 0; yw < K_SIZE; yw++) {
				window_buf[yw][K_SIZE - 1] = line_buf[yw][x];
			}

			for (int yw = 0; yw < K_SIZE; yw++) {
				for (int xw = 0; xw < K_SIZE; xw++) {
					pixel += window_buf[yw][xw] * KERNEL[yw][xw];
				}
			}

			imageOut[x + y * WIDTH] = pixel;
		}
	}
}

template<int WIDTH, int HEIGHT>
void SobelY(uint8_t *imageIn, uint8_t *imageOut) {
	const int K_SIZE = 3;
	uint8_t line_buf[K_SIZE][WIDTH];
	uint8_t window_buf[K_SIZE][K_SIZE];
	const int KERNEL[K_SIZE][K_SIZE] = { { 1, 2, -1 }, { 0, 0, 0 }, { 1, 2, -1 } };

#pragma HLS ARRAY_PARTITION variable=window_buf complete dim=0
#pragma HLS ARRAY_PARTITION variable=KERNEL complete dim=0
#pragma HLS ARRAY_RESHAPE variable=line_buf complete dim=1

	SobelXLoop:
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_FLATTEN off

			uint8_t pixel = 0;

			for (int yl = 0; yl < K_SIZE - 1; yl++) {
				line_buf[yl][x] = line_buf[yl + 1][x];
			}

			line_buf[K_SIZE - 1][x] = imageIn[x + y * WIDTH];

			for (int yw = 0; yw < K_SIZE; yw++) {
				for (int xw = 0; xw < K_SIZE - 1; xw++) {
					window_buf[yw][xw] = window_buf[yw][xw + 1];
				}
			}

			for (int yw = 0; yw < K_SIZE; yw++) {
				window_buf[yw][K_SIZE - 1] = line_buf[yw][x];
			}

			for (int yw = 0; yw < K_SIZE; yw++) {
				for (int xw = 0; xw < K_SIZE; xw++) {
					pixel += window_buf[yw][xw] * KERNEL[yw][xw];
				}
			}

			imageOut[x + y * WIDTH] = pixel;
		}
	}
}

template<int WIDTH, int HEIGHT>
void Sobel(uint8_t *imageIn, directedPixel *imageOut){
    const int KERNEL_SIZE = 3;

    uint8_t line_buf[KERNEL_SIZE][WIDTH];
    uint8_t window_buf[KERNEL_SIZE][KERNEL_SIZE];

    #pragma HLS ARRAY_RESHAPE variable=line_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=window_buf complete dim=0


    const int H_SOBEL_KERNEL[KERNEL_SIZE][KERNEL_SIZE] = {  { 1,  0, -1},
                                                            { 2,  0, -2},
                                                            { 1,  0, -1}   };

    const int V_SOBEL_KERNEL[KERNEL_SIZE][KERNEL_SIZE] = {  { 1,  2,  1},
                                                            { 0,  0,  0},
                                                            {-1, -2, -1}   };

    #pragma HLS ARRAY_PARTITION variable=H_SOBEL_KERNEL complete dim=0
    #pragma HLS ARRAY_PARTITION variable=V_SOBEL_KERNEL complete dim=0

    sobelXY:
    for(int yi = 0; yi < HEIGHT; yi++) {
        for(int xi = 0; xi < WIDTH; xi++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN off


            int pix_sobel;
            direction grad_sobel;


            for(int yl = 0; yl < KERNEL_SIZE - 1; yl++) {
                line_buf[yl][xi] = line_buf[yl + 1][xi];
            }

            line_buf[KERNEL_SIZE - 1][xi] = imageIn[xi + yi*WIDTH];


            for(int yw = 0; yw < KERNEL_SIZE; yw++) {
                for(int xw = 0; xw < KERNEL_SIZE - 1; xw++) {
                    window_buf[yw][xw] = window_buf[yw][xw + 1];
                }
            }

            for(int yw = 0; yw < KERNEL_SIZE; yw++) {
                window_buf[yw][KERNEL_SIZE - 1] = line_buf[yw][xi];
            }


            int pix_h_sobel = 0;
            int pix_v_sobel = 0;

            for(int yw = 0; yw < KERNEL_SIZE; yw++) {
                for(int xw = 0; xw < KERNEL_SIZE; xw++) {
                    pix_h_sobel += window_buf[yw][xw] * H_SOBEL_KERNEL[yw][xw];
                }
            }

            // convolution using by vertical kernel
            for(int yw = 0; yw < KERNEL_SIZE; yw++) {
                for(int xw = 0; xw < KERNEL_SIZE; xw++) {
                    pix_v_sobel += window_buf[yw][xw] * V_SOBEL_KERNEL[yw][xw];
                }
            }

            pix_sobel = hls::sqrt(float(pix_h_sobel * pix_h_sobel + pix_v_sobel * pix_v_sobel));

            // to consider saturation
            if(255 < pix_sobel) {
                pix_sobel = 255;
            }

            // evaluate gradient direction
            int t_int;
            if(pix_h_sobel != 0) {
                t_int = pix_v_sobel * 256 / pix_h_sobel;
            }
            else {
                t_int = 0x7FFFFFFF;
            }

            if(-618 < t_int && t_int <= -106) {
                grad_sobel = grad135;
            }
            else if(-106 < t_int && t_int <= 106) {
                grad_sobel = grad0;
            }

            else if(106 < t_int && t_int < 618) {
                grad_sobel = grad45;
            }
            else {
                grad_sobel = grad90;
            }


            if((KERNEL_SIZE < xi && xi < WIDTH - KERNEL_SIZE) &&
               (KERNEL_SIZE < yi && yi < HEIGHT - KERNEL_SIZE)) {
            	imageOut[xi + yi*WIDTH].pixel = pix_sobel;
            	imageOut[xi + yi*WIDTH].dir  = grad_sobel;
            }
            else {
            	imageOut[xi + yi*WIDTH].pixel = 0;
            	imageOut[xi + yi*WIDTH].dir  = grad_sobel;
            }
        }
    }
}

template<int WIDTH,int HEIGHT>
void RGB2GRAY(uint8_t *imageIn, uint8_t *imageOut){
	cvtColor:
	for(int y=0; y<HEIGHT*3;y++){
		for(int x=0; x<WIDTH*3;x+=3){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
			uint8_t r = imageIn[x + 0 + y * WIDTH];
			uint8_t g = imageIn[x + 1 + y * WIDTH];
			uint8_t b = imageIn[x + 2 + y * WIDTH];
			ap_fixed<16,15> c1 = 0.299;
			ap_fixed<16,15> c2 = 0.587;
			ap_fixed<16,15> c3 = 0.144;
			imageOut[x + y * WIDTH] = r * c1 + g * c2 + b * c3;
		}
	}
}

template<int WIDTH, int HEIGHT>
void GRAY2RGB(uint8_t *imageIn, uint8_t *imageOut) {
	int nextVal = 0;
	cvtColor:
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x ++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_FLATTEN off
			uint8_t v = imageIn[x + y * WIDTH];
			imageOut[nextVal] = v;
			nextVal++;
			imageOut[nextVal] = v;
			nextVal++;
			imageOut[nextVal] = v;
			nextVal++;
		}
	}
}

template<int WIDTH, int HEIGHT>
void NonMaxSuppression(directedPixel* imageIn, uint8_t* imageOut) {
	const int WINDOW_SIZE = 3;
	directedPixel line_buf[WINDOW_SIZE][WIDTH];
	directedPixel window_buf[WINDOW_SIZE][WINDOW_SIZE];

#pragma HLS ARRAY_RESHAPE variable=line_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=window_buf complete dim=0

	nonMaxLoop:
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
		#pragma HLS PIPELINE II=1
		#pragma HLS LOOP_FLATTEN off


			uint8_t value_nms;
			direction grad_nms;


			for (int i = 0; i < WINDOW_SIZE - 1; i++)
				line_buf[i][x] = line_buf[i + 1][x];
			
			line_buf[WINDOW_SIZE - 1][x] = imageIn[x + y * WIDTH];


			for (int y2 = 0; y2 < WINDOW_SIZE; y2++) {
				for (int x2 = 0; x2 < WINDOW_SIZE - 1; x2++) {
					window_buf[y2][x2] = window_buf[y2][x2 + 1];
				}
			}

			for (int i = 0; i < WINDOW_SIZE; i++)
				window_buf[i][WINDOW_SIZE - 1] = line_buf[i][x];
			

			value_nms = window_buf[WINDOW_SIZE / 2][WINDOW_SIZE / 2].pixel;
			grad_nms = window_buf[WINDOW_SIZE / 2][WINDOW_SIZE / 2].dir;

			if (grad_nms == grad0) {
				if (value_nms < window_buf[WINDOW_SIZE / 2][0].pixel
						|| value_nms
								< window_buf[WINDOW_SIZE / 2][WINDOW_SIZE - 1].pixel) {
					value_nms = 0;
				}
			}
			else if (grad_nms == grad45) {
				if (value_nms < window_buf[0][0].pixel
						|| value_nms
								< window_buf[WINDOW_SIZE - 1][WINDOW_SIZE - 1].pixel) {
					value_nms = 0;
				}
			}
			else if (grad_nms == grad90) {
				if (value_nms < window_buf[0][WINDOW_SIZE - 1].pixel
						|| value_nms
								< window_buf[WINDOW_SIZE - 1][WINDOW_SIZE / 2].pixel) {
					value_nms = 0;
				}
			}

			else if (grad_nms == grad135) {
				if (value_nms < window_buf[WINDOW_SIZE - 1][0].pixel
						|| value_nms < window_buf[0][WINDOW_SIZE - 1].pixel) {
					value_nms = 0;
				}
			}


			if ((WINDOW_SIZE < x && x < WIDTH - WINDOW_SIZE)
					&& (WINDOW_SIZE < y && y < HEIGHT - WINDOW_SIZE)) {
				imageOut[x + y * WIDTH] = value_nms;
			} else {
				imageOut[x + y * WIDTH] = 0;
			}
		}
	}
}

template<int WIDTH, int HEIGHT>
void Hysteresis(uint8_t* src, uint8_t* dst, uint8_t thr) {
    hysLoop:
    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN off
            uint8_t pix;
            if(src[x + y*WIDTH] < thr) {
            	pix = 0;
            }else {
            	pix = 255;
            }
            dst[x + y*WIDTH] = pix;
        }
    }
}

template<uint32_t WIDTH, uint32_t HEIGHT>
void ZeroBorder(uint8_t* src, uint8_t* dst,uint32_t size) {

	borderPadding:
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_FLATTEN off
			uint8_t pix = src[x + y * WIDTH];
			if ((size < x && x < WIDTH - size)
					&& (size < y && y < HEIGHT - size)) {
				dst[x + y * WIDTH] = pix;
			} else {
				dst[x + y * WIDTH] = 0;
			}
		}
	}
}

template<int WIDTH, int HEIGHT>
void Mul(uint8_t *image1, uint8_t *image2, uint16_t *imageOut) {
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
			imageOut[x+y*WIDTH] = image1[x+y*WIDTH] * image2[x+y*WIDTH];
		}
	}
}

template<int WIDTH, int HEIGHT>
void Dublicate(uint8_t *imageIn, uint8_t *imageOut1, uint8_t *imageOut2){
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
			imageOut1[x+y*WIDTH] = imageIn[x+y*WIDTH];
			imageOut2[x+y*WIDTH] = imageIn[x+y*WIDTH];
		}
	}
}

template<int WIDTH, int HEIGHT>
void tripleSignal(uint8_t *imageIn, uint8_t *imageOut1, uint8_t *imageOut2,uint8_t *imageOut3){
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
			imageOut1[x+y*WIDTH] = imageIn[x+y*WIDTH];
			imageOut2[x+y*WIDTH] = imageIn[x+y*WIDTH];
			imageOut3[x+y*WIDTH] = imageIn[x+y*WIDTH];
		}
	}
}

template<int WIDTH, int HEIGHT>
void ResponseCalc(uint16_t *sobelXX, uint16_t *sobelYY, uint16_t *sobelXY,int32_t *imageOut){
	//float k = 0.05;//between 0.04-0.06
	int  k = 2621;//13;
	//ap_int<12> k = 2621;

	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_FLATTEN off
			int det = sobelXX[x + y * WIDTH] * sobelYY[x + y * WIDTH]
					- sobelXY[x + y * WIDTH] * sobelXY[x + y * WIDTH];
			int tra = sobelXX[x + y * WIDTH] + sobelYY[x + y * WIDTH];
			int32_t R = det - k* tra * tra;
			imageOut[x+y*WIDTH] = R >> 16;
		}
	}
}

template<int WIDTH, int HEIGHT>
void decide(int32_t *imageIn, weightPixel *imageOut,int low,int high){

	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
			int32_t val = imageIn[x+y*WIDTH];
			if (val < low){
				//Edge
				imageOut[x+y*WIDTH].t = edge;
				imageOut[x+y*WIDTH].value = (-imageIn[x+y*WIDTH]) >> 8;
			}else if(val > high){
				imageOut[x+y*WIDTH].t = corner;
				imageOut[x+y*WIDTH].value = imageIn[x+y*WIDTH] >> 8;

			}else{
				imageOut[x+y*WIDTH].t = flat;
				imageOut[x+y*WIDTH].value = 0;
			}
		}
	}
}

template<int WIDTH, int HEIGHT>
void NonMaxSurpression(weightPixel *imageIn,weightPixel *imageOut){
	const int WINDOW_SIZE = 5;

	weightPixel line_buf[WINDOW_SIZE][WIDTH];
	weightPixel window_buf[WINDOW_SIZE][WINDOW_SIZE];

	for(int i=0;i<WINDOW_SIZE;i++){
		for(int j=0;j<WINDOW_SIZE;j++){
			window_buf[i][j].t=flat;
			window_buf[i][j].value=0;
		}
	}
	for(int i=0;i<WINDOW_SIZE;i++){
		for(int j=0;j<WIDTH;j++){
			line_buf[i][j].t=flat;
			line_buf[i][j].value=0;
		}
	}

#pragma HLS ARRAY_RESHAPE variable=line_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=window_buf complete dim=0

	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_FLATTEN off

			for (int i = 0; i < WINDOW_SIZE - 1; i++)
				line_buf[i][x] = line_buf[i + 1][x];


			line_buf[WINDOW_SIZE - 1][x] = imageIn[x + y * WIDTH];

			for (int y2 = 0; y2 < WINDOW_SIZE; y2++) {
				for (int x2 = 0; x2 < WINDOW_SIZE - 1; x2++) {
					window_buf[y2][x2] = window_buf[y2][x2 + 1];
				}
			}

			for (int i = 0; i < WINDOW_SIZE; i++)
				window_buf[i][WINDOW_SIZE - 1] = line_buf[i][x];

			uint16_t max=0;
			if (window_buf[WINDOW_SIZE/2][WINDOW_SIZE/2].t==corner){
				for (int xw = 0; xw < WINDOW_SIZE; xw++) {
					for (int yw = 0; yw < WINDOW_SIZE; yw++) {
						weightPixel tmp = window_buf[xw][yw];
						if(tmp.t==corner && tmp.value>max)
							max = tmp.value;
					}
				}
				if (window_buf[WINDOW_SIZE/2][WINDOW_SIZE/2].value==max){
					imageOut[x+y*WIDTH].t=corner;
					imageOut[x+y*WIDTH].value = max;
				}else{
					//std::cout <<"Set to 0 \n";
					imageOut[x+y*WIDTH].t=flat;
					imageOut[x+y*WIDTH].value = 0;
				}

			}else{
				imageOut[x+y*WIDTH].t=imageIn[x + y * WIDTH].t;
				imageOut[x+y*WIDTH].value = imageIn[x + y * WIDTH].value;
			}

		}
	}
}

template<int WIDTH, int HEIGHT>
void MinMax(int32_t *imageIn, int32_t *imageOut, int32_t &max) {
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
			int32_t tmp = imageIn[x + y * WIDTH];
			if (tmp > max)
				max = tmp;
			imageOut[x + y * WIDTH] = tmp;
		}

	}
}

/**
 * Harris Corner detector
 *
 */
template<int WIDTH, int HEIGHT>
void harris(RGB_IMAGE &src, weightPixel *dst,int thresUp){

#pragma HLS DATAFLOW
	static uint8_t 		fifo1[WIDTH*HEIGHT*3];
	static uint8_t 		fifo2[WIDTH*HEIGHT];
	static uint8_t 		fifo3[WIDTH*HEIGHT];
	static uint8_t 		fifo4[WIDTH*HEIGHT];
	static uint8_t 		SobelXFIFO[WIDTH*HEIGHT];
	static uint8_t 		SobelYFIFO[WIDTH*HEIGHT];
	static uint8_t 		fifo5[WIDTH*HEIGHT];
	static uint8_t 		fifo6[WIDTH*HEIGHT];
	static uint8_t 		fifo7[WIDTH*HEIGHT];
	static uint8_t 		fifo8[WIDTH*HEIGHT];
	static uint8_t 		fifo9[WIDTH*HEIGHT];
	static uint8_t 		fifoA[WIDTH*HEIGHT];
	static uint16_t 	SobelXX[WIDTH*HEIGHT];
	static uint16_t 	SobelYY[WIDTH*HEIGHT];
	static uint16_t 	SobelXY[WIDTH*HEIGHT];
	static int32_t 		Response[WIDTH*HEIGHT];
	static weightPixel  harris[WIDTH*HEIGHT];
	static int32_t  	min_max[WIDTH*HEIGHT];

	int max=0;

#pragma HLS STREAM variable=fifo1 depth=1 dim=1
#pragma HLS STREAM variable=fifo2 depth=1 dim=1
#pragma HLS STREAM variable=fifo3 depth=1 dim=1
#pragma HLS STREAM variable=fifo4 depth=1 dim=1
#pragma HLS STREAM variable=fifo5 depth=1 dim=1
#pragma HLS STREAM variable=fifo6 depth=1 dim=1
#pragma HLS STREAM variable=fifo7 depth=1 dim=1
#pragma HLS STREAM variable=fifo8 depth=1 dim=1
#pragma HLS STREAM variable=fifo9 depth=1 dim=1
#pragma HLS STREAM variable=fifoA depth=1 dim=1
#pragma HLS STREAM variable=SobelXX depth=1 dim=1
#pragma HLS STREAM variable=SobelYY depth=1 dim=1
#pragma HLS STREAM variable=SobelXY depth=1 dim=1
#pragma HLS STREAM variable=Response depth=1 dim=1
#pragma HLS STREAM variable=SobelXFIFO depth=1 dim=1
#pragma HLS STREAM variable=SobelYFIFO depth=1 dim=1
#pragma HLS STREAM variable=min_max depth=1 dim=1
#pragma HLS STREAM variable=harris depth=1 dim=1

	MatToGrayArray<WIDTH,HEIGHT>(src,fifo1);
	Gauss3<WIDTH,HEIGHT>(fifo1,fifo2);
	Dublicate<WIDTH,HEIGHT>(fifo2,fifo3,fifo4);
	SobelY<WIDTH,HEIGHT>(fifo3,SobelYFIFO);
	SobelX<WIDTH,HEIGHT>(fifo4,SobelXFIFO);
	tripleSignal<WIDTH,HEIGHT>(SobelXFIFO,fifo5,fifo6,fifo7);
	tripleSignal<WIDTH,HEIGHT>(SobelYFIFO,fifo8,fifo9,fifoA);
	Mul<WIDTH,HEIGHT>(fifo5,fifo6,SobelXX);
	Mul<WIDTH,HEIGHT>(fifo8,fifo9,SobelYY);
	Mul<WIDTH,HEIGHT>(fifo7,fifoA,SobelXY);
	ResponseCalc<WIDTH,HEIGHT>(SobelXX,SobelYY,SobelXY,Response);
	MinMax<WIDTH,HEIGHT>(Response,min_max,max);
	decide<WIDTH,HEIGHT>(min_max,harris,42,max-thresUp);
	NonMaxSurpression<WIDTH,HEIGHT>(harris,dst);

}


template<int WIDTH, int HEIGHT>
void canny(RGB_IMAGE &src, RGB_IMAGE &dst,int hyst_thres){

#pragma HLS DATAFLOW
	static uint8_t 		fifo1[WIDTH*HEIGHT*3];
	static uint8_t 		fifo2[WIDTH*HEIGHT];
	static directedPixel 	fifo3[WIDTH*HEIGHT];
	static uint8_t 		fifo4[WIDTH*HEIGHT];
	static uint8_t 		fifo5[WIDTH*HEIGHT];
	static uint8_t 		fifo6[WIDTH*HEIGHT];


#pragma HLS STREAM variable=fifo1 depth=1 dim=1
#pragma HLS STREAM variable=fifo2 depth=1 dim=1
#pragma HLS STREAM variable=fifo3 depth=1 dim=1
#pragma HLS STREAM variable=fifo4 depth=1 dim=1
#pragma HLS STREAM variable=fifo5 depth=1 dim=1
#pragma HLS STREAM variable=fifo6 depth=1 dim=1


	MatToGrayArray<WIDTH,HEIGHT>(src,fifo1);
	Gauss3<WIDTH,HEIGHT>(fifo1,fifo2);
	Sobel<WIDTH,HEIGHT>(fifo2,fifo3);
	NonMaxSuppression<WIDTH,HEIGHT>(fifo3,fifo4);
	Hysteresis<WIDTH,HEIGHT>(fifo4,fifo5,hyst_thres);
	ZeroBorder<WIDTH,HEIGHT>(fifo4,fifo6,5);
	ArrayToMat<WIDTH,HEIGHT>(fifo6, dst);

}
}
