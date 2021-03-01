#include "../src/top.hpp"



void canny_top(AXI_STREAM &Stream_IN,AXI_STREAM &Stream_OUT){
#pragma HLS INTERFACE axis port=Stream_IN
#pragma HLS INTERFACE axis port=Stream_OUT

#pragma HLS DATAFLOW
	RGB_IMAGE 	img1(MAX_HEIGHT,MAX_WIDTH);
	RGB_IMAGE 	img2(MAX_HEIGHT,MAX_WIDTH);

	hls::AXIvideo2Mat(Stream_IN, img1);
	canny<MAX_WIDTH,MAX_HEIGHT>(img1,img2,50);
	hls::Mat2AXIvideo(img2, Stream_OUT);
}

void harris_top(AXI_STREAM &Stream_IN,weightPixel *Stream_OUT,int thresUp){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE s_axilite port=thresUp
#pragma HLS INTERFACE axis port=Stream_IN
#pragma HLS INTERFACE axis port=Stream_OUT

#pragma HLS DATAFLOW
	RGB_IMAGE 	img1(MAX_HEIGHT,MAX_WIDTH);
	RGB_IMAGE 	img2(MAX_HEIGHT,MAX_WIDTH);

	hls::AXIvideo2Mat(Stream_IN, img1);
	harris<MAX_WIDTH,MAX_HEIGHT>(img1,Stream_OUT,thresUp);
	//hls::Mat2AXIvideo(img2, Stream_OUT);
}

