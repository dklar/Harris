#include "../src/top.hpp"
#include <cstdlib>
#include <hls_opencv.h>

int main(int argc, char *argv[]) {
	int thresUp = atoi(argv[1]);
	IplImage* src_image = new IplImage;
	IplImage* dst_image = new IplImage;
	AXI_STREAM src_stream, out_stream;
	src_image = cvLoadImage("Testpictures\\test.jpg");
	dst_image = cvCreateImage(cvSize(MAX_WIDTH, MAX_HEIGHT), src_image->depth,
			3);
	IplImage2AXIvideo(src_image, src_stream);
	static weightPixel harris[MAX_WIDTH * MAX_HEIGHT];
	harris_top(src_stream, harris, thresUp);
	cv::Mat image = cv::imread("test.jpg");

	int corn = 0;
	int edg = 0;
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (harris[x + y * MAX_WIDTH].t == corner) {
				cv::Vec3b pixel2;
				pixel2.val[0] = 0;
				pixel2.val[1] = 255;
				pixel2.val[2] = 0;

				if (y > 5 && y < MAX_HEIGHT - 5 && x > 5 && x < MAX_WIDTH) {
					image.at<cv::Vec3b>(y + 0-2, x-2) = pixel2;
					image.at<cv::Vec3b>(y - 1-2, x-2) = pixel2;
					image.at<cv::Vec3b>(y + 1-2, x-2) = pixel2;

					image.at<cv::Vec3b>(y + 0-2, x + 1-2) = pixel2;
					image.at<cv::Vec3b>(y + 1-2, x + 1-2) = pixel2;
					image.at<cv::Vec3b>(y - 1-2, x + 1-2) = pixel2;

					image.at<cv::Vec3b>(y + 0-2, x - 1-2) = pixel2;
					image.at<cv::Vec3b>(y + 1-2, x - 1-2) = pixel2;
					image.at<cv::Vec3b>(y - 1-2, x - 1-2) = pixel2;
					corn++;
				}

			}

		}
	}
	std::cout << "Corner " << corn << "\n";
	std::cout << "Edge " << edg << "\n";
	cv::imwrite("result.jpg", image);
	//AXIvideo2IplImage(out_stream,dst_image);
	//cvSaveImage("move.jpg", dst_image);
	cvReleaseImage(&src_image);
	cvReleaseImage(&dst_image);
}
