#include "CannyClass.h"

#include <opencv2/opencv.hpp>

using namespace cv;
using cv::Mat;

CannyClass::CannyClass() {}

CannyClass::CannyClass(std::string stringFileNameIn) {
	(*this).stringFileName = stringFileNameIn;
	(*this).image = cvLoadImage(stringFileNameIn.c_str(), 1);
	(*this).gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	(*this).dst = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	cvCvtColor((*this).image, (*this).gray, CV_RGB2GRAY);
	cvCanny((*this).gray, (*this).dst, 100, 100, 3);
}

void CannyClass::changeCanny(double cannyTreshold1, double cannyTreshold2, int operatorSobel) {
	if (operatorSobel == 0) operatorSobel = 3;
	else operatorSobel = 5;
	cvCanny((*this).gray, (*this).dst, cannyTreshold1, cannyTreshold2, operatorSobel);
}

void CannyClass::setStringFileName(std::string stringFileName) {
	(*this).stringFileName = stringFileName;
}

void CannyClass::setImage(IplImage* image) {
	(*this).image = image;
}

void CannyClass::setGray(IplImage* gray) {
	(*this).gray = gray;
}

void CannyClass::setDst(IplImage* dst) {
	(*this).dst = dst;
}

IplImage* CannyClass::getImage() {
	return image;
}

IplImage* CannyClass::getGray() {
	return gray;
}

IplImage* CannyClass::getDst() {
	return dst;
}
