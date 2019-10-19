#pragma once
#ifndef CANNYCLASS_H
#define CANNYCLASS_H

#include <opencv2/opencv.hpp>

using namespace cv;

class CannyClass
{
private:
	std::string stringFileName;
	IplImage* image;
	IplImage* gray;
	IplImage* dst;

public:
	CannyClass();
	CannyClass(std::string);

	void changeCanny(double, double, int);

	void setStringFileName(std::string);
	void setImage(IplImage*);
	void setGray(IplImage*);
	void setDst(IplImage*);

	IplImage* getImage();
	IplImage* getGray();
	IplImage* getDst();

};

#endif CANNYCLASS_H


