#pragma once
#ifndef RADONTRANSFORM_H
#define RADONTRANSFORM_H

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <search.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using cv::Mat;

struct characteristics {
	float sin;
	float cos;
};

struct angles {
	characteristics alpha;
	characteristics betta;
	characteristics gamma;
};

struct point {
	int x;
	int y;
};

typedef struct {
	int brightliness;
	point location;
} localMaximum;

class Radon
{
private:
	Mat model;
	Mat image;

	Mat modifImage;
	Mat modelShow;

	angles* anglesArray;

	std::vector<localMaximum> localMaximumsVector;
	std::vector<localMaximum> maximums;

	int toX(int);
	int toY(int);

	bool inspection(int, int);

public:
	Radon();
	Radon(IplImage*, int);

	Mat transformRadon(int, int, int);
	Mat modifiedImage(int, int);
	Mat info(int, int);
	Mat showLines(int, int, int);
	Mat showLinesOnModel(int, int);
	Mat showFinalModel(Mat, int);

	Mat getImage();
	Mat getModifiedImage();
	Mat getModel();

};




#endif