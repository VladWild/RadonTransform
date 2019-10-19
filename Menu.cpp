#include "Menu.h"
#include "CannyClass.h"
#include "RadonTransform.h"
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

using namespace cv;
using cv::Mat;

//название окон
std::string cvNameCanny = "cvCanny";
std::string nameOriginalImage = "OriginalImage";
std::string nameRadon = "Radon";

//название трекбаров
std::string cvNameCannyTrackbar = "cvCannyTrackbar";
std::string nameRadonTrackbar = "RadonTrackbar";
std::string nameSearchLines = "SearchLinesTrackbar";

//им€ картинки
//std::string filename = "1234567891011.jpg";
//std::string filename = "Ronaldo.jpg";
//std::string filename = "67.png";
//std::string filename = "123456.jpg";
//std::string filename = "4.png";
//std::string filename = "7.jpg";
//std::string filename = "rTrECqLx4YE.jpg";
//std::string filename = "the_cube_m171110_zo.jpg";
//std::string filename = "21.jpg";
//std::string filename = "6.png";
//std::string filename = "1doroga.jpg";
//std::string filename = "24c154085fda707.jpg";
//std::string filename = "doroga7.jpg";
//std::string filename = "IK0VbB2bp8A.jpg";
//std::string filename = "6.png";
//std::string filename = "1doroga.jpg";
//std::string filename = "xYASbDNCudo.jpg";
//std::string filename = "5555.png";
//std::string filename = "228164.jpg";
//std::string filename = "images.jpg";
//std::string filename = "sdvsrvdrvdrv.jpg";
//std::string filename = "7405.jpg";
//std::string filename = "56.jpg";
std::string filename = "0_39d17_e937dce4_L.jpg";
//std::string filename = "s800 (2).jpg";
//std::string filename = "golden_gate_bridge.jpg";


//им€ файла, в который записываетс€ Mat
const char* filenametxt = "ImagePixelsClassic.txt";

//запись Mat в файл
void writeMatToFile(cv::Mat& m, const char* filename)
{
	std::ofstream fout(filename);
	if (!fout)
	{
		std::cout << "File Not Opened" << std::endl;  
		return;
	}
	for (int i = 0; i<m.rows; i++)
	{
		for (int j = 0; j<m.cols; j++)
		{
			fout << m.at<unsigned short int>(i, j) << " ";
		}
		fout << std::endl;
	}
	fout.close();
}

//глобальные переменые объектов собственных классов
CannyClass operatorCanny(filename);
Radon transformRadon;

//глобальные переменые дл€ трекбаров
const int cannyTreshhold1Max = 1000;
int cannyTreshhold1 = 100;
const int cannyTreshhold2Max = 1000;
int cannyTreshhold2 = 100;
const int operatorSobelMax = 1;
int operatorSobel = 0;
const int contrastRadonMax = 10;
int contrastRadon = 1;
const int brightnessRadonMax = 42000;
int brightnessRadon = 0;
const int angleStepMax = 720;
int angleStep = 360;
const int numberOfLinesMax = 20;
int numberOfLines = 10;
const int maximumDistanceMax = 7;
int maximumDistance = 4;


//трекбар, измен€ющий первый порог
void onCannyTreshold1(int, void*)
{
	operatorCanny.changeCanny(cannyTreshhold1, cannyTreshhold2, operatorSobel);
	cvNamedWindow(cvNameCanny.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(cvNameCanny.c_str(), operatorCanny.getDst());
}

//трекбар, измен€ющий второй порог
void onCannyTreshold2(int, void*)
{
	operatorCanny.changeCanny(cannyTreshhold1, cannyTreshhold2, operatorSobel);
	cvNamedWindow(cvNameCanny.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(cvNameCanny.c_str(), operatorCanny.getDst());
}

//трекбар оператора собел€
void onOperatorSobel(int, void*)
{
	operatorCanny.changeCanny(cannyTreshhold1, cannyTreshhold2, operatorSobel);
	cvNamedWindow(cvNameCanny.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(cvNameCanny.c_str(), operatorCanny.getDst());
}

//трекбар, мен€ющий €ркость преобразовани€ радона
void onContrastRadon(int, void*)
{
	imshow(nameRadon, transformRadon.modifiedImage(contrastRadon, brightnessRadon));
}

//трекбар, мен€ющий €ркость преобразовани€ радона
void onBrightnessRadon(int, void*)
{
	imshow(nameRadon, transformRadon.modifiedImage(contrastRadon, brightnessRadon));
}

//трекбар, мен€ющий шаг угла
void onAngleStep(int, void*) {}

//трекбак, мен€ющий количества искомых границ на изображении
void onNumberOfLines(int, void*) {}

//трекбар, мен€ющий минимально допустимое рассто€ние между максимумами
void onMaximumDistance(int, void*) {}

//вывод информации о преобразовании радона на изображени€х
void myMouseCallbackinfo(int event, int x, int y, int flags, void* param)
{
	switch (event)
	{
		case CV_EVENT_LBUTTONDOWN:
				{
					Mat showModel = transformRadon.showLinesOnModel(numberOfLines, maximumDistance);
					imshow(nameRadon.c_str(), showModel);

					Mat showFinalModelMat = cvarrToMat(operatorCanny.getImage()).clone();
					Mat showFinalModel = transformRadon.showFinalModel(showFinalModelMat, angleStep);
					imshow(nameOriginalImage.c_str(), showFinalModel);

				}
				break;
		case CV_EVENT_RBUTTONDOWN:
				{
					Mat info = transformRadon.info(x, y);
					imshow(nameRadon.c_str(), info);

					Mat show = transformRadon.showLines(x, y, angleStep);
					imshow(cvNameCanny.c_str(), show);

				}
				break;
	}
}

//преобразование радона
void myMouseCallback(int event, int x, int y, int flags, void* param)
{
	switch (event)
	{
		case
			CV_EVENT_LBUTTONDOWN:
		{
			Radon transformRadonNew(operatorCanny.getDst(), angleStep);
			Mat imageOriginal = transformRadonNew.transformRadon(contrastRadon, brightnessRadon, angleStep);
			writeMatToFile(imageOriginal, filenametxt);

			cvNamedWindow(nameRadon.c_str(), CV_WINDOW_AUTOSIZE);
			imshow(nameRadon, imageOriginal);

			transformRadon = transformRadonNew;

			destroyWindow(nameSearchLines);
			namedWindow(nameSearchLines, WINDOW_AUTOSIZE);
			resizeWindow(nameSearchLines, 600, 80);
			createTrackbar("Lines", nameSearchLines.c_str(), &numberOfLines, numberOfLinesMax, onNumberOfLines);
			createTrackbar("Distance", nameSearchLines.c_str(), &maximumDistance, maximumDistanceMax, onMaximumDistance);

			cvSetMouseCallback(nameRadon.c_str(), myMouseCallbackinfo);

			break;
		}
	}
}

//точка входа в класс Menu
void Menu::expectation()
{
	(*this).loadImage();
	(*this).trackBarCanny();
	(*this).trackBarRadon();
	cvSetMouseCallback(cvNameCanny.c_str(), myMouseCallback);
	waitKey(0);
}

//показ трекбаров фильтра Canny
void Menu::trackBarCanny()
{
	namedWindow(cvNameCannyTrackbar, WINDOW_AUTOSIZE);
	resizeWindow(cvNameCannyTrackbar, 600, 120);
	createTrackbar("Threshold1", cvNameCannyTrackbar.c_str(), &cannyTreshhold1, cannyTreshhold1Max, onCannyTreshold1);
	createTrackbar("Threshold2", cvNameCannyTrackbar.c_str(), &cannyTreshhold2, cannyTreshhold2Max, onCannyTreshold2);
	createTrackbar("Sobel op", cvNameCannyTrackbar.c_str(), &operatorSobel, operatorSobelMax, onOperatorSobel);
}

//показ трекбаров преобразованного изображени€ Radon
void Menu::trackBarRadon()
{
	namedWindow(nameRadonTrackbar.c_str(), WINDOW_AUTOSIZE);
	resizeWindow(nameRadonTrackbar.c_str(), 600, 120);
	createTrackbar("Contrast", nameRadonTrackbar, &contrastRadon, contrastRadonMax, onContrastRadon);
	createTrackbar("Brightness", nameRadonTrackbar, &brightnessRadon, brightnessRadonMax, onBrightnessRadon);
	createTrackbar("Angle Step", nameRadonTrackbar, &angleStep, angleStepMax, onAngleStep);
}

//показ исходной картинки
void Menu::loadImage()
{
	cvNamedWindow(nameOriginalImage.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(nameOriginalImage.c_str(), operatorCanny.getImage());
	cvNamedWindow(cvNameCanny.c_str(), CV_WINDOW_AUTOSIZE);
	cvShowImage(cvNameCanny.c_str(), operatorCanny.getDst());
}





