#define _USE_MATH_DEFINES
#include "RadonTransform.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <search.h>

using namespace cv;
using cv::Mat;

bool myfunction(localMaximum i, localMaximum j) {
	return (i.brightliness > j.brightliness);
}

//конструктор по умолчанию
Radon::Radon() : model(0, 0, CV_16UC1) {}

//конструктор с моделью Mat
Radon::Radon(IplImage* image, int numberOfAngles)
{
	model = cvarrToMat(image);
	modelShow = model;
	model.convertTo(model, CV_16UC1);
	anglesArray = new angles[numberOfAngles + 1];
	int i = 0;
	for (float angleCurrent = 0; angleCurrent < 2 * M_PI; angleCurrent += 2 * M_PI / numberOfAngles, i++) {
		anglesArray[i].alpha.sin = sin(angleCurrent);
		anglesArray[i].alpha.cos = cos(angleCurrent);
		anglesArray[i].betta.sin = sin(angleCurrent + M_PI / 2);
		anglesArray[i].betta.cos = cos(angleCurrent + M_PI / 2);
		anglesArray[i].gamma.sin = sin(angleCurrent + 3 * M_PI / 2);
		anglesArray[i].gamma.cos = cos(angleCurrent + 3 * M_PI / 2);
	}
}

//преобразование координаты x
int Radon::toX(int x) {
	return x + model.cols / 2;
}

//преобразование координаты y
int Radon::toY(int y) {
	return model.rows / 2 - y;
}

//проверка выхода координат за пределы изображения
bool Radon::inspection(int xCurrent, int yCurrent) {
	if ((toX(xCurrent) > 0) && (toX(xCurrent) < model.cols) &&
		(toY(yCurrent) > 0) && (toY(yCurrent) < model.rows))
		return false;
	else
		return true;
}

//изменение яркости преобразования радона
Mat Radon::modifiedImage(int contrastRadon, int brightnessRadon) {

	int colorPix;
	Mat img((*this).image.rows, (*this).image.cols, CV_16UC1, Scalar::all(0));

	for (int i = 0; i < img.rows - 1; i++) {
		for (int j = 0; j < img.cols - 1; j++) {
			colorPix = contrastRadon*(*this).image.at<unsigned short int>(i, j) + brightnessRadon;
			if (colorPix > 65535)
				img.at<unsigned short int>(i, j) = 65535;
			else img.at<unsigned short int>(i, j) = colorPix;
		}
	}

	(*this).modifImage = img;
	return (*this).modifImage;
}

//вывод информации о нажатии на окно
Mat Radon::info(int x, int y) {
	Mat info = (*this).modifImage.clone();
	std::ostringstream sstr_x;
	sstr_x << "s=" << y;
	putText(info, sstr_x.str(), Point(2, info.rows - 28), FONT_HERSHEY_COMPLEX, 0.5, Scalar::all(65535), 1.6, 8);
	std::ostringstream sstr_y;
	sstr_y << "alpha=" << x;
	putText(info, sstr_y.str(), Point(2, info.rows - 14), FONT_HERSHEY_COMPLEX, 0.5, Scalar::all(65535), 1.6, 8);
	return info;
}

//показ линий, вдоль которых суммирутся пиксели
Mat Radon::showLines(int alpha, int s, int numberOfAngles) {
	std::cout << "alpha = " << alpha << "\n s = " << s << "   \n\n";
	Mat show = (*this).modelShow.clone();
	Scalar scalar = Scalar::all(156);
	line(show, Point(show.cols / 2, 0), Point(show.cols / 2, show.rows), scalar);
	line(show, Point(0, show.rows / 2), Point(show.cols, show.rows / 2), scalar);
	line(show, Point(show.cols / 2, 0), Point(show.cols / 2 - 5, 10), scalar);
	line(show, Point(show.cols / 2, 0), Point(show.cols / 2 + 5, 10), scalar);
	line(show, Point(show.cols, show.rows / 2), Point(show.cols - 10, show.rows / 2 - 5), scalar);
	line(show, Point(show.cols, show.rows / 2), Point(show.cols - 10, show.rows / 2 + 5), scalar);
	putText(show, "x", Point(show.cols - 24, show.cols / 2 - 24), FONT_HERSHEY_COMPLEX, 0.5, scalar, 1, 8);
	putText(show, "y", Point(show.cols / 2 + 24, 16), FONT_HERSHEY_COMPLEX, 0.5, scalar, 1, 8);

	float alpha_rad = 2 * alpha * M_PI / numberOfAngles;
	int xCurrent = s*cos(alpha_rad);
	int yCurrent = s*sin(alpha_rad);
	int x = toX(xCurrent);
	int y = toY(yCurrent);
	line(show, Point(show.cols / 2, show.rows / 2), Point(x, y), scalar);

	float betta = alpha_rad + M_PI / 2;
	float gamma = alpha_rad + 3 * M_PI / 2;

	int r = sqrt(pow(show.cols, 2) + pow(show.rows, 2));

	int x2Current = xCurrent + r*cos(betta);
	int y2Current = yCurrent + r*sin(betta);
	int x2 = toX(x2Current);
	int y2 = toY(y2Current);
	line(show, Point(x, y), Point(x2, y2), scalar);

	int x4Current = xCurrent + r*cos(gamma);
	int y4Current = yCurrent + r*sin(gamma);
	int x4 = toX(x4Current);
	int y4 = toY(y4Current);
	line(show, Point(x, y), Point(x4, y4), scalar);

	return show;
}

//формирование изображения преобразования Радона
Mat Radon::transformRadon(int contrast, int brightness, int numberOfAngles)
{
	int width = model.cols;
	int heigth = model.rows;
	int diagonal = (int)sqrt(pow(width, 2) + pow(heigth, 2));
	//std::cout << diagonal << " " << diagonal*360 << "\n";

	int k = 2;

	int xCurrent, yCurrent, x, y;
	unsigned int summ = 0;

	Mat image(diagonal / k + 1, numberOfAngles + 1, CV_16UC1, Scalar::all(0));
	(*this).image = image;

	int i = 0;

	int x_pred;
	int y_pred;

	for (int j = 0; j < numberOfAngles; j++) {
		for (int s = 0; s <= diagonal / k; s++) {
			xCurrent = s*anglesArray[j].alpha.cos;
			yCurrent = s*anglesArray[j].alpha.sin;
			for (int r = 0; r < diagonal / 2; r++) {
				x = xCurrent + r*anglesArray[j].betta.cos;
				y = yCurrent + r*anglesArray[j].betta.sin;
				if ((x == x_pred) && (y == y_pred)) continue;
				if (!inspection(x, y))
					summ += model.at<unsigned short int>(toY(y), toX(x));
				x_pred = x;
				y_pred = y;
			}
			for (int r = 0; r < diagonal / k; r++) {
				x = xCurrent + r*anglesArray[j].gamma.cos;
				y = yCurrent + r*anglesArray[j].gamma.sin;
				if ((x == x_pred) && (y == y_pred)) continue;
				if (!inspection(x, y))
					summ += model.at<unsigned short int>(toY(y), toX(x));
				x_pred = x;
				y_pred = y;
			}
			if (summ > 65535) image.at<unsigned short int>(s, i) = 65535;
			else image.at<unsigned short int>(s, i) = summ;
			summ = 0;
		}
		i++;
	}

	delete[] anglesArray;

	(*this).image = image;
	(*this).modifImage = (*this).modifiedImage(contrast, brightness);

	return (*this).modifImage;
}

//поиск локальных максимумов преобразования Радона
Mat Radon::showLinesOnModel(int numberOfLines, int maximumDistance)
{
	maximums.clear();

	localMaximum localMaximumElement;
	for (int i = 0; i < (*this).modifImage.rows; i++) {
		for (int j = 0; j < (*this).modifImage.cols; j++) {
			localMaximumElement.brightliness = (*this).modifImage.at<unsigned short int>(i, j);
			localMaximumElement.location.x = j;
			localMaximumElement.location.y = i;
			localMaximumsVector.push_back(localMaximumElement);
		}
	}

	std::sort(localMaximumsVector.begin(), localMaximumsVector.end(), myfunction);

	std::vector<localMaximum>::iterator it = localMaximumsVector.begin();
	localMaximum lastElement;
	bool pixel = false;
	maximums.push_back(*it);

	for (it = localMaximumsVector.begin(); it != localMaximumsVector.end(); ++it) {
		lastElement = *it;
		for (std::vector<localMaximum>::iterator it_max = maximums.begin(); it_max != maximums.end(); it_max++) {
			if ((abs((*it_max).location.x - lastElement.location.x) <= maximumDistance) ||
				(abs((*it_max).location.y - lastElement.location.y) <= maximumDistance)) {
				pixel = false;
				break;
			}
			else {
				pixel = true;
				continue;
			}
		};
		if (pixel == true) {
			maximums.push_back(lastElement);
			if (maximums.size() == numberOfLines) {
				break;
			}
			else {
				continue;
			}
		}
		else {
			continue;
		}
	};

	for (std::vector<localMaximum>::iterator it_max = maximums.begin(); it_max != maximums.end(); it_max++) {
		localMaximum k = *it_max;
		std::cout << "brightliness = " << k.brightliness
					<< " x=" << k.location.x
					<< " y=" << k.location.y
					<< "\n";
	};

	Mat modified = (*this).modifImage.clone();

	for (std::vector<localMaximum>::iterator it_max = maximums.begin(); it_max != maximums.end(); it_max++) {
		int x1 = (*it_max).location.x - maximumDistance;
		int y1 = (*it_max).location.y - maximumDistance;
		if (x1 <= 0) x1 = 0;
		if (y1 <= 0) y1 = 0;
		Point pt1(x1, y1);
		int x2 = (*it_max).location.x + maximumDistance;
		int y2 = (*it_max).location.y + maximumDistance;
		if (x2 >= modified.cols) x2 = modified.cols;
		if (y2 >= modified.rows) y2 = modified.rows;
		Point pt2(x2, y2);
		rectangle(modified, pt1, pt2, Scalar::all(65536), 1, 8, 0);
	};

	return modified;
}

//отрисовка прямолинейных границ на исходном изображении
Mat Radon::showFinalModel(Mat modelCurrent, int numberOfAngles) {
	for (std::vector<localMaximum>::iterator it_max = maximums.begin(); it_max != maximums.end(); it_max++) {
		int alpha = (*it_max).location.x;
		int s = (*it_max).location.y;

		std::cout << "alpha = " << alpha << "\n s = " << s << "   \n\n";

		//Scalar scalar = Scalar::all(255);
		Scalar scalar = Scalar::all(0);

		float alpha_rad = 2 * alpha * M_PI / numberOfAngles;
		int xCurrent = s*cos(alpha_rad);
		int yCurrent = s*sin(alpha_rad);
		int x = toX(xCurrent);
		int y = toY(yCurrent);

		float betta = alpha_rad + M_PI / 2;
		float gamma = alpha_rad + 3 * M_PI / 2;

		int r = sqrt(pow(modelCurrent.cols, 2) + pow(modelCurrent.rows, 2));

		int x2Current = xCurrent + r*cos(betta);
		int y2Current = yCurrent + r*sin(betta);
		int x2 = toX(x2Current);
		int y2 = toY(y2Current);
		line(modelCurrent, Point(x, y), Point(x2, y2), scalar, 2);

		int x4Current = xCurrent + r*cos(gamma);
		int y4Current = yCurrent + r*sin(gamma);
		int x4 = toX(x4Current);
		int y4 = toY(y4Current);
		line(modelCurrent, Point(x, y), Point(x4, y4), scalar, 2);
	};

	return modelCurrent;
}

//возврат изобажения
Mat Radon::getImage() {
	return image;
}

Mat Radon::getModifiedImage() {
	return modifImage;
}

//возврат модели
Mat Radon::getModel() {
	return model;
}
