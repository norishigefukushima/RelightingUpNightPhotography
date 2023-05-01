#pragma once
#include <opencv2/core.hpp>

enum DITHER_METHOD
{
	OSTROMOUKHOW,
	FLOYD_STEINBERG,
	SIERRA2,
	SIERRA3,
	JARVIS,
	STUCKI,
	BURKES,
	STEAVENSON,
	RANDOM_DIFFUSION,

	DITHERING_NUMBER_OF_METHODS,
};

enum DITHER_SCANORDER
{
	FORWARD,
	MEANDERING,
	IN2OUT, //for kernel sampling
	OUT2IN, //for kernel sampling
	FOURDIRECTION, //for kernel sampling
	FOURDIRECTIONIN2OUT,

	DITHERING_NUMBER_OF_ORDER,
};

std::string getDitheringOrderName(const int method);
std::string getDitheringMethodName(const int method);
int ditherDestruction(cv::Mat& src, cv::Mat& dest, const int dithering_method, int process_order = OUT2IN);

int ditheringFloydSteinberg(cv::Mat& remap, cv::Mat& dest, int process_order);
int ditheringFloydSteinbergPoints(cv::Mat& remap, cv::Mat& dest, int process_order, std::vector<cv::Point>& points);
int ditheringOstromoukhovCircle(cv::Mat& remap, cv::Mat& dest, const bool isMeandering);//not good

//visualize dithering order for debug
void ditheringOrderViz(cv::Mat& src, int process_order);