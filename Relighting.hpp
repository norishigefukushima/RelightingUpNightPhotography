#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "dithering.hpp"
void createImportanceMap_TexturenessPoints(cv::Mat& src, cv::Mat& dest, int& sample_num, float sampling_ratio, int ditheringMethod, std::vector<cv::Point>& points, float bin_ratio = 0.1f);
enum Diffusion {
	GAUSS_D,
	JBF,
	GAUSS,
	AMF,
	DTF
};

class Relighting
{
public:
	Relighting(const cv::Mat& src, const int lightsource_num = 1);
	~Relighting();

	void setLightSourceNum(const int num);
	void setSSRSigma(const float sigma);
	void setReflectanceMinFilter(const int minfilter_kernel_size);
	void setReflectancePostFilterIteration(const int iteration);

	void setDiffusion(Diffusion d, const float ss = 100, const float sr = 20, const float directrix = 5, const bool is_jbf = false);
	void setColorSpace(int r, int g, int b);
	void setColorSpace(int r, int g, int b, float lab_l, float lab_ab);
	void setIntensity(double k_param, double l_param);
	void setCenterOfLights(int x, int y);

	void filtering();
	void show(std::string wname, const bool isShowExtra);

private:
	cv::Mat input;
	cv::Mat gray32;
	cv::Mat reflectance;
	cv::Mat lightsource;
	cv::Mat output;
	cv::Mat light;
	cv::Mat base;
	std::vector<cv::Point> using_points;
	cv::Point center;

	float ssr_sigma = 90.f;
	int reflectancePostFilterIteration = 3;
	int minfilter_kernel_size;

	Diffusion diffusion = Diffusion::GAUSS;
	float diffuse_ss = 10.f;
	float diffuse_sr = 20.f;
	float directrix = 5.f;

	float k_param = 0.5f;
	float l_param = 0.5f;
	float lab_l = 1.f;
	float lab_ab = 1.f;

	int dsr = 600;
	int lightsource_num = 0;
	int red = 100;
	int green = 100;
	int blue = 100;

	bool is_jbf = false;
	bool is_lab = true;

	void SSR(cv::Mat& gray32, cv::Mat& reflectance);

	void blueNoiseSampling(const cv::Mat& reflectance, std::vector<cv::Point>& using_point);
	std::vector<cv::Point> getDitheringPoints(std::vector<cv::Point> points, cv::Point first_point, float min, const float gamma);
	std::vector<cv::Point> deletePoints(std::vector<cv::Point> points, const cv::Point center, const const float range);
	std::vector<cv::Point> getPointsInRange(std::vector<cv::Point> points, std::vector<cv::Point> selected_points, float range, int num);

	void gaussianBlurFromPoint(cv::Mat& dst);
	void mappingLightSource(const std::vector<cv::Point>& src_point, cv::Mat& dst);
	void diffuse(cv::Mat& lightsource, cv::Mat& dst);
	void multiplyLab(const cv::Mat& src, cv::Mat& dst);
	void multiplyRGB(const cv::Mat& src, const cv::Mat& light, cv::Mat& dst);//Eq. (14)
};
