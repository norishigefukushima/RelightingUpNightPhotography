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
	void setColorSpace(const int r, const int g, const int b);
	void setColorSpace(const int r, const int g, const int b, const float lab_l, const float lab_ab);
	void setLastIntensity(const float k_param, const float l_param);
	void setVanishingPointForDirectionalDiffusion(const int x, const int y);

	void run();
	void show(std::string wname, const int showSwitch);
	cv::Mat getOutputImage();
private:
	cv::Mat input8U;
	cv::Mat gray32;
	cv::Mat reflectance;
	cv::Mat lightSource;
	cv::Mat output;
	cv::Mat lightDiffused;
	cv::Mat base;
	std::vector<cv::Point> using_points;
	cv::Point vanishingPoint;

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
	std::vector<cv::Point> deletePoints(std::vector<cv::Point> points, const cv::Point vanishingPoint, const float range);
	std::vector<cv::Point> getPointsInRange(std::vector<cv::Point> points, std::vector<cv::Point> selected_points, float range, int num);

	void gaussianBlurFromPoint(const std::vector<cv::Point>& using_points, cv::Mat& dst);
	void convertPointsToImage(const std::vector<cv::Point>& src_point, cv::Mat& dst);
	void diffusing(const cv::Mat& lightSource, const std::vector<cv::Point>& src_point, cv::Mat& dst);
	void multiplyLab(const cv::Mat& src, cv::Mat& dst);
	void multiplyRGB(const cv::Mat& src, const cv::Mat& lightDiffused, cv::Mat& dst);//Eq. (14)
};
