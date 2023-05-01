#include <opencv2/core.hpp>
#include <opencp.hpp>
#include "Relighting.hpp"

#if _DEBUG
#pragma comment(lib,"opencpd.lib")
#pragma comment(lib,"opencv_saliency455d.lib")
#else
#pragma comment(lib,"opencp.lib")
#pragma comment(lib,"opencv_saliency455.lib")
#endif

using namespace cv;
using namespace std;


void onMouse(int event, int x, int y, int flags, void* param)
{
	Point* ret = (Point*)param;
	if (flags == EVENT_FLAG_LBUTTON)
	{
		ret->x = x;
		ret->y = y;
	}
}

void scaleTowidth(Mat& src, Mat& dst, int width)
{
	int w = src.cols;
	int h = src.rows;
	int d_h = round(h * (double(width) / w));
	//cout << w << " " << h << " " << width << " " << d_h << endl;
	resize(src, dst, Size(width, d_h));
}

int main()
{
	vector<Mat> src;
	{
		for (int i = 0; i < 8; i++)
		{
			Mat tmp = imread("img/image" + to_string(i) + ".png");
			scaleTowidth(tmp, tmp, 512);
			src.push_back(tmp(Rect(0, 0, 512, (tmp.rows / 8) * 8)));
		}
	}
	int topleft = 100;
	string output_wname = "output";
	namedWindow(output_wname);
	string parameter_wname = "parameter";
	namedWindow(parameter_wname);
	cp::ConsoleImage ci(Size(800, 400), parameter_wname);
	moveWindow(parameter_wname, 0, 0);
	moveWindow(output_wname, 800, 0);
	int isExtra = 1; createTrackbar("extra", parameter_wname, &isExtra, 1);
	int src_num = 0; createTrackbar("source index", parameter_wname, &src_num, src.size() - 1);
	int intensity_ = 50; createTrackbar("light intensity", parameter_wname, &intensity_, 100);
	int l_ = 100; createTrackbar("l", parameter_wname, &l_, 1000);
	int lightsource_ = 3; createTrackbar("the number of light sources (+1)", parameter_wname, &lightsource_, 100);
	int minfilter_kernel_size = 3; createTrackbar("reflectance min filter size", parameter_wname, &minfilter_kernel_size, 3);
	int iteration = 3; createTrackbar("reflectance post filter iteration", parameter_wname, &iteration , 5);
	int rgborlab = 1; createTrackbar("RGB: 0, Lab: 1", parameter_wname, &rgborlab, 1);
	int lab_l_ = 100; createTrackbar("Lab: L / 100", parameter_wname, &lab_l_, 100);
	int lab_ab_ = 100; createTrackbar("Lab: ab / 100", parameter_wname, &lab_ab_, 100);
	int red = 0; createTrackbar("red", parameter_wname, &red, 255);
	int green = 255; createTrackbar("green", parameter_wname, &green, 255);
	int blue = 0; createTrackbar("blue", parameter_wname, &blue, 255);
	int diffusion_method = 2; createTrackbar("Diffusion method", parameter_wname, &diffusion_method, 4);
	int diffusion_ss_ = 100; createTrackbar("Diffusion sigma space /10", parameter_wname, &diffusion_ss_, 10000);
	int diffusion_sr_ = 200; createTrackbar("Diffusion sigma range / 10", parameter_wname, &diffusion_sr_, 1000);
	int directrix = 5; createTrackbar("directrix", parameter_wname, &directrix, 1000);
	int ssr_ss_ = 300; createTrackbar("SSR sigma space", parameter_wname, &ssr_ss_, 1000);
	int is_jbf = 1; createTrackbar("Do you use jbf on diffusion?", parameter_wname, &is_jbf, 1);
	int key = 0;

	Point pt_mouse = Point(src[src_num].size().width - 1, src[src_num].size().height - 1);
	cv::setMouseCallback("output", (MouseCallback)onMouse, (void*)&pt_mouse);
	int pitch = (int)(180 * (double)pt_mouse.y / (double)(src[src_num].size().height - 1) + 0.5);
	int yaw = (int)(180 * (double)pt_mouse.x / (double)(src[src_num].size().width - 1) + 0.5);

	cp::UpdateCheck uc(src_num);
	cp::UpdateCheck uc1(lightsource_, minfilter_kernel_size, iteration, diffusion_method);
	cp::Timer timer("timer", cp::TIME_MSEC);
	srand((int)time(0));
	Relighting relighting = Relighting(src[src_num]);
	uc.setIsFourceRetTrueFirstTime(false);
	string dm = "";
	int lightsource = -1;
	while (key != 'q')
	{
#pragma region updateCheck
		if (uc.isUpdate(src_num))
		{
			relighting = Relighting(src[src_num]);
		}
		if (uc1.isUpdate(lightsource_, minfilter_kernel_size, iteration, diffusion_method) || key == 'c')
		{
			timer.clearStat();
			lightsource = lightsource_ + 1;
		}
#pragma endregion

#pragma region setter
		const float k_param = (float)intensity_ / 100.f;
		const float l_param = (float)l_ / 100.f;

		const float ssr_ss = (float)ssr_ss_ / 10.f;
		relighting.setSSRSigma(ssr_ss);
		relighting.setReflectanceMinFilter(minfilter_kernel_size);
		relighting.setReflectancePostFilterIteration(iteration);

		relighting.setIntensity(k_param, l_param);
		relighting.setLightSourceNum(lightsource);
		
		float lab_l, lab_ab;
		if (rgborlab == 0)
		{
			relighting.setColorSpace(red, green, blue);
		}
		else
		{
			float lab_l = (float)lab_l_ / 100.f;
			float lab_ab = (float)lab_ab_ / 100.f;
			relighting.setColorSpace(red, green, blue, lab_l, lab_ab);
		}
		const float diffusion_ss = (float)diffusion_ss_ / 10.f;
		const float diffusion_sr = (float)diffusion_sr_ / 10.f;
		switch (diffusion_method)
		{
		case 0:
		default:
			relighting.setDiffusion(Diffusion::GAUSS, diffusion_ss);
			dm = "Gaussian blur";
			break;
		case 1:
			relighting.setDiffusion(Diffusion::GAUSS_D, diffusion_ss, diffusion_sr, directrix, is_jbf);
			dm = "Gaussian blur with directivity";
			break;
		case 2:
			relighting.setDiffusion(Diffusion::JBF, diffusion_ss, diffusion_sr);
			dm = "Joint Bilateral Filtering";
			break;
		case 3:
			relighting.setDiffusion(Diffusion::AMF, diffusion_ss, diffusion_sr);
			dm = "Adaptive Manifolds Filtering";
			break;
		case 4:
			relighting.setDiffusion(Diffusion::DTF, diffusion_ss, diffusion_sr);
			dm = "Domain Transform Filtering";
			break;
		}
		
		relighting.setCenterOfLights(pt_mouse.x, pt_mouse.y);
#pragma endregion

#pragma region body
		timer.start();
		relighting.filtering();
		timer.getpushLapTime();
#pragma endregion

#pragma region output
		relighting.show("output", isExtra == 1);
		key = waitKey(1);

		ci("src: %d", src_num);
		ci("the number of light sources: %d", lightsource);
		ci("SSR:sigma_space: %3.3f", ssr_ss);
		ci("SSR:min_kernel_size: %d",  minfilter_kernel_size);
		ci("Diffusion method:\n %s", dm.c_str());

		if (diffusion_method == Diffusion::GAUSS)
		{
			ci("Diffusion: sigma_space: %3.3f", diffusion_ss);
		}
		else if (diffusion_method == Diffusion::GAUSS_D)
		{
			ci("Diffusion: sigma_space: %3.3f, directrix: %d, isJBF: %3.3d", diffusion_ss, directrix, is_jbf);
		}
		else
		{
			ci("Diffusion: sigma_space: %3.3f, sigma_range: %3.3f", diffusion_ss, diffusion_sr);
		}

		if (rgborlab == 0)
		{
			ci("RGB color space: (R,G,B) = (%d, %d, %d)", red, green, blue);
		}
		else
		{
			ci("Lab color space: (L, ab) = (%1.2f %1.2f)", lab_l, lab_ab);
		}

		ci("(%1.3f + Light * %1.3f) * src", l_param, k_param);
		ci("trial: %d", timer.getStatSize());
		ci("Time (Mean)  : %3.3f ms, (Median): %3.3f ms", timer.getLapTimeMean(), timer.getLapTimeMedian());
		ci.show();
#pragma endregion
	}
}
