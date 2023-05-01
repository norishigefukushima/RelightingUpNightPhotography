#include <opencv2/opencv.hpp>
#include <opencp.hpp>
#include <spatialfilter/SpatialFilter.hpp>
#include "Relighting.hpp"

#if _DEBUG
#pragma comment(lib,"opencpd.lib")
#pragma comment(lib,"opencv_saliency455d.lib")
#else
#pragma comment(lib,"opencp.lib")
#pragma comment(lib,"opencv_saliency455.lib")
#pragma comment(lib, "SpatialFilter.lib")
#pragma comment(lib, "multiscalefilter.lib")
#endif
//#define TIMER_TEST

using namespace cv;
using namespace std;

void createImportanceMap_TexturenessPoints(cv::Mat& src, cv::Mat& dest, int& sample_num, float sampling_ratio, int ditheringMethod, vector<cv::Point>& points, float bin_ratio)
{
	//CV_Assert(src.depth() == CV_32F);
#if 0
	static int ss1 = 2; createTrackbar("ss1", "", &ss1, 10);
	static int ss2 = 0; createTrackbar("ss2", "", &ss2, 100);
	static int sr = 70; createTrackbar("sr", "", &sr, 255);
	static int type = 0; createTrackbar("type", "", &type, 2);
#else
	int ss1 = 2;
	int ss2 = 0;
	int sr = 70;
	int type = 0;
#endif
	if (dest.empty() || dest.type() != CV_8UC1 || dest.size() != src.size())dest.create(src.size(), CV_8UC1);
	//	CV_Assert(src.type() == dest.type());
	Mat src_32f;
	Mat v(src.rows, src.cols, CV_32F);//importance map (n pixels)
	{
		//Timer t("tt1");
		//src.convertTo(src_32f, CV_32F, 1.0 / 255);
		normalize(src, src_32f, 0, 1, NORM_MINMAX);

		if (type == 0)
		{
			Size ksize = Size(5, 5);
			GaussianBlur(src_32f, v, ksize, ss1);
			//gf::GaussianFilterSlidingDCT5_AVX_32F gauss(src_32f.size(), ss1, 1, false);
			//gauss.filtering(src_32f, v);
		}
		else
		{
			double sigma_range = sr / 255.0;
			//bilateralFilter(src_32f, filtered, 10, 30, 8);
			//bilateralFilterLocalStatisticsPrior(src_32f, v, sigma_range, ss1, sigma_range * 0.8);
		}


		//start = cv::getTickCount();
		//boxFilter(src_32f, filtered, CV_32F, Size(15, 15));
		//sincFilterFFT(src_32f, filtered);
		absdiff(src_32f, v, v);
		//Mat temp;
		//filtered.convertTo(temp, CV_8U, 255);
		//imshow("t", temp);
		//end = cv::getTickCount();
		//std::cout << "absdiff time:" << (end - start) * 1000 / (cv::getTickFrequency()) << std::endl;
		//start = cv::getTickCount();

		if (ss2 != 0)
		{
			Size ksize = Size(5, 5);
			GaussianBlur(v, v, ksize, ss2);
		}
		//end = cv::getTickCount();
		//std::cout << "Gaussian:" << (end - start) * 1000 / (cv::getTickFrequency()) << std::endl;

		//boxFilter(filtered, filtered, CV_32F, Size(5, 5));
		//sincFilterFFT(filtered, filtered);
	}
	//if (type == 2) randu(v, 0, 1);

	//cp::imshowNormalize("imp", filtered); waitKey();
	//remap‚Ì‚½‚ß‚ÌƒqƒXƒgƒOƒ‰ƒ€ŒvŽZ(Appendix)
	//start = cv::getTickCount();

	{
		//cp::Timer t("tt2");
		const int m = 500;//number of bim
		//int binNum = (int)(bin_ratio*dest.size().area()*sampling_ratio);
		int histSize[] = { m };

		float value_ranges[] = { 0.f,1.f };
		const float* ranges[] = { value_ranges };
		Mat hist;

		int channels[] = { 0 };
		int dims = 1;
		calcHist(&v, 1, channels, Mat(), hist, dims, histSize, ranges, true, false); //ƒqƒXƒgƒOƒ‰ƒ€ŒvŽZ
		//double maxVal = 0;
		//minMaxLoc(hist, 0, &maxVal, 0, 0);
		//int c = 0;
		//for (int i = 0; i < binNum; i++)
		//{
		//	float binVal = hist.at<float>(i);
		//	cout << i << "\t" << binVal << endl;
		//	c += binVal;
		//}
		//cout << "c:" << c << endl;
		//cout << "fn : " << dest.size().area()*sampling_ratio << endl;
		//getchar();


		int H_k = 0;//cumulative sum of histogram
		float X_k = 0.f;//sum of hi*xi

		float s = 0.f;//scaling factor
		float x = 0.f;//bin center
		const float inv_m = 1.f / m;//1/m
		const float offset = inv_m * 0.5f;
		const int n = src.rows * src.cols;
		const int nt = int(n * (1.f - sampling_ratio));
		const float sx_max = 1.f + FLT_EPSILON;
		const float sx_min = 1.f - FLT_EPSILON;
		//cout << n<<","<<nt<<"," <<sampling_ratio<< endl;
		for (int i = 0; i < m; i++)
		{
			const int h_i = saturate_cast<int>(hist.at<float>(i));
			H_k += h_i;

			x = i * inv_m + offset;
			X_k += x * h_i;

			s = (H_k - nt) / X_k;//eq (5)
			float sx = s * x;
			if (sx_min < sx /*&& sx < sx_max*/)
			{
				break;
			}

		}

		const __m256 ms = _mm256_set1_ps(s);
		const __m256 ones = _mm256_set1_ps(1.f);
		//#pragma omp parallel for schedule (dynamic)
		const int n_simd = n / 8;
		float* v_ptr = v.ptr<float>();
		//result[i] = min(v[i] * s, 1.f);
		for (int i = 0; i < n_simd; i++)
		{
			_mm256_store_ps(v_ptr, _mm256_min_ps(_mm256_mul_ps(_mm256_load_ps(v_ptr), ms), ones));
			v_ptr += 8;
		}
	}
	//end = cv::getTickCount();
	//std::cout << "Calchist:" << (end - start) * 1000 / (cv::getTickFrequency()) << std::endl;
	//start = cv::getTickCount();


	{
		//sample_num = ditherDestruction(v, dest, ditheringMethod, MEANDERING);
		sample_num = ditheringFloydSteinbergPoints(v, dest, MEANDERING, points);
		//imshow("dither", filtered);

		//srand(cv::getTickCount());
		//int n = rand() % 4; if (n != 3) rotate(dest, dest, n);
	}
	//end = cv::getTickCount();
	//std::cout << "Dithering time:" << (end - start) * 1000 / (cv::getTickFrequency()) << std::endl;

	//{
	//	int ns = sampling_ratio * (src.cols*src.rows);
	//	RNG rng;
	//	for (; sample_num <= ns;)
	//	{
	//		int x = rng.uniform(0, src.cols);
	//		int y = rng.uniform(0, src.rows);
	//		if (dest.at<uchar>(y, x) == 0)
	//		{
	//			dest.at<uchar>(y, x) = 255;
	//			sample_num++;
	//		}
	//	}
	//}
	//cout << (sample_num * 100.f / (src.rows*src.cols)) << endl;
}

#pragma region setter
void Relighting::setLightSourceNum(const int num)
{
	this->lightsource_num = num;
}

void Relighting::setSSRSigma(const float sigma)
{
	this->ssr_sigma = sigma;
}

void Relighting::setReflectanceMinFilter(const int minfilter_kernel_size)
{
	this->minfilter_kernel_size = minfilter_kernel_size;
}

void Relighting::setReflectancePostFilterIteration(const int iteration)
{
	this->reflectancePostFilterIteration = iteration;
}

void Relighting::setDiffusion(Diffusion d, const float ss, const float sr, const float directrix, const bool is_jbf_directional_difusion)
{
	this->diffusion = d;
	this->diffuse_ss = ss;
	this->diffuse_sr = sr;
	this->directrix = directrix;
	this->is_jbf = is_jbf_directional_difusion;
}

void Relighting::setColorSpace(const int r, const int g, const int b)
{
	this->is_lab = false;
	this->red = r;
	this->green = g;
	this->blue = b;
}

void Relighting::setColorSpace(const int r, const int g, const int b, const float lab_l, const float lab_ab)
{
	this->is_lab = true;
	this->red = r;
	this->green = g;
	this->blue = b;
	this->lab_l = lab_l;
	this->lab_ab = lab_ab;
}

void Relighting::setLastIntensity(const float k_param, const float l_param)
{
	this->k_param = k_param;
	this->l_param = l_param;
}

void Relighting::setVanishingPointForDirectionalDiffusion(const int x, const int y)
{
	this->vanishingPoint = Point(x, y);
}
#pragma endregion

Relighting::Relighting(const Mat& src, const int lightsource_num)
{
	this->input8U = src;
	Mat input32; input8U.convertTo(input32, CV_32F);
	cvtColor(input32, this->gray32, COLOR_BGR2GRAY);
	this->output = input8U.clone();
	this->vanishingPoint = Point(this->input8U.cols / 2, 0);
	this->lightsource_num = lightsource_num;
	//unsigned int now = (unsigned int)time(0);
	//srand(now);
}

Relighting::~Relighting()
{
}

void Relighting::SSR(Mat& gray32, Mat& reflectance)
{
	const int order = 3;
	cp::SpatialFilterSlidingDCT5_AVX_32F gfilter(gray32.size(), ssr_sigma, order);
	if constexpr (false)//log
	{
		Mat ones = FLT_MIN * Mat::ones(gray32.size(), CV_32F);
		gfilter.filter(gray32, reflectance, ssr_sigma, order);
		Mat loggray; log(gray32 + FLT_EPSILON, loggray);
		Mat loggauss; log(reflectance + FLT_EPSILON, loggauss);
		this->reflectance = loggray - loggauss;
	}
	else
	{
		gfilter.filter(gray32, reflectance, ssr_sigma, order);
		reflectance = gray32 / (reflectance + FLT_EPSILON);
	}
}

void Relighting::blueNoiseSampling(const Mat& reflectance, vector<Point>& using_point)
{
	RNG rng(cv::getTickCount());
	const float sampling_ratio = 1.f / 400;
	const int ditheringMethod = FLOYD_STEINBERG;
	std::vector<Point> points;
	int tmp;
	Mat dithering_saliency_map;
	createImportanceMap_TexturenessPoints((Mat)reflectance, dithering_saliency_map, tmp, sampling_ratio, ditheringMethod, points);
	if (points.size() < 1) exit(0);

	Point first_point = points[rng.uniform(0, (int)points.size())];
	int min_r = (int)powf(float(input8U.size().area() / lightsource_num), 0.5);
	using_points = getDitheringPoints(points, first_point, float(min_r), 2.f);
}

vector<Point> Relighting::getDitheringPoints(vector<Point> points, Point first_point, float min, const float gamma) {
	vector<Point> dst(lightsource_num);
	vector<Point> can_select_points = points;
	vector<Point> temp_points, temp_points2;
	vector<Point> next_points = points;

	int temp;
	int sample_num;
	int count;
	dst[0] = first_point;
	while (true)
	{
		float max = min * gamma;
		can_select_points = points; // B
		temp_points = deletePoints(can_select_points, dst[0], min); // A: dest
		next_points = getPointsInRange(temp_points, dst, max, 1);
		can_select_points = temp_points; // B
		count = 1;
		for (int i = 1; i < lightsource_num; i++)
		{
			if (next_points.size() != 0)
			{
				sample_num = (int)next_points.size();
				int max_num = -1;
				temp = 0;
				for (int j = 0; j < sample_num; j++)
				{
					temp_points2 = deletePoints(can_select_points, next_points[j], min);
					if (max_num < temp_points2.size()) // 9, temp_points2: 
					{
						max_num = (int)temp_points2.size();
						temp = j;
					}
				}
				dst[i] = next_points[temp];
			}
			else if (can_select_points.size() != 0) // 7
			{
				sample_num = (int)can_select_points.size();
				temp = rand() % sample_num;
				dst[i] = can_select_points[temp];
			}
			else
			{
				break;
			}
			temp_points = deletePoints(can_select_points, dst[i], min);
			next_points = getPointsInRange(temp_points, dst, max, i + 1);
			can_select_points = temp_points;
			count = i + 1;
		}
		if (count == lightsource_num)
		{
			return dst;
		}
		min -= 4;
		if (min <= 0)
		{
			cerr << "cannot search " << count << " light sources" << endl;
			exit(0);
			return dst;
		}
	}
}

vector<Point> Relighting::deletePoints(vector<Point> points, const Point point, const float range)
{
	vector<Point> dst = points;
	int num = 0;
	const float R = range * range;
	for (int i = 0; i < points.size(); i++)
	{
		float dist = (float)(points[i].x - point.x) * (points[i].x - point.x) + (points[i].y - point.y) * (points[i].y - point.y);
		if (dist < R)
		{
			dst.erase(dst.begin() + num--);
		}
		num++;
	}
	return dst;
}

vector<Point> Relighting::getPointsInRange(vector<Point> points, vector<Point> selected_points, float range, int num)
{
	vector<Point> dst = points;
	vector<vector<Point>> vector_dst(selected_points.size());
	float R = range * range;
	for (int n = 0; n < num; n++)
	{
		for (int i = 0; i < points.size(); i++)
		{
			float d = float((points[i].x - selected_points[n].x) * (points[i].x - selected_points[n].x) + (points[i].y - selected_points[n].y) * (points[i].y - selected_points[n].y));
			if (d < R)
			{
				vector_dst[n].push_back(points[i]);
			}
		}
	}
	dst.resize(vector_dst[0].size());
	dst = vector_dst[0];
	for (int i = 1; i < selected_points.size(); i++)
	{
		dst.reserve(dst.size() + vector_dst[i].size());
		copy(vector_dst[i].begin(), vector_dst[i].end(), back_inserter(dst));
		sort(dst.begin(), dst.end(), [](Point a, Point b) {
			if (a.x != b.x)
			{
				return a.x < b.x;
			}
			else {
				return a.y < b.y;
			}
			});
		dst.erase(std::unique(dst.begin(), dst.end()), dst.end());
	}
	return dst;
}

void Relighting::convertPointsToImage(const std::vector<Point>& src_point, Mat& dest32FC3)
{
	dest32FC3.create(input8U.size(), CV_32FC3);
	dest32FC3.setTo(0);
	for (int i = 0; i < src_point.size(); i++)
	{
		dest32FC3.at<Vec3f>(src_point[i]) = Vec3f((float)this->blue, (float)this->green, (float)this->red);
	}
}

void Relighting::diffusing(const cv::Mat& lightSource32FC3, const std::vector<Point>& sourcePoints, Mat& dest32FC3)
{
	const int r = (int)ceil(3.f * diffuse_ss);

	if (diffusion == Diffusion::GAUSS)
	{
		vector<Mat> split_l; split(lightSource32FC3, split_l);
		int order = 3;
#pragma omp parallel for
		for (int i = 0; i < split_l.size(); i++)
		{
			//GaussianBlur(split_l[i], split_l[i], Size(0, 0), diffuse_ss);
			cp::SpatialFilterSlidingDCT5_AVX_32F gfilter(input8U.size(), diffuse_ss, order);
			gfilter.filter(split_l[i], split_l[i], diffuse_ss, order);
		}
		merge(split_l, dest32FC3);
	}
	else if (diffusion == Diffusion::GAUSS_D)
	{
		gaussianBlurFromPoint(sourcePoints, dest32FC3);
	}
	else if (diffusion == Diffusion::JBF)
	{
		cp::jointBilateralFilter(lightSource32FC3, gray32, dest32FC3, 2 * r + 1, diffuse_sr, diffuse_ss);
	}
	else if (diffusion == Diffusion::AMF)
	{
		Mat lightsource_; GaussianBlur(lightSource32FC3, lightsource_, Size(0, 0), 5);
		Ptr<ximgproc::AdaptiveManifoldFilter> amf = ximgproc::AdaptiveManifoldFilter::create();
		//amf->setSigmaS((float)diffuse_ss / 100);
		//amf->setSigmaR((float)diffuse_sr / 100);
		amf->filter(lightsource_, dest32FC3, gray32 / 255);
	}
	else if (diffusion == Diffusion::DTF)
	{
		Mat lightsource_;
		GaussianBlur(lightSource32FC3, lightsource_, Size(0, 0), 10);
		cp::domainTransformFilter(lightsource_, gray32, dest32FC3, diffuse_sr, diffuse_ss, 1, cp::DTF_L1, cp::DTF_NC);
	}
	else
	{
		exit(EXIT_FAILURE);
	}

	//normalization
	Vec3f p = dest32FC3.at<Vec3f>(sourcePoints[0]);
	vector<Mat> split_l; split(dest32FC3, split_l);
	if (p[0] != 0.f)multiply(split_l[0], blue / p[0] / 255.f, split_l[0]);
	if (p[1] != 0.f)multiply(split_l[1], green / p[1] / 255.f, split_l[1]);
	if (p[2] != 0.f)multiply(split_l[2], red / p[2] / 255.f, split_l[2]);
	merge(split_l, dest32FC3);
}

void Relighting::gaussianBlurFromPoint(const vector<Point>& using_points, Mat& dst)
{
	Mat input32; input8U.convertTo(input32, CV_32F);
	vector<Mat> split_dst(using_points.size());
	int x = vanishingPoint.x;
	int y = vanishingPoint.y;
#pragma omp parallel for
	for (int p = 0; p < using_points.size(); p++)
	{
		split_dst[p] = Mat::zeros(input8U.size(), CV_32FC3);
		Point fp = using_points[p];
		float ip = float(y - fp.y);
		float denom = powf((float)((y - fp.y) * (y - fp.y) + (x - fp.x) * (x - fp.x)), 0.5);
		float cosine = ip / denom;
		float angle = acosf(cosine);
		if (x < fp.x)
		{
			angle *= -1;
		}

		const Vec3f sfp = input32.ptr<Vec3f>(fp.y)[fp.x];
		const Vec3f color = Vec3f(float(blue), float(green), float(red));
		Vec3f denom_vec(0, 0, 0);
		for (int j = 0; j < input32.rows; j++)
		{
			Vec3f* sp = input32.ptr<Vec3f>(j);
			Vec3f* dp = split_dst[p].ptr<Vec3f>(j);
			for (int i = 0; i < input32.cols; i++)
			{
				float px = float(i - fp.x);
				float py = float(j - fp.y);
				float i2 = px * cos(angle) - py * sin(angle);
				float j2 = px * sin(angle) + py * cos(angle) + directrix;
				float fx = (i2 * i2) / (4.f * directrix);
				if (j2 >= fx)
				{
					Vec3f rk(1, 1, 1);
					if (is_jbf)
					{
						rk = (sp[i] - sfp) * (sp[i] - sfp) / (-2 * diffuse_sr * diffuse_sr);
						exp(rk, rk);
					}
					const float sk = exp((px * px + py * py) / (-2 * diffuse_ss * diffuse_ss));
					Vec3f tmp; multiply(sk, rk, tmp);
					add(denom_vec, tmp, denom_vec);
				}
			}
		}
		for (int j = 0; j < input32.rows; j++)
		{
			Vec3f* sp = input32.ptr<Vec3f>(j);
			Vec3f* dp = split_dst[p].ptr<Vec3f>(j);
			for (int i = 0; i < input32.cols; i++)
			{
				float px = float(i - fp.x);
				float py = float(j - fp.y);
				float i2 = px * cos(angle) - py * sin(angle);
				float j2 = px * sin(angle) + py * cos(angle) + directrix;
				float fx = (i2 * i2) / (4 * directrix);
				if (j2 >= fx)
				{
					Vec3f rk(1, 1, 1);
					if (is_jbf)
					{
						rk = (sp[i] - sfp) * (sp[i] - sfp) / (-2 * diffuse_sr * diffuse_sr);
						exp(rk, rk);
					}
					float sk = exp((px * px + py * py) / (-2 * diffuse_ss * diffuse_ss));
					multiply(sk * color, rk, dp[i]);
					divide(dp[i], denom_vec, dp[i]);
				}
			}
		}
	}
	dst = Mat::zeros(input32.size(), input32.type());
	for (size_t i = 0; i < split_dst.size(); i++)
	{
		dst += split_dst[i];
	}
}

void Relighting::multiplyLab(const Mat& src8U, Mat& dst8U)
{
	cvtColor(src8U, dst8U, COLOR_BGR2Lab);
	const int size = src8U.size().area();
	uchar* s = dst8U.ptr<uchar>();
	for (int i = 0; i < size; i++)
	{
		s[3 * i + 0] = saturate_cast<uchar>(s[3 * i + 0] * lab_l);
		s[3 * i + 1] = saturate_cast<uchar>((s[3 * i + 1] - 128.f) * lab_ab + 128.f);
		s[3 * i + 2] = saturate_cast<uchar>((s[3 * i + 2] - 128.f) * lab_ab + 128.f);
	}
	cvtColor(dst8U, dst8U, COLOR_Lab2BGR);
}

void Relighting::multiplyRGB(const Mat& src, const Mat& lightDiffused, Mat& dst)
{
	const uchar* s = src.ptr<uchar>();
	const float* lmap = lightDiffused.ptr<float>();
	uchar* d = dst.ptr<uchar>();
	const int size = src.size().area() * src.channels();
	for (int i = 0; i < size; i++)
	{
		d[i] = saturate_cast<uchar>((l_param + k_param * lmap[i]) * s[i]);
	}
}

void Relighting::run()
{
	{
#ifdef TIMER_TEST
		cp::Timer t("SSR");
#endif
		SSR(gray32, reflectance);
	}

	{
#ifdef TIMER_TEST
		cp::Timer t("min filter");
#endif
		cp::minFilter(reflectance, reflectance, minfilter_kernel_size);
	}
	{
#ifdef TIMER_TEST
		cp::Timer t("post filter");
#endif
		for (int i = 0; i < reflectancePostFilterIteration; i++)
		{
			cp::domainTransformFilter(reflectance, gray32, reflectance, 30, 10, 2, cp::DTF_L1, cp::DTF_NC);
			//cp::jointBilateralFilter(reflectance, gray32, reflectance, 61, 15, 10);
			//cp::guidedImageFilter(reflectance, gray32, reflectance, 15, 30);
		}
	}

	{
#ifdef TIMER_TEST
		cp::Timer t("bluenoise");
#endif
		blueNoiseSampling(reflectance, using_points);
	}
	{
#ifdef TIMER_TEST
		cp::Timer t("light souce");
#endif
		convertPointsToImage(using_points, lightSource);//Eq. (7)
	}
	{
#ifdef TIMER_TEST
		cp::Timer t("diffuse");
#endif
		diffusing(lightSource, using_points, lightDiffused);
	}

	//Eq. (14) or (15)
	{
#ifdef TIMER_TEST
		cp::Timer t("multiply");
#endif
		if (is_lab)
		{
			multiplyLab(input8U, base);
			multiplyRGB(base, lightDiffused, output);
		}
		else
		{
			multiplyRGB(input8U, lightDiffused, output);
		}
	}
}

void Relighting::show(string wname, const int showSwitch)
{
	if (showSwitch == 1)
	{
		Mat outC3, out8u, show;
		cvtColor(reflectance, outC3, COLOR_GRAY2BGR);
		outC3.convertTo(out8u, CV_8U, 255);
		cv::vconcat(output, out8u, show);
		lightDiffused.convertTo(out8u, CV_8U, 255);
		if (diffusion == Diffusion::GAUSS_D) cp::drawGrid(out8u, vanishingPoint, COLOR_RED);
		cv::vconcat(show, out8u, show);
		imshow(wname, show);
	}
	else if (showSwitch == 2)
	{
		Mat outC3, out8u, show;
		cvtColor(reflectance, outC3, COLOR_GRAY2BGR);
		outC3.convertTo(out8u, CV_8U, 255);
		if (diffusion == Diffusion::GAUSS_D) cp::drawGrid(out8u, vanishingPoint, COLOR_RED);
		cv::hconcat(output, out8u, show);
		lightDiffused.convertTo(out8u, CV_8U, 255);
		cv::hconcat(show, out8u, show);
		imshow(wname, show);
	}
	else
	{
		imshow(wname, output);
	}
}