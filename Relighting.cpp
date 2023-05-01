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
		const int nt = n * (1.f - sampling_ratio);
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

void Relighting::setDiffusion(Diffusion d, const float ss, const float sr, const float directrix, const bool is_jbf)
{
	this->diffusion = d;
	this->diffuse_ss = ss;
	this->diffuse_sr = sr;
	this->directrix = directrix;
	this->is_jbf = is_jbf;
}

void Relighting::setColorSpace(int r, int g, int b)
{
	this->is_lab = false;
	this->red = r;
	this->green = g;
	this->blue = b;
}

void Relighting::setColorSpace(int r, int g, int b, float lab_l, float lab_ab)
{
	this->is_lab = true;
	this->red = r;
	this->green = g;
	this->blue = b;
	this->lab_l = lab_l;
	this->lab_ab = lab_ab;
}

void Relighting::setIntensity(double k_param, double l_param)
{
	this->k_param = k_param;
	this->l_param = l_param;
}

void Relighting::setCenterOfLights(int x, int y)
{
	this->center = { x, y };
}
#pragma endregion

void Relighting::show(string wname, bool isShowExtra)
{
	if (isShowExtra)
	{
		Mat tmp, tmp2, show;
		cvtColor(reflectance, tmp, COLOR_GRAY2BGR);
		tmp.convertTo(tmp2, CV_8U, 255);
		cv::vconcat(output, tmp2, show);
		light.convertTo(tmp2, CV_8U, 255);
		cv::vconcat(show, tmp2, show);
		imshow(wname, show);
	}
	else
	{
		imshow(wname, output);
	}
}

Relighting::Relighting(const Mat& src, const int lightsource_num)
{
	this->input = src;
	Mat input32; input.convertTo(input32, CV_32F);
	cvtColor(input32, this->gray32, COLOR_BGR2GRAY);
	this->output = input.clone();
	this->center = Point(this->input.cols / 2, 0);
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

	Point first_point = points[rng.uniform(0, points.size())];
	int min_r = (int)powf(input.size().area() / lightsource_num, 0.5);
	using_points = getDitheringPoints(points, first_point, min_r, 2);
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
				sample_num = next_points.size();
				int max_num = -1;
				temp = 0;
				for (int j = 0; j < sample_num; j++)
				{
					temp_points2 = deletePoints(can_select_points, next_points[j], min);
					if (max_num < temp_points2.size()) // 9, temp_points2: 
					{
						max_num = temp_points2.size();
						temp = j;
					}
				}
				dst[i] = next_points[temp];
			}
			else if (can_select_points.size() != 0) // 7
			{
				sample_num = can_select_points.size();
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
			float d = (points[i].x - selected_points[n].x) * (points[i].x - selected_points[n].x) + (points[i].y - selected_points[n].y) * (points[i].y - selected_points[n].y);
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

void  Relighting::diffuse(Mat& lightsource, Mat& dst)
{
	const int r = ceil(3 * diffuse_ss);

	if (diffusion == Diffusion::GAUSS)
	{
		vector<Mat> split_l; split(lightsource, split_l);
		int order = 3;
#pragma omp parallel for
		for (int i = 0; i < split_l.size(); i++)
		{
			//GaussianBlur(split_l[i], split_l[i], Size(0, 0), diffuse_ss);
			cp::SpatialFilterSlidingDCT5_AVX_32F gfilter(input.size(), diffuse_ss, order);
			gfilter.filter(split_l[i], split_l[i], diffuse_ss, order);
		}

		merge(split_l, dst);
	}
	else if (diffusion == Diffusion::GAUSS_D)
	{
		gaussianBlurFromPoint(dst);
	}
	else if (diffusion == Diffusion::JBF)
	{
		cp::jointBilateralFilter(lightsource, gray32, dst, 2 * r + 1, diffuse_sr, diffuse_ss);
	}
	else if (diffusion == Diffusion::AMF)
	{
		Mat lightsource_; GaussianBlur(lightsource, lightsource_, Size(0, 0), 5);
		Ptr<ximgproc::AdaptiveManifoldFilter> amf = ximgproc::AdaptiveManifoldFilter::create();
		//amf->setSigmaS((float)diffuse_ss / 100);
		//amf->setSigmaR((float)diffuse_sr / 100);
		amf->filter(lightsource_, dst, gray32 / 255);
	}
	else if (diffusion == Diffusion::DTF)
	{
		Mat lightsource_; GaussianBlur(lightsource, lightsource_, Size(0, 0), 10);
		cp::domainTransformFilter(lightsource_, gray32, dst, diffuse_sr, diffuse_ss, 1, cp::DTF_L1, cp::DTF_NC);
	}
	else
	{
		exit(EXIT_FAILURE);
	}
}

void Relighting::mappingLightSource(const std::vector<Point>& src_point, Mat& dst)
{
	dst = Mat::zeros(input.size(), CV_32FC3);
	for (int i = 0; i < using_points.size(); i++)
	{
		dst.at<Vec3f>(src_point[i]) = Vec3f((float)this->blue, (float)this->green, (float)this->red);
	}
}

void Relighting::gaussianBlurFromPoint(Mat& dst)
{
	Mat input32; input.convertTo(input32, CV_32F);
	vector<Mat> split_dst(using_points.size());
	int x = center.x;
	int y = center.y;
#pragma omp parallel for
	for (int p = 0; p < using_points.size(); p++)
	{
		split_dst[p] = Mat::zeros(input.size(), CV_32FC3);
		Point fp = using_points[p];
		float ip = y - fp.y;
		float denom = powf((float)((y - fp.y) * (y - fp.y) + (x - fp.x) * (x - fp.x)), 0.5);
		float cosine = ip / denom;
		float angle = acosf(cosine);
		if (x < fp.x)
		{
			angle *= -1;
		}

		const Vec3f sfp = input32.ptr<Vec3f>(fp.y)[fp.x];
		const Vec3f color(blue, green, red);
		Vec3f denom_vec(0, 0, 0);
		for (int j = 0; j < input32.rows; j++)
		{
			Vec3f* sp = input32.ptr<Vec3f>(j);
			Vec3f* dp = split_dst[p].ptr<Vec3f>(j);
			for (int i = 0; i < input32.cols; i++)
			{
				float px = i - fp.x;
				float py = j - fp.y;
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
				float px = i - fp.x;
				float py = j - fp.y;
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

void Relighting::multiplyLab(const Mat& src, Mat& dst)
{
	cvtColor(src, dst, COLOR_BGR2Lab);
	const int size = src.size().area();
	uchar* s = dst.ptr<uchar>();
	for (int i = 0; i < size; i += 3)
	{
		s[3 * i + 0] = saturate_cast<uchar>(s[3 * i + 0] * lab_l);
		s[3 * i + 1] = saturate_cast<uchar>((s[3 * i + 1] - 128.f) * lab_ab + 128.f);
		s[3 * i + 2] = saturate_cast<uchar>((s[3 * i + 2] - 128.f) * lab_ab + 128.f);
	}
	cvtColor(dst, dst, COLOR_Lab2BGR);
}

void Relighting::multiplyRGB(const Mat& src, const Mat& light, Mat& dst)
{
	const uchar* s = src.ptr<uchar>();
	const float* lmap = light.ptr<float>();
	uchar* d = dst.ptr<uchar>();
	const int size = src.size().area() * src.channels();
	const float intens_32 = (float)k_param;
	for (int i = 0; i < size; i++)
	{
		d[i] = saturate_cast<uchar>((l_param + k_param * lmap[i]) * s[i]);
	}
}

void Relighting::filtering()
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
		mappingLightSource(using_points, lightsource);//Eq. (7)
	}
	{
#ifdef TIMER_TEST
		cp::Timer t("diffuse");
#endif
		diffuse(lightsource, light);
	}

	//Eq. (14) or (15)
	{
#ifdef TIMER_TEST
		cp::Timer t("multiply");
#endif
		if (is_lab)
		{
			multiplyLab(input, base);
			multiplyRGB(base, light, output);
		}
		else
		{
			multiplyRGB(input, light, output);
		}
	}
}