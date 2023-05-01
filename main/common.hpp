#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>

//ã´äEèàóù
//#define USE_BORDER_REPLICATE 1 //BORDER_REPLICATE   = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
#define USE_BORDER_REFLECT 1 //BORDER_REFLECT     = 2, //!< `fedcba|abcdefgh|hgfedcb`
//#define USE_BORDER_REFLECT_101 1 //BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`

		//extrapolation functions
		//(atE() and atS() require variables w and h in their scope respectively)
#ifdef USE_BORDER_REPLICATE 
#define LREF(n) (std::max(n,0))
#define RREF(n) (std::min(n,width-1))
#define UREF(n) (std::max(n,0) * width)
#define DREF(n) (std::min(n,height-1) * width)
#elif USE_BORDER_REFLECT 
//#define REFLECT(x,y) ((y < 0 ? abs(y + 1) : (height <= y ? 2*width - (y) - 1: y)) * width + (x < 0 ? abs(x + 1) : (width <= x ? 2*width - (x) - 1: x)))
#define LREF(n) (n < 0 ? - (n) - 1: n)
#define RREF(n) (n < width ? n: 2*width - (n) - 1)
#define UREF(n) ((n < 0 ? - (n) - 1: n) * width)
#define DREF(n) ((n < height ? n : 2*height - (n) - 1) * width)
//#define RREF(n) ((width - 1) - std::abs(int(int(std::abs(n+0.5f))%(2*width) - ((2*width - 1)/ 2.f))))
//#define LREF(n) ((width - 1) - std::abs(int(int(std::abs(n+0.5f))%(2*width) - ((2*width - 1)/ 2.f))))
//#define UREF(n) (((height - 1) - std::abs(int(int(std::abs(n+0.5f))%(2*height) - ((2*height - 1)/ 2.f))))*width)
//#define DREF(n) (((height - 1) - std::abs(int(int(std::abs(n+0.5f))%(2*height) - ((2*height - 1)/ 2.f))))*width)
#elif USE_BORDER_REFLECT_101
#define LREF(n) (std::abs(n))
#define RREF(n) (width-1-std::abs(width-1-(n)))
#define UREF(n) (std::abs(n) * width)
#define DREF(n) ((height-1-std::abs(height-1-(n))) * width)
#endif

////FFTW
//#pragma comment(lib, "libfftw3-3.lib")
//#pragma comment(lib, "libfftw3f-3.lib")
//#pragma comment(lib, "libfftw3l-3.lib")


//VYV
#define VYV_NUM_NEWTON_ITERATIONS       6
#define VYV_ORDER_MAX 5
#define VYV_ORDER_MIN 3
#define VYV_VALID_ORDER(K)  (VYV_ORDER_MIN <= (K) && (K) <= VYV_ORDER_MAX)

//Deriche
#define DERICHE_ORDER_MIN       2
#define DERICHE_ORDER_MAX       4
#define DERICHE_VALID_ORDER(K)  (DERICHE_ORDER_MIN <= (K) && (K) <= DERICHE_ORDER_MAX)

#define COLOR_WHITE cv::Scalar(255,255,255)
#define COLOR_GRAY10 cv::Scalar(10,10,10)
#define COLOR_GRAY20 cv::Scalar(20,20,20)
#define COLOR_GRAY30 cv::Scalar(10,30,30)
#define COLOR_GRAY40 cv::Scalar(40,40,40)
#define COLOR_GRAY50 cv::Scalar(50,50,50)
#define COLOR_GRAY60 cv::Scalar(60,60,60)
#define COLOR_GRAY70 cv::Scalar(70,70,70)
#define COLOR_GRAY80 cv::Scalar(80,80,80)
#define COLOR_GRAY90 cv::Scalar(90,90,90)
#define COLOR_GRAY100 cv::Scalar(100,100,100)
#define COLOR_GRAY110 cv::Scalar(101,110,110)
#define COLOR_GRAY120 cv::Scalar(120,120,120)
#define COLOR_GRAY130 cv::Scalar(130,130,140)
#define COLOR_GRAY140 cv::Scalar(140,140,140)
#define COLOR_GRAY150 cv::Scalar(150,150,150)
#define COLOR_GRAY160 cv::Scalar(160,160,160)
#define COLOR_GRAY170 cv::Scalar(170,170,170)
#define COLOR_GRAY180 cv::Scalar(180,180,180)
#define COLOR_GRAY190 cv::Scalar(190,190,190)
#define COLOR_GRAY200 cv::Scalar(200,200,200)
#define COLOR_GRAY210 cv::Scalar(210,210,210)
#define COLOR_GRAY220 cv::Scalar(220,220,220)
#define COLOR_GRAY230 cv::Scalar(230,230,230)
#define COLOR_GRAY240 cv::Scalar(240,240,240)
#define COLOR_GRAY250 cv::Scalar(250,250,250)
#define COLOR_BLACK cv::Scalar(0,0,0)

#define COLOR_RED cv::Scalar(0,0,255)
#define COLOR_GREEN cv::Scalar(0,255,0)
#define COLOR_BLUE cv::Scalar(255,0,0)
#define COLOR_ORANGE cv::Scalar(0,100,255)
#define COLOR_YELLOW cv::Scalar(0,255,255)
#define COLOR_MAGENDA cv::Scalar(255,0,255)
#define COLOR_CYAN cv::Scalar(255,255,0)

//utility function
namespace util
{
	void cvt32F8U(const cv::Mat& src, cv::Mat& dest);
	double Combination(const int N, const int n);

	double MSE_64F(cv::Mat img1, cv::Mat img2);
	double trancatedPSNR(cv::Mat img1, cv::Mat img2, const double diff_max);
	double PSNR_64F(cv::Mat img1, cv::Mat img2);
	double Linf_64F(cv::Mat img1, cv::Mat img2);

	void cvt32F16F(cv::Mat& srcdst);

	void bilateralFilter32f(const cv::Mat& src, cv::Mat& dst, const cv::Size kernelSize, float sigma_range, float sigma_space, const int borderType, const bool isRectangle, const bool isKahan);
	void bilateralFilter64f(const cv::Mat& src, cv::Mat& dst, const cv::Size kernelSize, double sigma_range, double sigma_space, const int borderType, const bool isRectangle, const bool isKahan);
	void bilateralFilter64f_Laplacian(const cv::Mat& src, cv::Mat& dst, const cv::Size kernelSize, double sigma_range, double sigma_space, const int borderType, const bool isRectangle);
	void bilateralFilter64f_Hat(const cv::Mat& src, cv::Mat& dst, const cv::Size kernelSize, double sigma_range, double sigma_space, const int borderType, const bool isRectangle);

	template<typename T>
	inline int typeToDepth();

	void drawMinMax(cv::Mat& src, cv::Mat& dest, const uchar minv = 0, const uchar maxv = 255, cv::Scalar minColor = cv::Scalar(0, 0, 255), cv::Scalar maxColor = cv::Scalar(255, 0, 0), const int circle_r = 3);

	cv::Size getSubImageAlignSize(const cv::Size src, const cv::Size div_size, const int r, const int align_x, const int align_y, const int left_multiple = 1, const int top_multiple = 1);
	cv::Size getSubImageSize(const cv::Size src, const cv::Size div_size, const int r);
	void createSubImage(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType = cv::BORDER_DEFAULT);
	void createSubImage(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT);
	void createSubImageAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	void setSubImage(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int top, const int left);
	void setSubImage(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r);

	void setSubImageAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int left_multiple = 1, const int top_multiple = 1);

	void mergeSubImage(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const int r);
	void mergeSubImageAlign(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const int r, const int left_multiple = 1, const int top_multiple = 1);
	void splitSubImage(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const int r, const int borderType = cv::BORDER_DEFAULT);
	void splitSubImageAlign(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int left_multiple = 1, const int top_multiple = 1);

	void calcMinMax(const cv::Mat& src, uchar& minv, uchar& maxv);
	void calcMaxDiffParallel(const cv::Mat& src, cv::Mat& buff, const int r, int& T);
	int calcMaxDiff(const cv::Mat& src, cv::Mat& buff, const int r);
	int calcMaxDiffV(const cv::Mat& src, const int rad);
	void imshowNormalize(std::string wname, cv::Mat& src);

	void blockAnalysis(cv::Mat& reference, cv::Mat& ideal, cv::Mat& filtered, cv::Size div, const int r);
	void downsampleNN(cv::InputArray src, cv::OutputArray dest, const int scale);
	void downsampleArea(cv::InputArray src, cv::OutputArray dest, const int scale);
	void upsampleNN(cv::InputArray src, cv::OutputArray dest, const int scale);
	void upsampleLinear(cv::InputArray src, cv::OutputArray dest, const int scale);
	void upsampleCubic(cv::InputArray src, cv::OutputArray dest, const int scale, const double a = -1.0);

	//void upsample32fLinearScale2(cv::Mat& src, cv::Mat& dest);
	//void upsample32fCubicScale2(cv::Mat& src, cv::Mat& dest, double a = -1.0);
	//void upsample32fCubicScale4(cv::Mat& src, cv::Mat& dest, double a = -1.0);

	void copyMakeBorderInteral(cv::Mat& src, cv::Mat& dest, int top, int bottom, int left, int right, int borderType);
	int countDenormalizedNumber(const cv::Mat& src);
	double countDenormalizedNumberRatio(const cv::Mat& src);

	class UpdateCheck
	{
		std::vector<double> parameters;
	public:

		UpdateCheck(double p0);
		UpdateCheck(double p0, double p1);
		UpdateCheck(double p0, double p1, double p2);
		UpdateCheck(double p0, double p1, double p2, double p3);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5);

		bool isUpdate(double p0);
		bool isUpdate(double p0, double p1);
		bool isUpdate(double p0, double p1, double p2);
		bool isUpdate(double p0, double p1, double p2, double p3);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5);
	};

	class Stat
	{
	public:
		std::vector<double> data;
		int num_data;
		Stat();
		~Stat();
		double getSum();
		double getMin();
		double getMax();
		double getMean();
		double getStd();
		double getMedian();

		void push_back(double val);

		void clear();
		void show();
		void drawStat(std::string wname, int div);
		void drawStat(std::string wname, int div, double min, double max);
	};

	void triangle(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar& color, int thickness = 1);
	void triangleinv(cv::InputOutputArray src, cv::Point pt, int length, cv::Scalar& color, int thickness = 1);
	void drawPlus(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);
	void drawTimes(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_typee = 8, int shift = 0);
	void drawGrid(cv::InputOutputArray src, cv::Point crossCenter, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);
	void drawAsterisk(cv::InputOutputArray src, cv::Point crossCenter, int length, cv::Scalar& color, int thickness = 1, int line_type = 8, int shift = 0);

	class Plot
	{
	protected:
		struct PlotInfo
		{
			std::vector<cv::Point2d> data;
			cv::Scalar color;
			int symbolType;
			int lineType;
			int thickness;

			std::string keyname;
		};
		std::vector<PlotInfo> pinfo;

		std::string xlabel;
		std::string ylabel;

		int data_max;

		cv::Scalar background_color;

		cv::Size plotsize;
		cv::Point origin;

		double xmin;
		double xmax;
		double ymin;
		double ymax;
		double xmax_no_margin;
		double xmin_no_margin;
		double ymax_no_margin;
		double ymin_no_margin;

		void init();
		void point2val(cv::Point pt, double* valx, double* valy);

		bool isZeroCross;
		bool isXYMAXMIN;
		bool isXYCenter;

		bool isPosition;
		cv::Scalar getPseudoColor(uchar val);
		cv::Mat plotImage;
		cv::Mat keyImage;
	public:
		//symbolType
		enum
		{
			SYMBOL_NOPOINT = 0,
			SYMBOL_PLUS,
			SYMBOL_TIMES,
			SYMBOL_ASTERRISK,
			SYMBOL_CIRCLE,
			SYMBOL_RECTANGLE,
			SYMBOL_CIRCLE_FILL,
			SYMBOL_RECTANGLE_FILL,
			SYMBOL_TRIANGLE,
			SYMBOL_TRIANGLE_FILL,
			SYMBOL_TRIANGLE_INV,
			SYMBOL_TRIANGLE_INV_FILL,
		};

		//lineType
		enum
		{
			LINE_NONE,
			LINE_LINEAR,
			LINE_H2V,
			LINE_V2H
		};

		cv::Mat render;
		cv::Mat graphImage;

		Plot(cv::Size window_size = cv::Size(1024, 768));
		~Plot();

		void setXYOriginZERO();
		void setXOriginZERO();
		void setYOriginZERO();

		void recomputeXYMAXMIN(bool isCenter = false, double marginrate = 0.9);
		void setPlotProfile(bool isXYCenter_, bool isXYMAXMIN_, bool isZeroCross_);
		void setPlotImageSize(cv::Size s);
		void setXYMinMax(double xmin_, double xmax_, double ymin_, double ymax_);
		void setXMinMax(double xmin_, double xmax_);
		void setYMinMax(double ymin_, double ymax_);
		void setBackGoundColor(cv::Scalar cl);

		void makeBB(bool isFont);

		void setPlot(int plotnum, cv::Scalar color = COLOR_RED, int symboltype = SYMBOL_PLUS, int linetype = LINE_LINEAR, int thickness = 1);
		void setPlotThickness(int plotnum, int thickness_);
		void setPlotColor(int plotnum, cv::Scalar color);
		void setPlotSymbol(int plotnum, int symboltype);
		void setPlotLineType(int plotnum, int linetype);
		void setPlotKeyName(int plotnum, std::string name);

		void setPlotSymbolALL(int symboltype);
		void setPlotLineTypeALL(int linetype);

		void plotPoint(cv::Point2d = cv::Point2d(0.0, 0.0), cv::Scalar color = COLOR_BLACK, int thickness_ = 1, int linetype = LINE_LINEAR);
		void plotGrid(int level);
		void plotData(int gridlevel = 0, int isKey = 0);

		void plotMat(cv::InputArray src, std::string name = "Plot", bool isWait = true, std::string gnuplotpath = "pgnuplot.exe");
		void plot(std::string name = "Plot", bool isWait = true, std::string gnuplotpath = "pgnuplot.exe");

		void makeKey(int num);

		void save(std::string name);

		void push_back(std::vector<cv::Point> point, int plotIndex = 0);
		void push_back(std::vector<cv::Point2d> point, int plotIndex = 0);
		void push_back(double x, double y, int plotIndex = 0);

		void erase(int sampleIndex, int plotIndex = 0);
		void insert(cv::Point2d v, int sampleIndex, int plotIndex = 0);
		void insert(cv::Point v, int sampleIndex, int plotIndex = 0);
		void insert(double x, double y, int sampleIndex, int plotIndex = 0);

		void clear(int datanum = -1);

		void swapPlot(int plotIndex1, int plotIndex2);
	};

	enum
	{
		PLOT_ARG_MAX = 1,
		PLOT_ARG_MIN = -1
	};
	class Plot2D
	{
		std::vector<std::vector<double>> data;
		cv::Mat graphBase;
		int w;
		int h;
		void createPlot();
		void setMinMaxX(double minv, double maxv, int count);
		void setMinMaxY(double minv, double maxv, int count);
	public:
		cv::Mat show;
		cv::Mat graph;
		cv::Size size;
		double minx;
		double maxx;
		int countx;

		double miny;
		double maxy;
		int county;

		Plot2D(cv::Size graph_size, double xmin, double xmax, double xstep, double ymin, double ymax, double ystep);

		void setMinMax(double xmin, double xmax, double xstep, double ymin, double ymax, double ystep);
		void add(int x, int y, double val);
		void writeGraph(bool isColor, int arg_min_max, double minvalue = 0, double maxvalue = 0, bool isMinMaxSet = false);
		void setLabel(std::string namex, std::string namey);
		//void plot(CSV& result, vector<ExperimentalParameters>& parameters);
	};

	void plotGraph(cv::OutputArray graph, std::vector<cv::Point2d>& data, double xmin, double xmax, double ymin, double ymax,
		cv::Scalar color = COLOR_RED, int lt = Plot::SYMBOL_PLUS, int isLine = Plot::LINE_LINEAR, int thickness = 1, int ps = 4);

	int get_simd_floor(const int val, const int simdwidth);
	int get_simd_ceil(const int val, const int simdwidth);

	void splitBGRLineInterleaveAVX(cv::InputArray src, cv::OutputArray dest);

	class Search1DInt
	{
		int xmin;
		int xmax;
		int x1;
		int x2;
		const double phi = (1.0 + sqrt(5.0))*0.5;//for golden search
		int research = 2;

		virtual double getError(int x) = 0;
		int getGSLeft();
		int getGSRight();

	public:
		int linearSearch(const int search_min, const int search_max);
		int goldenSearch(const int search_min, const int search_max);
	};

	class Search1D
	{
		double xmin;
		double xmax;
		double x1;
		double x2;
		const double phi = (1.0 + sqrt(5.0))*0.5;//for golden search
		
		virtual double getError(double x) = 0;
		double getGSLeft();
		double getGSRight();

	public:
		double linearSearch(const double search_min, const double search_max, const double step, bool isPlot=false, std::string wname="search");
		double binarySearch(const double search_min, const double search_max, const double eps, const int loop);
		double goldenSectionSearch(const double search_min, const double search_max, const double eps, const int loop);
	};
}