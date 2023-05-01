#pragma once

//splatting with GF
//#define COMPILE_GFforBF_DCT3 1
//#define COMPILE_GFforBF_DCT5 1

#define COMPILE_GF_DCT3_16F_ORDER_TEMPLATE 1
#define COMPILE_GF_DCT3_32F_ORDER_TEMPLATE 1
#define COMPILE_GF_DCT3_64F_ORDER_TEMPLATE 1

#define COMPILE_GF_DCT5_16F_ORDER_TEMPLATE 1
#define COMPILE_GF_DCT5_32F_ORDER_TEMPLATE 1
#define COMPILE_GF_DCT5_64F_ORDER_TEMPLATE 1

//#define COMPILE_COLOR_BF_TEST 1

#include <opencv2/opencv.hpp>
#define __AVX2__ 1
#define __FMA__ 1
#define __AVX__ 1


#include "common.hpp"
//#include "avx_util.hpp"
//#include "GaussianFilter.hpp"

//#include <opencp.hpp>





//#include "ConstantTimeBFColor_NormalForm_template.hpp"
