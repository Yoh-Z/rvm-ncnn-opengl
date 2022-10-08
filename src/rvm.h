#pragma once
#include <net.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/core/core.hpp>
#include <memory>

class RVM {
public:
	RVM();
	~RVM();
	int m_init(int w, int h, const char* path, int nFlag);
	int matting(cv::Mat& mPic, cv::Mat& mask, cv::Mat& foreground);
	int draw(cv::Mat& mPic, cv::Mat& mask, cv::Mat& bg, cv::Mat& fgr);
	int chw2whc(float* pha_data, uchar * pha);
private:
	int times;
	int mW, mH;
	uchar* cv_pha;
	uchar* cv_fgr;
	std::shared_ptr<ncnn::Net> mNet;
	std::vector<ncnn::Mat> mPics;
	std::vector<ncnn::VkMat> mVkPics;
	const int mSize[3][4] = {{135, 68, 34, 17}, {240, 120, 60, 30}, { 16, 20, 40, 64 }};
};