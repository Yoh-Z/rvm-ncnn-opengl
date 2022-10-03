#include <vector>
#include <iostream>
#include <gpu.h>
#include "rvm.h"

#define NCNN_GPU 0

RVM::RVM() :
	mNet(nullptr),
	mH(0),
	mW(0),
	mPics(0),
	cv_pha(nullptr),
	times(0),
	mVkPics(0){

}
RVM::~RVM() {
	if (cv_pha != nullptr) delete[]cv_pha;
#if NCNN_GPU
	mNet->vulkan_device()->reclaim_blob_allocator(mNet->opt.blob_vkallocator);
	mNet->vulkan_device()->reclaim_staging_allocator(mNet->opt.staging_vkallocator);
	mNet->vulkan_device()->reclaim_blob_allocator(mNet->opt.workspace_vkallocator);
#endif // NCNN_GPU
}

int RVM::m_init(int w, int h, const char* path, int nFlag) {
	mW = w, mH = h;

	mPics.resize(4);
	mVkPics.resize(4);
	for (int i = 0; i < 4; i++)
	{
		mPics[i] = ncnn::Mat(mSize[0][i], mSize[1][i], mSize[2][i]), mPics[i].fill(0);
		std::cout << mSize[0][i] << " " << mSize[1][i] << " " << mSize[2][i] << std::endl;
	}

	mNet = std::make_shared<ncnn::Net>();

	if (mNet == nullptr)
	{
		return 1;
	}

	if (nFlag == 1)
	{
		mNet->opt.use_vulkan_compute = 1;
		mNet->opt.use_fp16_packed = false;
		mNet->opt.use_fp16_storage = false;
		mNet->opt.use_fp16_arithmetic = false;
		mNet->opt.use_int8_storage = false;
		mNet->opt.use_int8_arithmetic = false;
	}

	std::string model = path;
	model += "\\..\\model\\rvm_mobilenetv3_fp32-1080-1920-opt.bin.bin";
	std::string param = path;
	param += "\\..\\model\\rvm_mobilenetv3_fp32-1080-1920-opt.bin.param";
	
	std::cout << model << std::endl;
	std::cout << param << std::endl;

	mNet->load_param(param.c_str());
	mNet->load_model(model.c_str());

	printf("load success!\n");

	if (nFlag == 1)
	{
		mNet->opt.blob_vkallocator = mNet->vulkan_device()->acquire_blob_allocator();
		mNet->opt.staging_vkallocator = mNet->vulkan_device()->acquire_staging_allocator();
		mNet->opt.workspace_vkallocator = mNet->vulkan_device()->acquire_blob_allocator();
	}

	cv_pha = new uchar[mW * mH];

	return 0;
}

int RVM::matting(cv::Mat& mPic, cv::Mat& mask, cv::Mat& foreground) {
	ncnn::Extractor ex_matting = mNet->create_extractor();
	ex_matting.set_num_threads(1);
	ncnn::Mat ncnn_in;

	long long time1 = cv::getTickCount();
	ncnn_in = ncnn::Mat::from_pixels_resize(mPic.data, ncnn::Mat::PIXEL_RGB, mPic.cols, mPic.rows, mW, mH);
	const float means[3] = { 0,0,0 };
	const float norms[3] = { 1 / 255.0, 1 / 255.0, 1 / 255.0 };
	ncnn_in.substract_mean_normalize(means, norms);
	long long time2 = cv::getTickCount();
	std::cout << ((double)time2 - time1) / cv::getTickFrequency() << std::endl;

	ncnn::Mat pha, fgr;
	ex_matting.input("src", ncnn_in);
	if (times == 0) {
		ex_matting.input("r1i", mPics[0]);
		ex_matting.input("r2i", mPics[1]);
		ex_matting.input("r3i", mPics[2]);
		ex_matting.input("r4i", mPics[3]);
	}
	else {
		ex_matting.input("r1i", mVkPics[0]);
		ex_matting.input("r2i", mVkPics[1]); 
		ex_matting.input("r3i", mVkPics[2]);
		ex_matting.input("r4i", mVkPics[3]);
	}
	
#if NCNN_GPU
	ncnn::VkCompute cmd(mNet->vulkan_device());
	ex_matting.extract("r4o", mVkPics[3], cmd);
	ex_matting.extract("r3o", mVkPics[2], cmd);
	ex_matting.extract("r2o", mVkPics[1], cmd);
	ex_matting.extract("r1o", mVkPics[0], cmd);
	cmd.submit_and_wait();
#else
	ex_matting.extract("r4o", mPics[3]);
	ex_matting.extract("r3o", mPics[2]);
	ex_matting.extract("r2o", mPics[1]);
	ex_matting.extract("r1o", mPics[0]);
#endif // NCNN_GPU
	ex_matting.extract("pha", pha);
	ex_matting.extract("fgr", fgr);
	ex_matting.clear();

	float *pha_data = (float*)pha.data;
	int ret = chw2whc(pha_data, cv_pha);
	if (ret == 1) return 1;
	
	mask = cv::Mat(mPic.size(), CV_8UC1, cv_pha);
#if NCNN_GPU
	++times;
#endif
	if (times < 0 || times == INT_MAX) return 1;

	return 0;
}

int RVM::draw(cv::Mat& mPic, cv::Mat& mask, cv::Mat& bg, cv::Mat& fgr) {
	cv::Mat alpha;
	cv::resize(mask, alpha, mPic.size());
	cv::Size sz = mPic.size();
	cv::resize(fgr, fgr, mPic.size());
	const int color[] = { 120, 255, 155 };
	uchar data;

	for (int i = 0; i < alpha.rows; i++)
	{
		for (int j = 0; j < alpha.cols; j++)
		{
			data = alpha.at<uchar>(i, j);
			float alpha = (float)data / 255;
			cv::Vec3b bgcolor = bg.at < cv::Vec3b>(i, j);

			mPic.at < cv::Vec3b>(i, j)[0] = mPic.at < cv::Vec3b>(i, j)[0] * alpha + (1 - alpha) * bgcolor[0];
			mPic.at < cv::Vec3b>(i, j)[1] = mPic.at < cv::Vec3b>(i, j)[1] * alpha + (1 - alpha) * bgcolor[1];
			mPic.at < cv::Vec3b>(i, j)[2] = mPic.at < cv::Vec3b>(i, j)[2] * alpha + (1 - alpha) * bgcolor[2];
		}
	}

	return 0;
}


int RVM::chw2whc(float* pha_data, uchar * pha)
{
	if (pha_data == nullptr) return 1;
	int dst;

	#pragma omp parallel for num_threads(2)
	for (int i = 0; i < mH; i++)
	{
		int len = i * mW;
		for (int j = 0; j < mW; j++)
		{
			dst = len + j;
			pha[dst] = (uchar)(pha_data[dst] * 255.0f);
		}
	}

	return 0;
}