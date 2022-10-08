#include <iostream>

#include "opencv2/opencv.hpp"
#include <net.h>
#include "rvm.h"

#if defined(_MSC_VER)
#include <direct.h>
#define GetCurrentDir _getcwd
#elif defined(__unix__)
#include <unistd.h>
#define GetCurrentDir getcwd
#else
#endif

std::string get_current_directory()
{
    char buff[250];
    GetCurrentDir(buff, 250);
    std::string current_working_directory(buff);
    return current_working_directory;
}
int main(int argc, char** argv)
{
    RVM rvm;
    rvm.m_init(1920, 1080, get_current_directory().c_str(), 0);

    if (argc < 2 || argv == nullptr)
    {
        printf("please input iamge place");
        return -1;
    }

    cv::Mat pic = cv::imread(argv[1]);
    cv::resize(pic, pic, cv::Size(1920, 1080));

    printf("%d %d %d\n", pic.rows, pic.cols, pic.size);
    cv::Mat bgr, fgr;
    int ret = rvm.matting(pic, bgr, fgr);
    if (ret != 0)
    {
        printf("matting inference error!");
        return -1;
    }
    cv::Mat bg;
    rvm.draw(pic, bgr, bg, fgr);
    
    cv::imshow("pic", bgr);
    cv::waitKey(0);

    //ncnn::destroy_gpu_instance();
}