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
int main()
{
    RVM rvm;
    rvm.m_init(1920, 1080, get_current_directory().c_str(), 0);

    cv::Mat pic(1920, 1080, CV_8UC3);
    printf("%d %d %d\n", pic.rows, pic.cols, pic.size);
    cv::Mat bgr, fgr;
    rvm.matting(pic, bgr, fgr);
}