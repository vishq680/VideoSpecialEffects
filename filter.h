/*
    * File: filter.h
    * Created on: Jan 23,2024
    Team members:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628
    Name: Vishaq Jayakumar
    NUID: 002737793

    Purpose: This file contains the function declarations for the filters.


*/

#ifndef FILTER_H // If FILTER_H is not defined
#define FILTER_H // Define FILTER_H

#include <opencv2/opencv.hpp> // Include OpenCV library

// Function to convert an image to greyscale
int greyscale(cv::Mat &src, cv::Mat &dst);

// Function to apply sepia filter to an image
int sepia(cv::Mat &src, cv::Mat &dst);

// Function to apply 5x5 blur filter to an image (version 1)
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

// Function to apply 5x5 blur filter to an image (version 2)
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

// Function to apply 3x3 Sobel filter in X direction to an image
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

// Function to apply 3x3 Sobel filter in Y direction to an image
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// Function to compute the magnitude of the gradient of an image
int magnitude(cv::Mat &src, cv::Mat &dst);

// Function to apply blur and quantization to an image
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

// Function to apply emboss effect to an image
int emboss(cv::Mat &src, cv::Mat &dst);

// Function to create a negative of an image
int negativeImage(cv::Mat &src, cv::Mat &dst);

// Function to apply color pop effect to an image
int colorPop(cv::Mat &src, cv::Mat &dst, cv::Scalar lowerb, cv::Scalar upperb);

// Function to apply cartoon effect to an image
int cartoonize(cv::Mat &src, cv::Mat &dst);

// Function to apply sketch effect to an image
int sketch(cv::Mat &src, cv::Mat &dst);

#endif // FILTER_H