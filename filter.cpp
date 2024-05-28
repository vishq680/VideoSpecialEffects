/*
    * File: filter.cpp
    * Created on : Jan 23,2024
    Team Members:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628
    Name: Vishaq Jayakumar
    NUID: 002737793

    Purpose: This file contains the implementation of the functions defined in filter.h
*/

// Include the header files
#include "filter.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"

// Function to convert an image to the alternate greyscale
/*
 Arguments:
 cv::Mat src - A source input image to convert to greyscale.
 cv::Mat dst - A destination image, converted to greyscale
 */
int greyscale(cv::Mat &src, cv::Mat &dst)
{
    // Check if the source image is empty
    if (src.empty())
    {
        // Return -1 if the source image is empty
        return -1;
    }

    // Clone the source image to the destination image
    dst = src.clone();

    // Loop over each pixel in the image
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {

            // Access the color of the current pixel in the destination image
            cv::Vec3b &color = dst.at<cv::Vec3b>(y, x);

            // Convert the color to greyscale using a simple formula
            // Here, it subtracts the red component from 255 to obtain the greyscale intensity
            int grey = 255 - color[2];

            // Set the greyscale intensity for all three color channels
            color[0] = grey;
            color[1] = grey;
            color[2] = grey;
        }
    }

    return 0;
}

// Function to apply sepia filter to an image
/*
 Arguments:
 cv::Mat src - A source input image
 cv::Mat dst - Destination image, converted to sepia
 */
int sepia(cv::Mat &src, cv::Mat &dst)
{
    // Check if the source image is empty
    if (src.empty())
    {
        return -1;
    }

    // Clone the source image to the destination image
    dst = src.clone();

    // Loop over each pixel in the image
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            // Access the color of the current pixel in the destination image
            cv::Vec3b &color = dst.at<cv::Vec3b>(y, x);
            int oldBlue = color[0];
            int oldGreen = color[1];
            int oldRed = color[2];

            // Apply the sepia filter to each color channel
            color[0] = cv::saturate_cast<uchar>(0.272 * oldRed + 0.534 * oldGreen + 0.131 * oldBlue);
            color[1] = cv::saturate_cast<uchar>(0.349 * oldRed + 0.686 * oldGreen + 0.168 * oldBlue);
            color[2] = cv::saturate_cast<uchar>(0.393 * oldRed + 0.769 * oldGreen + 0.189 * oldBlue);
        }
    }

    return 0;
}

/*
 Function to apply a 5x5 blur kernel to an image
 Arguments:
   - cv::Mat src: Source image
   - cv::Mat dst: Destination image, after applying the blur
 Returns:
   - 0 on success
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
    // Clone the source image to the destination image
    dst = src.clone();

    // Define a 5x5 blur kernel
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}};

    for (int y = 2; y < src.rows - 2; y++)
    {
        for (int x = 2; x < src.cols - 2; x++)
        {
            cv::Vec3f sum = cv::Vec3f(0, 0, 0);
            int kernelSum = 0;

            for (int ky = -2; ky <= 2; ky++)
            {
                for (int kx = -2; kx <= 2; kx++)
                {
                    // Access the color of the current pixel in the source image
                    cv::Vec3b color = src.at<cv::Vec3b>(y + ky, x + kx);
                    int weight = kernel[ky + 2][kx + 2];

                    // Accumulate the weighted color components
                    sum[0] += weight * color[0];
                    sum[1] += weight * color[1];
                    sum[2] += weight * color[2];

                    // Accumulate the kernel weight
                    kernelSum += weight;
                }
            }

            // Calculate the average color in the local neighborhood and assign it to the destination pixel
            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(sum[0] / kernelSum, sum[1] / kernelSum, sum[2] / kernelSum);
        }
    }

    return 0;
}

/*
 Function to apply a separable 5x5 blur kernel to an image
 Arguments:
   - src: Source image
   - dst: Destination image, after applying the blur
 Returns:
   - 0 on success
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
    // Define padding size
    int padding = 2;

    // Pad the source image with zeros so that the edges are also blurred
    cv::Mat paddedSrc;
    cv::copyMakeBorder(src, paddedSrc, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat temp = cv::Mat::zeros(paddedSrc.size(), paddedSrc.type());
    dst = cv::Mat::zeros(paddedSrc.size(), paddedSrc.type());

    // Define a 1x5 filter kernel
    int kernel[5] = {1, 2, 4, 2, 1};

    // Apply 1x5 filter horizontally
    for (int y = 0; y < paddedSrc.rows; y++)
    {
        for (int x = padding; x < paddedSrc.cols - padding; x++)
        {
            cv::Vec3f sum = cv::Vec3f(0, 0, 0);
            int kernelSum = 0;

            for (int kx = -padding; kx <= padding; kx++)
            {
                cv::Vec3b color = paddedSrc.ptr<cv::Vec3b>(y)[x + kx];
                int weight = kernel[kx + padding];

                sum[0] += weight * color[0];
                sum[1] += weight * color[1];
                sum[2] += weight * color[2];

                kernelSum += weight;
            }

            temp.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(sum[0] / kernelSum, sum[1] / kernelSum, sum[2] / kernelSum);
        }
    }

    // Apply 1x5 filter vertically
    for (int y = padding; y < paddedSrc.rows - padding; y++)
    {
        for (int x = 0; x < paddedSrc.cols; x++)
        {
            cv::Vec3f sum = cv::Vec3f(0, 0, 0);
            int kernelSum = 0;

            for (int ky = -padding; ky <= padding; ky++)
            {
                cv::Vec3b color = temp.ptr<cv::Vec3b>(y + ky)[x];
                int weight = kernel[ky + padding];

                sum[0] += weight * color[0];
                sum[1] += weight * color[1];
                sum[2] += weight * color[2];

                kernelSum += weight;
            }

            dst.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(sum[0] / kernelSum, sum[1] / kernelSum, sum[2] / kernelSum);
        }
    }

    // Remove padding
    dst = dst(cv::Rect(padding, padding, src.cols, src.rows));

    return 0;
}

// Function to apply sobel filter in X direction
/*
Arguments:
- src: Source image
- dst: Destination image, after applying the sobel filter
Returns:
- 0 on success
*/
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    // Check if the source image is empty
    if (src.empty())
    {
        return -1;
    }

    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // Define the 3x3 Sobel kernel for the X direction
    int kernel[3] = {-1, 0, 1};

    // Loop over each pixel in the image, excluding the border
    for (int y = 1; y < src.rows - 1; y++)
    {
        for (int x = 1; x < src.cols - 1; x++)
        {
            // Initialize the sum for each color channel
            cv::Vec3s sum = cv::Vec3s(0, 0, 0);

            // Convolve with the 3x3 Sobel filter in the horizontal direction
            for (int kx = -1; kx <= 1; kx++)
            {
                cv::Vec3b color = src.at<cv::Vec3b>(y, x + kx);
                int weight = kernel[kx + 1];

                // Apply the filter to each color channel
                sum[0] += weight * color[0];
                sum[1] += weight * color[1];
                sum[2] += weight * color[2];
            }

            // Assign the result to the destination image
            dst.at<cv::Vec3s>(y, x) = sum;
        }
    }

    return 0;
}

// Function to apply sobel filter in Y direction
/*
Arguments:
- src: Source image
- dst: Destination image, after applying the sobel filter
Returns:
- 0 on success
*/
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    int kernel[3] = {-1, 0, 1};

    for (int y = 1; y < src.rows - 1; y++)
    {
        for (int x = 1; x < src.cols - 1; x++)
        {
            cv::Vec3s sum = cv::Vec3s(0, 0, 0);

            for (int ky = -1; ky <= 1; ky++)
            {
                cv::Vec3b color = src.at<cv::Vec3b>(y + ky, x);
                int weight = kernel[ky + 1];

                sum[0] += weight * color[0];
                sum[1] += weight * color[1];
                sum[2] += weight * color[2];
            }

            dst.at<cv::Vec3s>(y, x) = sum;
        }
    }

    return 0;
}

// Function to compute the magnitude of the gradient of an image
/*
Arguments:
- src: Source image
- dst: Destination image, after calculating the gradient magnitude
Returns:
- 0 on success
*/
int magnitude(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    cv::Mat sx, sy;

    // Create intermediate images for Sobel gradients in X and Y directions
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);

    dst = cv::Mat::zeros(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; y++)
    {
        for (int x = 0; x < sx.cols; x++)
        {
            // Extract color components from Sobel gradient images
            cv::Vec3s colorX = sx.at<cv::Vec3s>(y, x);
            cv::Vec3s colorY = sy.at<cv::Vec3s>(y, x);

            cv::Vec3b &colorDst = dst.at<cv::Vec3b>(y, x);

            // Calculate the magnitude of the gradient for each color channel
            for (int i = 0; i < 3; i++)
            {
                float mag = std::sqrt(colorX[i] * colorX[i] + colorY[i] * colorY[i]);
                colorDst[i] = cv::saturate_cast<uchar>(mag);
            }
        }
    }

    return 0;
}

// Function to apply blur and quantization to an image
/*
Arguments:
- src: Source image
- dst: Destination image, after applying blur and quantization
- levels: Number of quantization levels
Returns:
- 0 on success
*/
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels)
{
    if (src.empty())
    {
        return -1;
    }

    // Apply a 5x5 blur to the source image
    cv::Mat blurred;
    blur5x5_2(src, blurred);

    dst = cv::Mat::zeros(blurred.size(), CV_8UC3);
    int bucketSize = 255 / levels;

    for (int y = 0; y < blurred.rows; y++)
    {
        for (int x = 0; x < blurred.cols; x++)
        {
            // Extract color components from the blurred image
            cv::Vec3b colorSrc = blurred.at<cv::Vec3b>(y, x);
            cv::Vec3b &colorDst = dst.at<cv::Vec3b>(y, x);

            // Apply quantization to each color channel
            for (int i = 0; i < 3; i++)
            {
                int temp = colorSrc[i] / bucketSize;
                colorDst[i] = temp * bucketSize;
            }
        }
    }

    return 0;
}

// Function to apply emboss effect to an image
/*
Arguments:
- src: Source image
- dst: Destination image, after applying the embossing effect
Returns:
- 0 on success
*/
int emboss(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
    {
        return -1;
    }

    // Calculate the horizontal and vertical Sobel gradients
    cv::Mat sobelX, sobelY;
    cv::Sobel(src, sobelX, CV_32F, 1, 0);
    cv::Sobel(src, sobelY, CV_32F, 0, 1);

    dst = cv::Mat::zeros(src.size(), CV_32F);

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            cv::Vec3f colorX = sobelX.at<cv::Vec3f>(y, x);
            cv::Vec3f colorY = sobelY.at<cv::Vec3f>(y, x);

            // Calculate the dot product of the Sobel gradients
            float dotProduct = colorX[0] * 0.7071 + colorY[0] * 0.7071;
            dst.at<float>(y, x) = dotProduct;
        }
    }

    // Normalize the dst image for display
    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
    cv::convertScaleAbs(dst, dst);

    return 0;
}

// Function to create a negative of an image
/*
Arguments:
- src: Source image
- dst: Destination image, after creating the negative image
Returns:
- 0 on success
*/
int negativeImage(cv::Mat &src, cv::Mat &dst)
{
    // Calculate the negative image using the formula: dst = 255 - src
    dst = cv::Scalar::all(255) - src;
    return 0;
}

// Function to apply color pop effect to an image
/*
Arguments:
- src: Source image
- dst: Destination image, after extracting a specific color range
- lowerb: Lower bound of the color range (input), specified as a Scalar in HSV color space
- upperb: Upper bound of the color range (input), specified as a Scalar in HSV color space
Returns:
- 0 on success
*/
int colorPop(cv::Mat &src, cv::Mat &dst, cv::Scalar lowerb, cv::Scalar upperb)
{
    // Convert the image to the HSV color space
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // Create a mask of the pixels that are in the range of the strong color
    cv::Mat mask;
    cv::inRange(hsv, lowerb, upperb, mask);

    // Convert the original image to grayscale
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);

    // Use the mask to copy the strong color pixels from the original image to the grayscale image
    src.copyTo(dst, mask);

    return 0;
}

// Function to apply cartoon effect to an image
/*
Arguments:
- src: Source image
- dst: Destination image, after applying the cartoon effect
Returns:
- 0 on success
*/
int cartoonize(cv::Mat &src, cv::Mat &dst)
{
    //  Apply a bilateral filter to reduce the color palette
    cv::Mat imgColor;
    for (int i = 0; i < 7; i++)
        cv::bilateralFilter(src, imgColor, 9, 9, 7);

    //  Convert the original image to grayscale
    cv::Mat imgGray;
    cv::cvtColor(src, imgGray, cv::COLOR_BGR2GRAY);

    //  Apply a median blur to reduce noise
    cv::medianBlur(imgGray, imgGray, 7);

    //  Use adaptive thresholding to detect and emphasize the edges
    cv::Mat imgEdge;
    cv::adaptiveThreshold(imgGray, imgEdge, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 2);

    //  Combine the color and edge images to create a cartoon effect
    cv::cvtColor(imgEdge, imgEdge, cv::COLOR_GRAY2BGR);
    dst = imgColor & imgEdge;

    return 0;
}

// Function to apply sketch effect to an image
/*
Arguments:
- src: Source image
- dst: Destination image, after applying the sketch effect
Returns:
- 0 on success
*/
int sketch(cv::Mat &src, cv::Mat &dst)
{
    //  Convert the image to grayscale
    cv::Mat imgGray;
    cv::cvtColor(src, imgGray, cv::COLOR_BGR2GRAY);

    //  Apply a Gaussian blur to the grayscale image
    cv::Mat blurredFrame;
    cv::GaussianBlur(imgGray, blurredFrame, cv::Size(5, 5), 0);

    //  Detect the edges in the blurred image using the Laplacian function
    cv::Mat edgeFrame;
    cv::Laplacian(blurredFrame, edgeFrame, CV_8U, 5);

    //  Invert the colors of the edge-detected image to get a sketch effect
    cv::threshold(edgeFrame, dst, 100, 255, cv::THRESH_BINARY_INV);

    return 0;
}