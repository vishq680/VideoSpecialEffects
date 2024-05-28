/*
    * vidDisplay.cpp
    *
    *  Created on: Jan 24, 2024
    Team Members:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628
    Name: Vishaq Jayakumar
    NUID: 002737793

    Pupose: This file contains the main function that displays the video stream from the webcam and applies filters to it.

*/

// Include necessary libraries
#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"
#include "faceDetect.h"

// Main function
int main(int, char **)
{
    // Task 1: Open the default camera
    cv::VideoCapture cap(0);

    // Check if webcam is opened successfully
    if (!cap.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // Define a map to store filter names and their corresponding keys
    std::map<char, std::string> filterNames = {
        {'g', "Grayscale"},
        {'c', "Normal Color"},
        {'h', "Alternate Grayscale"},
        {'v', "Sepia"},
        {'b', "Blur"},
        {'x', "Sobel X"},
        {'y', "Sobel Y"},
        {'m', "Magnitude"},
        {'l', "Quantize"},
        {'e', "Emboss"},
        {'n', "Negative"},
        {'p', "Color Pop"},
        {'a', "Cartoon"},
        {'f', "Face Detect"},
        {'t', "Sketch"},

    };

    // Define a vector to store faces
    std::vector<cv::Rect> faces;

    // Create a window
    cv::namedWindow("Video", 1);

    // Define matrices to store frames
    cv::Mat frame, quantizedImage;

    // Initialize frame count
    int frameCount = 0;

    // Initialize last key pressed
    char lastKey = 'c';

    // Print instructions to the console
    std::cout << "\n Please find below the list of operations that can be performed on the video stream:\n";
    std::cout << "Press a key to perform the operation\n";
    std::cout << "List of keys and their respective operations:\n";
    for (const auto &pair : filterNames)
    {
        std::cout << pair.first << ": " << pair.second << '\n';
    }
    std::cout << "Press 's' to save the current frame\n";
    std::cout << "Press 'q' to quit \n";
    std::cout << "Press '+' to increase brightness and contrast\n";
    std::cout << "Press '-' to decrease brightness and contrast\n";

    // Define a VideoWriter object
    cv::VideoWriter videoWriter;

    // Initialize recording status
    bool isRecording = false;

    // Initialize contrast control
    double alpha = 1.0;

    // Initialize brightness control
    int beta = 0;

    // Loop over all frames
    for (;;)
    {
        // Get a new frame from camera
        cap >> frame;

        // Break if frame is empty
        if (frame.empty())
        {
            std::cerr << "frame is empty\n";
            break;
        }

        // Apply filters based on the last key pressed

        // Task 3: Grayscale
        if (lastKey == 'g')
        {
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        }

        // Task 4: Alternate Grayscale
        if (lastKey == 'h')
        {
            cv::Mat greyFrame;
            if (greyscale(frame, greyFrame) == 0)
            {
                frame = greyFrame;
            }
        }

        // Task 5: Sepia Filter
        if (lastKey == 'v')
        {
            cv::Mat sepiaFrame;
            if (sepia(frame, sepiaFrame) == 0)
            {
                frame = sepiaFrame;
            }
        }

        // Task 6: Blur Filter
        if (lastKey == 'b')
        {
            cv::Mat blurFrame;
            if (blur5x5_2(frame, blurFrame) == 0)
            {
                frame = blurFrame;
            }
        }

        // Task 7: SobelX Filter
        if (lastKey == 'x')
        {
            cv::Mat sobelXFrame;
            if (sobelX3x3(frame, sobelXFrame) == 0)
            {
                cv::Mat displayImage;
                cv::convertScaleAbs(sobelXFrame, displayImage);
                frame = displayImage;
            }
        }

        // Task 7: SobelY filter
        if (lastKey == 'y')
        {
            cv::Mat sobelYFrame;
            if (sobelY3x3(frame, sobelYFrame) == 0)
            {

                cv::Mat displayImage;
                cv::convertScaleAbs(sobelYFrame, displayImage);
                frame = displayImage;
            }
        }

        // Task 8: Magnitude filter
        if (lastKey == 'm')
        {
            cv::Mat magnitudeFrame;
            if (magnitude(frame, magnitudeFrame) == 0)
            {
                frame = magnitudeFrame;
            }
        }

        // Task 9: Quantize filter
        if (lastKey == 'l')
        {

            cv::Mat quantizedFrame;
            if (blurQuantize(frame, quantizedFrame, 10) == 0)
            {
                frame = quantizedFrame;
            }
        }

        // Task 10: Face Detect filter
        if (lastKey == 'f')
        {
            cv::Mat greyFrame;
            cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
            if (detectFaces(greyFrame, faces) == 0)
            {
                drawBoxes(frame, faces, 50, 1.0); // assuming minimum width of face is 50 and scale is 1.0
            }
        }

        // Task 11 Additional Filters

        // Emboss filter
        if (lastKey == 'e')
        {
            cv::Mat embossedFrame;
            if (emboss(frame, embossedFrame) == 0)
            {
                frame = embossedFrame;
            }
        }

        // Negative filter
        if (lastKey == 'n')
        {
            cv::Mat negativeFrame;
            if (negativeImage(frame, negativeFrame) == 0)
            {
                frame = negativeFrame;
            }
        }

        // Color Pop filter
        if (lastKey == 'p')
        {
            cv::Mat colorPopFrame;
            cv::Scalar lowerb(110, 100, 100); // Lower bound for the strong color in the HSV color space
            cv::Scalar upperb(130, 255, 255); // Upper bound for the strong color in the HSV color space
            if (colorPop(frame, colorPopFrame, lowerb, upperb) == 0)
            {
                frame = colorPopFrame;
            }
        }

        // Cartoon filter
        if (lastKey == 'a')
        {
            cv::Mat cartoonFrame;
            if (cartoonize(frame, cartoonFrame) == 0)
            {
                frame = cartoonFrame;
            }
        }

        // Extensions : Additional Filters and Features

        // Extension 1: Sketch filter
        if (lastKey == 't')
        {
            cv::Mat sketchFrame;
            if (sketch(frame, sketchFrame) == 0)
            {
                frame = sketchFrame;
            }
        }

        // Display the name of the current filter on the frame
        if (filterNames.count(lastKey) > 0)
        {
            std::string filterName = filterNames[lastKey];
            cv::putText(frame, filterName, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        // Quit if 'q' is pressed
        char key = (char)cv::waitKey(30);

        if (key == 'q')
            break;

        // Extension 2: Increase Brightness and Contrast if '+' is pressed
        if (key == '+')
        {
            alpha += 0.1;
            beta += 10;

            cv::putText(frame, "Brightness and contrast increased", cv::Point(frame.cols / 2 - 200, frame.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        // Decrease Brightness and Contrast if '-' is pressed
        if (key == '-')
        {

            alpha -= 0.1;
            beta -= 10;

            cv::putText(frame, "Brightness and contrast decreased", cv::Point(frame.cols / 2 - 200, frame.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        // Apply brightness and contrast adjustments
        frame.convertTo(frame, -1, alpha, beta);

        // Extension 3: Start/Stop recording if 'r' is pressed
        if (key == 'r')
        {
            if (!isRecording)
            {
                // Start recording
                isRecording = true;
                videoWriter.open("E:/MSCS/CVPR/projects/project1/task2/data/output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame.cols, frame.rows));
                cv::putText(frame, "Video Recording: ON", cv::Point(frame.cols - 220, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            }
            else
            {
                // Stop recording
                isRecording = false;
                videoWriter.release();
            }
        }

        // Display recording status
        if (isRecording)
        {
            cv::putText(frame, "Video Recording: ON", cv::Point(frame.cols - 220, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            videoWriter.write(frame);
        }

        // Save the current frame as image if 's' is pressed
        if (key == 's')
        {
            std::string filename = "E:/MSCS/CVPR/projects/project1/task2/data/frame_" + std::to_string(frameCount++) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "Saved " << filename << std::endl;
        }

        // Display the resulting frame
        cv::imshow("Video", frame);

        // Update last key pressed
        if (key == 'g' || key == 'c' || key == 'h' || key == 'v' || key == 'b' || key == 'x' || key == 'y' || key == 'm' || key == 'l' || key == 'e' || key == 'n' || key == 'p' || key == 'a' || key == 'f' || key == 't')
        {
            lastKey = key;
        }
    }

    // At the end of program, release the video capture object
    cv::destroyAllWindows();
    return 0;
}
