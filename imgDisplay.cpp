/*
    *imageDisplay.cpp
    *Created on: Jan 21, 2024

    * Team Member:
    Team Members:
    Name: Ravi Shankar Sankara Narayanan
    NUID: 001568628
    Name: Vishaq Jayakumar
    NUID: 002737793

    * Purpose: This program reads an image from the disk and displays it in a window.

*/

// Include the libraries
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    // Read the image file
    cv::Mat img = cv::imread("E:/MSCS/CVPR/projects/project1/task1/lena.jpeg", cv::IMREAD_COLOR);

    // Check if the image file has been correctly loaded
    if (!img.data)
    {
        // Print an error message if the image cannot be opened or found
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Create a window to display the image
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    // Show the image in the created window
    cv::imshow("Display Image", img);

    // Infinite loop to keep the program running until the user decides to quit
    while (true)
    {
        // Wait for a key press
        int key = cv::waitKey(0);
        // If the key pressed is 'q', break the loop
        if (key == 'q')
        {
            // Inform the user that the program will quit
            std::cout << "You pressed 'q'. The program will now quit." << std::endl;
            break;
        }
    }

    // Destroy all the created windows
    cv::destroyAllWindows();
    return 0;
}