#include "CameraThread.h"
#include <QMutexLocker>
#include <QDebug>

CameraThread::CameraThread(QObject *parent) 
    : QThread(parent), running(false) {
    // Initialize OpenCV video capture
    cap.open(0); // Change this if the USB camera is not at index 0
    if (!cap.isOpened()) {
        qDebug() << "Could not open USB camera.";
    }
}

void CameraThread::run() {
    running = true;
    int width = 640;
    int height = 480;
    while (running) {
        cv::Mat frame;
        if (cap.read(frame)) {
            // Convert from BGR to RGB
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            cv::resize(frame, frame, cv::Size(width, height));
            //std::cout<<"Channels of frame:"<<frame.at<cv::Vec3b>(3, 6); - getting the matrix element
            // Convert to QImage
            QImage image(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
            emit updateVideoFrame(image); // Emit signal with the new frame
        }
        QThread::msleep(30); // Sleep to control frame rate
    }
}

void CameraThread::stop() {
    running = false;
    cap.release(); // Release the camera
    wait(); // Wait for the thread to finish
}
