#ifndef CAMERATHREAD_H
#define CAMERATHREAD_H

#include <QThread>
#include <QImage>
#include <opencv2/opencv.hpp>

class CameraThread : public QThread {
    Q_OBJECT

public:
    CameraThread(QObject *parent = nullptr);
    void run() override;
    void stop();

signals:
    void updateVideoFrame(const QImage &frame);

private:
    bool running;
    cv::VideoCapture cap; // OpenCV video capture object
};

#endif // CAMERATHREAD_H
