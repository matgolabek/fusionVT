#include "RgbtThread.h"
#include <iostream>

RGBTThread::RGBTThread(QObject *parent)
    : QThread(parent), thermalImage(), cameraImage() {
}

RGBTThread::~RGBTThread() {
    this->quit();
    this->wait();
}

// Set the thermal image (thread-safe with a mutex)
void RGBTThread::setThermalImage(const QImage &img1) {
    QMutexLocker locker(&mutex);
    thermalImage = img1;
}

// Set the camera image (thread-safe with a mutex)
void RGBTThread::setCameraImage(const QImage &img2) {
    QMutexLocker locker(&mutex);
    cameraImage = img2;
}

// Thread's run function where the fusion happens
void RGBTThread::run() {
    while (true) {
        QMutexLocker locker(&mutex);
        if (!thermalImage.isNull() && !cameraImage.isNull()) {
            QImage fusedImage = blendImages(thermalImage, cameraImage);
            emit updateFusedImage(fusedImage);
        }
        locker.unlock();
        QThread::msleep(100);  // Adjust interval as needed
    }
}

// Basic blending function
QImage RGBTThread::blendImages(const QImage &img1, const QImage &img2) {
    QImage result(img2.size(), img2.format());
    //std::cout<<"img1 size: "<<img1.size().width()<<img1.size().height()<<", img2 size: "<<img2.size().width()<<img2.size().height()<<std::endl;
	QImage img1s = img1.scaled(640, 480, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    for (int y = 0; y < img2.height(); ++y) {
        for (int x = 0; x < img2.width(); ++x) {
            QColor color1 = img1s.pixelColor(x, y);
            QColor color2 = img2.pixelColor(x, y);
            int red = (color1.red() + color2.red()) / 2;
            int green = (color1.green() + color2.green()) / 2;
            int blue = (color1.blue() + color2.blue()) / 2;
            result.setPixelColor(x, y, QColor(red, green, blue));
        }
    }

    return result;
}
