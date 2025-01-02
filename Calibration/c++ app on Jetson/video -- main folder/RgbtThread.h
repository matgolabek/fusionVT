#ifndef RGBTTHREAD_H
#define RGBTTHREAD_H

#include <QThread>
#include <QImage>
#include <QMutex>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

// RGBTThread class to combine RGB and thermal images
class RGBTThread : public QThread {
    Q_OBJECT

public:
    RGBTThread(QObject *parent) : QThread(parent) {}
    ~RGBTThread();

    // Separate methods to set the thermal and camera images
    void setThermalImage(const QImage &image1);
    void setCameraImage(const QImage &image2);
    void takePhoto();

signals:
    void updateFusedImage(const QImage &fusedImage);

protected:
    void run() override;

private:
    QImage blendImages(const QImage &image1, const QImage &image2);
    void loadMapFromJson(const nlohmann::json& mapJson, cv::Mat& map);

    QImage thermalImage;
    QImage cameraImage;
    QMutex mutex;
    
    cv::Mat map1Vis;
    cv::Mat map2Vis;
    cv::Mat map1Thermal;
    cv::Mat map2Thermal;
    
};

#endif // RGBTTHREAD_H
