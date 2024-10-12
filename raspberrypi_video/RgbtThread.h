#ifndef RGBTTHREAD_H
#define RGBTTHREAD_H

#include <QThread>
#include <QImage>
#include <QMutex>

// RGBTThread class to combine RGB and thermal images
class RGBTThread : public QThread {
    Q_OBJECT

public:
    RGBTThread(QObject *parent = nullptr);
    ~RGBTThread();

    // Separate methods to set the thermal and camera images
    void setThermalImage(const QImage &image1);
    void setCameraImage(const QImage &image2);

signals:
    void updateFusedImage(const QImage &fusedImage);

protected:
    void run() override;

private:
    QImage blendImages(const QImage &image1, const QImage &image2);

    QImage thermalImage;
    QImage cameraImage;
    QMutex mutex;
};

#endif // RGBTTHREAD_H
