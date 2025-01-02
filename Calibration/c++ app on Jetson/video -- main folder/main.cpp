#include <QApplication>
#include <QThread>
#include <QMutex>
#include <QMessageBox>
#include <QColor>
#include <QLabel>
#include <QtDebug>
#include <QString>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QMessageBox>
#include "LeptonThread.h"
#include "MyLabel.h"
#include "CameraThread.h"
#include "RgbtThread.h"

// Print usage instructions
void printUsage(char *cmd) {
    char *cmdname = basename(cmd);
    printf("Usage: %s [OPTION]...\n"
           " -h      display this help and exit\n"
           " -cm x   select colormap\n"
           "           1 : rainbow\n"
           "           2 : grayscale\n"
           "           3 : ironblack [default]\n"
           " -tl x   select type of Lepton\n"
           "           2 : Lepton 2.x [default]\n"
           "           3 : Lepton 3.x\n"
           " -ss x   SPI bus speed [MHz] (10 - 30)\n"
           "           20 : 20MHz [default]\n"
           " -min x  override minimum value for scaling (0 - 65535)\n"
           "           [default] automatic scaling range adjustment\n"
           "           e.g. -min 30000\n"
           " -max x  override maximum value for scaling (0 - 65535)\n"
           "           [default] automatic scaling range adjustment\n"
           "           e.g. -max 32000\n"
           " -d x    log level (0-255)\n",
           cmdname, cmdname);
    return;
}

int main(int argc, char **argv) {
    int typeColormap = 3; // colormap_ironblack
    int typeLepton = 3; // Lepton 3.x
    int spiSpeed = 20; // SPI bus speed 20MHz
    int rangeMin = -1; //
    int rangeMax = -1; //
    int loglevel = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            exit(0);
        }
        else if (strcmp(argv[i], "-d") == 0) {
            int val = 3;
            if ((i + 1 != argc) && (strncmp(argv[i + 1], "-", 1) != 0)) {
                val = std::atoi(argv[i + 1]);
                i++;
            }
            if (0 <= val) {
                loglevel = val & 0xFF;
            }
        }
        else if ((strcmp(argv[i], "-cm") == 0) && (i + 1 != argc)) {
            int val = std::atoi(argv[i + 1]);
            if ((val == 1) || (val == 2)) {
                typeColormap = val;
                i++;
            }
        }
        else if ((strcmp(argv[i], "-tl") == 0) && (i + 1 != argc)) {
            int val = std::atoi(argv[i + 1]);
            if (val == 3) {
                typeLepton = val;
                i++;
            }
        }
        else if ((strcmp(argv[i], "-ss") == 0) && (i + 1 != argc)) {
            int val = std::atoi(argv[i + 1]);
            if ((10 <= val) && (val <= 30)) {
                spiSpeed = val;
                i++;
            }
        }
        else if ((strcmp(argv[i], "-min") == 0) && (i + 1 != argc)) {
            int val = std::atoi(argv[i + 1]);
            if ((0 <= val) && (val <= 65535)) {
                rangeMin = val;
                i++;
            }
        }
        else if ((strcmp(argv[i], "-max") == 0) && (i + 1 != argc)) {
            int val = std::atoi(argv[i + 1]);
            if ((0 <= val) && (val <= 65535)) {
                rangeMax = val;
                i++;
            }
        }
    }

    // Create the application
    QApplication a(argc, argv);
    
    QWidget *myWidget = new QWidget;
    myWidget->setWindowTitle("Fusion of Vision and Thermal Imaging");

	QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *layout = new QHBoxLayout();
    QHBoxLayout *buttonsLayout = new QHBoxLayout();
    
    // Create a label for the thermal image
    MyLabel leptonLabel(myWidget);
    leptonLabel.setFixedSize(640, 480);
    layout->addWidget(&leptonLabel);

    // Create a label for the USB camera video feed
    QLabel *cameraLabel = new QLabel(myWidget);
    cameraLabel->setFixedSize(640, 480);
    layout->addWidget(cameraLabel);

    // Create threads for Lepton and USB camera
    LeptonThread *leptonThread = new LeptonThread();
    leptonThread->setLogLevel(loglevel);
    leptonThread->useColormap(typeColormap);
    leptonThread->useLepton(typeLepton);
    leptonThread->useSpiSpeedMhz(spiSpeed);
    leptonThread->setAutomaticScalingRange();
    if (0 <= rangeMin) leptonThread->useRangeMinValue(rangeMin);
    if (0 <= rangeMax) leptonThread->useRangeMaxValue(rangeMax);
    QObject::connect(leptonThread, SIGNAL(updateImage(QImage)), &leptonLabel, SLOT(setImage(QImage)));

    // Create camera thread
    CameraThread *cameraThread = new CameraThread();
    QObject::connect(cameraThread, SIGNAL(updateVideoFrame(QImage)), cameraLabel, SLOT(setPixmap(QPixmap::fromImage(QImage))));
    
    // Assuming you have a QLabel* label; to display the frame.
	QObject::connect(cameraThread, &CameraThread::updateVideoFrame, 
                     [cameraLabel](const QImage &frame) {
        cameraLabel->setPixmap(QPixmap::fromImage(frame));
    });
    
    QLabel *fusedImageLabel = new QLabel(myWidget);
    fusedImageLabel->setFixedSize(640, 480);
    layout->addWidget(fusedImageLabel);

    // Create the RGBT thread for fusing thermal and RGB images
    RGBTThread *rgbtThread = new RGBTThread(nullptr);

    // Connect the update signals from Lepton and Camera threads to the RGBT thread
    QObject::connect(leptonThread, &LeptonThread::updateImage, rgbtThread, &RGBTThread::setThermalImage);
    QObject::connect(cameraThread, &CameraThread::updateVideoFrame, rgbtThread, &RGBTThread::setCameraImage);

    
    // Connect the RGBT thread's update signal to the fused image label
    QObject::connect(rgbtThread, &RGBTThread::updateFusedImage,
                     fusedImageLabel, [fusedImageLabel](const QImage &fusedImage) {
        fusedImageLabel->setPixmap(QPixmap::fromImage(fusedImage));
    });

    // Create a button for taking a photo    
    QPushButton *photoButton = new QPushButton("Take a Photo");
    QObject::connect(photoButton, &QPushButton::clicked, rgbtThread, &RGBTThread::takePhoto);
    buttonsLayout->addWidget(photoButton);

    leptonThread->start();
    cameraThread->start();
    rgbtThread->start();
    
    mainLayout->addLayout(layout);
    mainLayout->addLayout(buttonsLayout);
    
    myWidget->setLayout(mainLayout);
    myWidget->show();

    return a.exec();
}
