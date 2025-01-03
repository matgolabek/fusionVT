#include "RgbtThread.h"
#include <iostream>
#include <sstream>
#include <QMessageBox>
#include <QString>
#include <QString>


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
	const std::string jsonFilePath = "/home/lsriw/Documents/Flir/Modified/software/raspberrypi_video/calibStereo.json";
	std::ifstream file(jsonFilePath);
	if(!file.is_open()) {
		throw std::runtime_error("Failed to open JSON file");
	}
	
	nlohmann::json jsonData;
	file >> jsonData;
	file.close();
	for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
	}
	loadMapFromJson(jsonData["map1_vis"], map1Vis);
	loadMapFromJson(jsonData["map2_vis"], map2Vis);
	loadMapFromJson(jsonData["map1_thermal"], map1Thermal);
	loadMapFromJson(jsonData["map2_thermal"], map2Thermal);
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
	QImage img1s = img1.scaled(640, 480, Qt::KeepAspectRatio, Qt::SmoothTransformation);
	
	cv::Mat imgVis(img2.height(), img2.width(), CV_8UC3, const_cast<uchar*>(img2.bits()), img2.bytesPerLine());
	cv::Mat imgThermal(img1s.height(), img1s.width(), CV_8UC3, const_cast<uchar*>(img1s.bits()), img1s.bytesPerLine());
	cv::Mat remappedVis;
	cv::Mat remappedThermal;
	
	cv::Mat map2;
	
	cv::remap(imgThermal, remappedThermal, map1Thermal, map2Thermal, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
	cv::remap(imgVis, remappedVis, map1Vis, map2Vis, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
	
	QImage imgVisQ(remappedVis.data, remappedVis.cols, remappedVis.rows, remappedVis.step, QImage::Format_RGB888);
	QImage imgThermalQ(remappedThermal.data, remappedThermal.cols, remappedThermal.rows, remappedThermal.step, QImage::Format_RGB888);
	
	imgVisQ.rgbSwapped();
	imgThermalQ.rgbSwapped();
	
	int pix_shift = 70;
	int x1 = 90, x2 = 450, y1 = 130, y2 = 360;
	
	QImage imgV = imgVisQ.copy(x1, y1, x2 - x1, y2 - y1);
	QImage imgT = imgThermalQ.copy(x1 + pix_shift, y1, x2 - x1, y2 - y1);
	
	QImage imgVs = imgV.scaled(640, 480, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
	QImage imgTs = imgT.scaled(640, 480, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
		
	QImage meanColor(imgVs.size(), imgVs.format());
	
    for (int y = 0; y < 480; ++y) {
        for (int x = 0; x < 640; ++x) {
            QColor color1 = imgVs.pixelColor(x, y);
            QColor color2 = imgTs.pixelColor(x, y);
            int red = (color1.red() + color2.red()) / 2;
            int green = (color1.green() + color2.green()) / 2;
            int blue = (color1.blue() + color2.blue()) / 2;
            meanColor.setPixelColor(x, y, QColor(red, green, blue));
        }
    }
    
    return meanColor;
}


void RGBTThread::takePhoto()
{
	std::time_t now = std::time(nullptr);
    std::stringstream ss;
    ss << "thermal_photo_" << now << ".jpg";  // Format: photo_<timestamp>.jpg
    std::string tfilename = ss.str();
    
    now = std::time(nullptr);
    ss.str("");
    ss << "vis_photo_" << now << ".jpg";  // Format: photo_<timestamp>.jpg
    std::string vfilename = ss.str();
    
    QString qtfilename = QString::fromStdString(tfilename);
    QString qvfilename = QString::fromStdString(vfilename);
    
    thermalImage.save(qtfilename);
    cameraImage.save(qvfilename);
/* Optional message boxes
    // Save the image to disk
    if (thermalImage.save(qtfilename)) {
        QMessageBox::information(nullptr, "Thermal Photo Taken", ("Photo saved as " + tfilename).c_str());
    } else {
        QMessageBox::warning(nullptr, "Error", "Failed to save the thermal photo.");
    }
    // Save the image to disk
    if (cameraImage.save(qvfilename)) {
        QMessageBox::information(nullptr, "Camera Photo Taken", ("Photo saved as " + vfilename).c_str());
    } else {
        QMessageBox::warning(nullptr, "Error", "Failed to save the camera photo.");
    }
*/
}


void RGBTThread::loadMapFromJson(const nlohmann::json& mapJson, cv::Mat& map){
	int rows = mapJson.size();
	int cols = mapJson[0].size();
	int channels = mapJson[0][0].size();
	
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (channels == 2) {
				map.at<cv::Vec2s>(i, j)[0] = mapJson[i][j][0];
				map.at<cv::Vec2s>(i, j)[1] = mapJson[i][j][1];
			} else {
				map.at<ushort>(i, j) = mapJson[i][j];
			}
		}
	}	
}

