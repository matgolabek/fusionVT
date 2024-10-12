#include "MyLabel.h"

MyLabel::MyLabel(QWidget *parent) : QLabel(parent)
{
}
MyLabel::~MyLabel()
{
}

//when the system calls setImage, we'll set the label's pixmap
void MyLabel::setImage(QImage image) {
  QPixmap pixmap = QPixmap::fromImage(image);
  int w = this->width();
  int h = this->height();
  setPixmap(pixmap.scaled(w, h, Qt::KeepAspectRatio));
}


// Getter to retrieve the current image (as QImage)
QImage MyLabel::getImage() const {
    // Get the current pixmap and return it as a QImage
    if (this->pixmap()) {
        return this->pixmap()->toImage();
    }
    return QImage();  // Return an empty image if no pixmap is set
}
