#!/usr/bin/python3

# http://rolandtapken.de/blog/2008-12/create-screenshots-web-page-using-python-and-qtwebkit

import sys
import signal

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebKitWidgets import QWebPage

url = "about:blank"
output = "/tmp/webshot.png"
width = 1600
height = 900
zoom_factor = 1.0
webpage = None
main_frame = None

def onLoadFinished(result):
    if not result:
        sys.exit(1)

    webpage.setViewportSize(QSize(width, height))
    main_frame.setZoomFactor(zoom_factor);

    image = QImage(webpage.viewportSize(), QImage.Format_ARGB32)
    painter = QPainter(image)
    webpage.mainFrame().render(painter)
    painter.end()
    image.save(output)
    sys.exit(0)

def loadPage():
    global webpage, main_frame

    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    webpage = QWebPage()
    webpage.loadFinished.connect(onLoadFinished)
    main_frame = webpage.mainFrame()
    main_frame.load(QUrl(url))

    sys.exit(app.exec_())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} url output.png [width height [zoom_factor]]")
        sys.exit(2)

    url = sys.argv[1]
    output = sys.argv[2]

    if len(sys.argv) >= 5:
        width = int(sys.argv[3])
        height = int(sys.argv[4])

    if len(sys.argv) >= 6:
        zoom_factor = float(sys.argv[5])

    loadPage()
