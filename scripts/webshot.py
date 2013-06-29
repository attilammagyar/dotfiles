#!/usr/bin/python

# Dependencies: libqt4-webkit python-qt4
# http://rolandtapken.de/blog/2008-12/create-screenshots-web-page-using-python-and-qtwebkit

import sys
import signal

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *

url = "about:blank"
output = "/tmp/webshot.png"
width = 1600
height = 900
webpage = None

def onLoadFinished(result):
    if not result:
        sys.exit(1)

    webpage.setViewportSize(QSize(width, height))

    image = QImage(webpage.viewportSize(), QImage.Format_ARGB32)
    painter = QPainter(image)
    webpage.mainFrame().render(painter)
    painter.end()
    image.save(output)
    sys.exit(0)

def loadPage():
    global webpage

    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    webpage = QWebPage()
    webpage.connect(webpage, SIGNAL("loadFinished(bool)"), onLoadFinished)
    webpage.mainFrame().load(QUrl(url))

    sys.exit(app.exec_())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: %s url output.png [width height]" % sys.argv[0]
        sys.exit(2)

    url = sys.argv[1]
    output = sys.argv[2]

    if len(sys.argv) == 5:
        width = int(sys.argv[3])
        height = int(sys.argv[4])

    loadPage()
