from pycocotools import mask as cocomask
import cv2

# the RLE input format for coco api.
a={'size':[375,1242],'counts':"aaU19];3M5J6L2M1O2N1O101O00000O2O0O1O1O010O001O1O100O1O1O100000O0100O1O100O10000000001O0000001N2O001O0O100O2O000000O10O1000O100O1O1O100O1O1O1000000000000000000000000000000O1000000O100O1001O001O001O000010O01O000100O001N2N3M3N2M4J6K4Laag;"}
b=cocomask.decode(a)
cv2.imshow('tt',b*255)
cv2.waitKey(0)