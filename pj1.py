import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.misc import derivative

'''안경쓴 눈, 그냥 눈 , 얼굴, 몸 ,상반신, 하반신, 웃는 입 등을 인식'''
face_cascade = cv2.CascadeClassifier(
    './data/haarcascades/haarcascade_mcs_leftear.xml')
image = cv2.imread('test12.JPG')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayImage, 1.03, 5)
blue_color = (255,0,0)
point = []
point = faces
x_point = point[0][0]
y_point = point[0][1]
fix_x = 80
fix_y = 176
Good = 1.1106
FairOrBad = 0.9325
gap = 27
print('귀의 x좌표 : %d,귀의 y좌표 : %d' %(x_point,y_point+gap))
print('가이드라인 point x좌표 : %d,귀의 y좌표 : %d' %(fix_x,fix_y))

cv2.resize(image, dsize=(480, 800), interpolation=cv2.INTER_AREA)
#목 후단부
image = cv2.line(image,(fix_x,fix_y),(fix_x,fix_y),blue_color,5)
#귀 위치
image = cv2.line(image,(x_point+20,y_point+gap),(x_point+20,y_point+gap),blue_color,5)
#목 후단부의 수평선
image = cv2.line(image,(0,fix_y),(480,fix_y),blue_color,1)
#두 점 사이의 선분
image = cv2.line(image,(x_point+20,y_point+gap),(fix_x,fix_y),blue_color,1)
#image = cv2.resize(image, (720, 960))
'''이미지를 회색조 모드로 로드'''

#figsize는 output 화면의 크기
plt.figure(figsize=(12,8))
plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

angleA_X, angleA_Y = (x_point+20,y_point+gap)
angleB_X, angleB_Y = (fix_x,fix_y)

deltaX = angleA_X - angleB_X
deltaY = angleB_Y - angleA_Y

tan = deltaY / deltaX

print(tan)

if  tan >= Good :
    print("GOOD")
elif tan < Good and tan >= FairOrBad :
    print("Fair")
else :
    print("Bad")

'''CascadeClassifier의 detectMultiScale 함수에 grayImage 이미지를 입력하여 얼굴을 검출한다. 얼굴이 검출되면
그 위치를 리스트로 리턴해줌. 위치는(x,y,w,h)와 같은 튜플이며 (x,y)는 검출된 얼굴의 좌상단 위치, w,h는 가로,세로크기
여기서 파라메터로 쓰인 5는 minNeighbors이고 1.03은 scalefactor
minNeighbors의 수를 작게해야 탐지가 더 잘됨 하지만 품질은 안좋음
'''

#print(faces.shape)
#print("Number of faces detected: " + str(faces.shape[0]))

#사각형 그리기
#파일명을 위한 imgNum 변수 초기화


for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.rectangle(image, ((0,image.shape[0] -25)),
              (270, image.shape[0]), (255,255,255), -1);
cv2.putText(image, "earDetection", (0,image.shape[0] -10),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1);

plt.figure(figsize=(12,12))
plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

imgNum = 0

'''
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.rectangle(image,(x+80,y+20),(x+w+240,y+h+150),(0,255,0),4)
    #크롭할 범위설정
    cropped = image[y-40 :y + h +210, x +70 :x + w + 230]
    #1번째는 아래에서부터 높이 ,2번째는 위에서부터 높이 아닌가? 3번째는 늘리면 더 ->이렇게 마지막은 줄이면 <-이렇게 폭줄임
    #크롭한 이미지를 저장, 즉 목만 크롭한 이미지 추출
    cv2.imwrite("neckdata" + str(imgNum) + ".JPG", cropped)
    image = cv2.imread("neckdata" + str(imgNum) + ".JPG")
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imgNum += 1

cv2.rectangle(image, ((0,image.shape[0] -25)),
              (270, image.shape[0]), (255,255,255), -1);
cv2.putText(image, "neck image extraction", (0,image.shape[0] -10),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1);

#목을 크롭해준다
cv2.imshow('Image view',cropped)
plt.figure(figsize=(12,12))
plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
#얼굴의 좌표를 리턴해줌

print(faces)
image2 = cv2.imread("neckdata0.JPG")
edges = cv2.Canny(image2,250,500,apertureSize = 3)
cv2.imshow('Edges',edges)

def difference_quotient(edges, x ,h):
    print((edges(x + h) - edges(x)) / h)
    return (edges(x + h) - edges(x)) / h


def square(x):
    return x * x

def derivative(x):
    return 2 * x
derivative_estimate = lambda x: difference_quotient(edges, x, h=0.00001)
x = range(-10,10)
plt.title("Actual Derivatives vs Estimate")
plt.plot(x, list(map(derivative, x)), 'rx', label='Actual')
plt.plot(x, list(map(derivative_estimate, x)), 'b+', label='Estimate')
plt.legend(loc=9)
plt.show()


'''
'''
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.rectangle(image, ((0,image.shape[0] -25)),
              (270, image.shape[0]), (255,255,255), -1);
cv2.putText(image, "PinkWink test", (0,image.shape[0] -10),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1);

plt.figure(figsize=(12,12))
plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
'''
'''
body_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_fullbody.xml')
body = body_cascade.detectMultiScale(grayImage, 1.03, 4)

for (x,y,w,h) in body:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)

plt.figure(figsize=(12,12))
plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()





edges = cv2.Canny(grayImage,250,500,apertureSize = 3)
cv2.imshow('Edges',edges)
lines = cv2.HoughLines(edges,1,np.pi/180,170)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('Lines',image)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

'''
