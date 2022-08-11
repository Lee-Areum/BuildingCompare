import cv2
#from matplotlib import pyplot as plt
#import numpy as np

class compareImg:
    def __init__(self):
        pass

    def showImage(self,img):    #이미지를 화면에 창으로 띄우는 코드
        if (img.shape[0] >= 1000 or img.shape[1] >= 1000):  # 이미지 크기가 너무 클때 화면 보다 커질 때가 있어서 크기 줄이는 코드
            img = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        #print(img.shape[0])     

        cv2.imshow('image', img)    #이미지를 띄움
        cv2.waitKey(0)  #이미지를 띄운 상태에서 정지
        cv2.destroyAllWindows() 

    def readImg(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        """
        if (img.shape[0] >= 1000 or img.shape[1] >= 1000):  # 이미지 크기가 너무 클때 화면 보다 커질 때가 있어서 크기 줄이는 코드
            img = cv2.resize(img, dsize=(0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
        """
        #color로 이미지를 가져옴
        if (img.shape[0] < 1000 or img.shape[1] < 1000):  # 이미지 크기가 너무 클때 화면 보다 커질 때가 있어서 크기 줄이는 코드
            img = cv2.resize(img, dsize=(0, 0), fx=4.5, fy=4.5, interpolation=cv2.INTER_LINEAR)
        elif (img.shape[0] < 700 or img.shape[1] < 700):  # 이미지 크기가 너무 클때 화면 보다 커질 때가 있어서 크기 줄이는 코드
            img = cv2.resize(img, dsize=(0, 0), fx=6, fy=6, interpolation=cv2.INTER_LINEAR)
        #self.showImage(img) #이미지를 창의로 띄움.
        # cv2.imwrite("content/save_image.jpg", img)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        img = cv2.equalizeHist(img)
        return img

    def checking(self, count):  #이미지가 일치 하는지 안하는지 비교
        print("count good dot : ", count)
        if count >= 15:  # 일정 거리 이상인 점을 count 해서 임계값 보다 높으면 일치함.
            print("일치합니다.")
        else:
            print("일치하지 않습니다.")

    def diffImg(self, img1, img2):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        lowe_ratio = 0.75  #임계값(거리)
        # find the keypoints and descriptors with orb
        kp1, des1 = orb.detectAndCompute(img1, None)    #img1 특징점 추출
        kp2, des2 = orb.detectAndCompute(img2, None)    #img2 특징점 추출

        bf = cv2.BFMatcher()    #특징점 매칭클래스 호출
        matches = bf.knnMatch(des1, des2, k=2)  #특징점 매칭

        #Apply ratio test
        good = []   #정확도 높은 특징점 모을 배열 생성

        for m, n in matches:    #모든 점의 매칭을 돌면서
            if m.distance < lowe_ratio * n.distance:    #임계치보다 거리가 가까운 것을 good에 추가
                good.append([m])

        count = (len(good)) #good의 개수 확인
        print("matches: ", len(matches), "count: ", count)  #원래 특징점과 good 개수 출력

        knn_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)   #좋은 성능의 매칭점 그림

        self.checking(count)
        """
        아래 코드는 연결은 더 잘되지만 일치 불일치를 알 수 없어서 주석 처리함
        나중에 연결되는 그림을 보여주고 싶으면 아래 코드 사용하면 될 듯
        """        

        bf = cv2.BFMatcher()
        matches = bf.match(des1, des2)
    
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        #print(sorted_matches)
        knn_image = cv2.drawMatches(img1, kp1, img2, kp2, sorted_matches[:50], None, flags=2)


        #Draw first 10 matches.
        knn_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

        self.showImage(knn_image)   #매칭 된 그림을 화면에 띄움
        cv2.imwrite("content/matching_image.jpg", knn_image)    #띄운 그림을 저장함
        


    def run(self):
        # 이미지 파일 경로 설정
        filepath1 = "content/gong1.jpg"
        filepath2 = "content/gong2.jpg"

        # 이미지 객체 가져옴
        img1 = self.readImg(filepath1)  # 1번째 사진 프린트
        img2 = self.readImg(filepath2)  # 2번째 사진 프린트

        # 2개의 이미지 비교
        self.diffImg(img1, img2)

if __name__ == '__main__':
    cImg = compareImg()  # 클래스 생성
    cImg.run()  # 실행

