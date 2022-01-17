#Video Linki -> https://youtu.be/e4pTDg3Zb3I
 
#Kütüphanalerimizi ekledik
import cv2
import numpy as np 

#Resimlerimizi aldık
base_image = cv2.imread('base_img.jpg')
base_image_copy = base_image.copy()
subject_image = cv2.imread('subject.jpg')

#Resmimizi açıp köşe noktalarına tıklayarak köşe koordinatlarını alıp kaydediyoruz
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(base_image_copy, (x, y), 4, (0, 0, 255), -1)
        points.append([x, y])
        if len(points) <= 4:
            cv2.imshow('Ana Resim', base_image_copy)
points = []
base_image = cv2.imread('base_img.jpg')
base_image_copy = base_image.copy()
subject_image = cv2.imread('subject.jpg')

cv2.imshow('Ana Resim', base_image_copy)
cv2.setMouseCallback('Ana Resim', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Seçtiğimiz noktaları sıralama Sol üst noktadan saat yönüne doğru
def sort_pts(points):
    sorted_pts = np.zeros((4, 2), dtype="float32")
    s = np.sum(points, axis=1)
    sorted_pts[0] = points[np.argmin(s)]
    sorted_pts[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    sorted_pts[1] = points[np.argmin(diff)]
    sorted_pts[3] = points[np.argmax(diff)]

    return sorted_pts

sorted_pts = sort_pts(points)

#kordinatları ve sıralanmış kordinatları pts lere akaratdık
h_base, w_base, c_base = base_image.shape
h_subject, w_subject = subject_image.shape[:2]

pts1 = np.float32([[0, 0], [w_subject, 0], [w_subject, h_subject],                     [0, h_subject]])
pts2 = np.float32(sorted_pts)

#Seçtiğimiz noktalar arası resmi koyup gösteriyor
transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)

warped_img = cv2.warpPerspective(subject_image, transformation_matrix, (w_base, h_base))
cv2.imshow('Kirpilmis Resim', warped_img)

#Maskeliyor
mask = np.zeros(base_image.shape, dtype=np.uint8)

#Köşeler arasını beyaza boyuyor
roi_corners = np.int32(sorted_pts)
cv2.fillConvexPoly(mask, roi_corners, (255, 255, 255))

#Boyanan rengi tersine çeviriyor
mask = cv2.bitwise_not(mask)

#Maskelenmiş resmi aktarıyoruz
masked_image = cv2.bitwise_and(base_image, mask)

#Son resmi açıyor ve resmin çıktısını alıyoruz.
output = cv2.bitwise_or(warped_img, masked_image)
cv2.imshow('Final', output)
cv2.imwrite('Final_Resim.png', output)
cv2.waitKey(0)
cv2.destroyAllWindows()