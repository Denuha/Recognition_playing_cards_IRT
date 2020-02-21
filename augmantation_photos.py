import os
import cv2

path1 = "my_photos/"
path2 = "my_photos2/"
path3 = "my_photos3/"

res_path = "result_photos/"

x_train_cards = []
cards = os.listdir(path=path1)


i = 0
for card in cards:
    image = cv2.imread(path1 + card, cv2.IMREAD_COLOR)
    value = card.split('[')[1].split(']')[0]
    cv2.imwrite(res_path + "cards-[" + value + "]-" + str(i) + ".jpg",image)
    i+=1
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated_l = cv2.warpAffine(image, M, (w, h))
    cv2.imwrite(res_path + "cards-[" + value + "]-" + str(i) + ".jpg",rotated_l)
    i+=1

    M = cv2.getRotationMatrix2D(center, -15, 1.0)
    rotated_r = cv2.warpAffine(image, M, (w, h))
    cv2.imwrite(res_path + "cards-[" + value + "]-" + str(i) + ".jpg",rotated_r)
    i+=1
    print(i)

print("END")
