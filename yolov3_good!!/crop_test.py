from PIL import Image

img = Image.open("123.png")
H = img.height
W = img.width

area = [(0, 0, W/2, H/2), (W/2, 0, W, H/2), (0, H/2, W/2, H), (W/2, H/2, W, H)]

for number in range(0,4):
    print(number)
    print(area[number])
    (img.crop(area[number])).show()
