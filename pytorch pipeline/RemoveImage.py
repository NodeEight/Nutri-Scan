from PIL import Image
import os 



path = r"C:\Users\TechWatt\Desktop\nutriscan dataset\nourished"

unwantedExtention = 'webp'
for imglist in os.listdir(path):
    
    d = imglist.split('.')
    if d[1] == unwantedExtention:
        avipath = os.path.join(path,imglist)
        os.remove(avipath)
    
    
    else: print(os.path.join(path,imglist))
    