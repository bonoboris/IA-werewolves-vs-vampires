# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:15:26 2020

@author: simed
"""
import os
import cv2

def convert_img_to_vid(path):
    imgs = os.listdir(path)
    final_imgs = []
    for img in imgs:
        if img.endswith('toc2'):
            continue
        final_imgs.append(cv2.imread(f'{path}/{img}'))
    write_video(final_imgs, f'{path}/output.mp4')
def write_video(imgs, output_path):
    shape = (imgs[0].shape[1],imgs[0].shape[0])
    vw = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'), 1, shape)
    for image in imgs:
        vw.write(image)
    vw.release()
    
convert_img_to_vid('D:/Ecole/3A/IA/super-secret/werevamp/videos/15')    
