import os
import lib.utils.file as FU

def download_youtube(key, path):

    # pip install youtube-dl

    url = "https://www.youtube.com/watch?v={}".format(key)
    
    # quality = 'worst[ext=mp4]'
    quality = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]'

    command = ['youtube-dl',
                   '-o', "{}/{}.mp4".format(path,key),
                   '-f', quality,
                   '-i', '-q',
                   url]

    subprocess.call(command)

    return "{}/{}.mp4".format(path,key)

def get_fps(path):

    import cv2

    cap= cv2.VideoCapture(path)
    return cap.get(cv2.CAP_PROP_FPS)

def v2i(vid_file, img_folder):

    command = ['ffmpeg',
                   '-i', vid_file,
                   '-f', 'image2',
                   '-v', 'error',
                   f'{img_folder}/%06d.jpg']

    if os.path.exists(vid_file):
        FU.checkdir(img_folder)
        subprocess.call(command)
