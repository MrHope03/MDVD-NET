import glob, os
import subprocess
import cv2

l = glob.glob("frames/*")
k = os.listdir("frames/")
print(l, k)
file = open("frames_count.txt", "w")
for i in range(len(l)):
    filename = f"E:/FYP/videos/{k[i]}.mp4"
    # res = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
    #                      "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filename],
    #                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    dur = cv2.VideoCapture(filename).get(cv2.CAP_PROP_FRAME_COUNT)
    m = glob.glob(l[i]+"/*")
    text = f"{k[i]} - {len(m)} - {dur}\n"
    file.write(text)
file.close()
