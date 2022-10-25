import numpy as np


dict = {"vk_1080p": 5, "vk_720p": 4, "vk_540p": 3, "vk_360p": 2, "vk_270p": 1, "vk_180p": 0}
video_chunk_duration = 2
video_chunk_len = video_chunk_duration * 30  # 30FPS

for task in dict.keys():
    frame_size_path = "/home/zyhe/Documents/Mobile_gaming/Video_conceal/video_data/" + task + "_frame_size.txt"
    chunk_size = 0
    count = 1
    f1 = open(frame_size_path, "r")
    f2 = open("video_size/video_size_" + str(dict[task]) + ".txt", "w")
    for each_size in f1.readlines():
        chunk_size += int(each_size)
        count += 1
        if count % video_chunk_len == 0:
            f2.write(str(chunk_size))
            f2.write("\n")
            chunk_size = 0

    f1.close()
    f2.close()