import glob
import cv2
import os
import pandas as pd
import pytube

def videos_to_frames(class_):
    frame_skips = 3
    max_frames = 5.5
    files = glob.glob('./'+class_+'/*.mp4')
    vid_count = 1

    for file in files:
        frame_count = 1
        video = cv2.VideoCapture(file)
        while True:
            try:
                os.mkdir('./' + class_ + "/" + "video" + str(vid_count))
                break
            except:
                vid_count += 1

        dir_ ='./' + class_ + "/" + "video" + str(vid_count) + '/frame'
        while video.isOpened():
            for _ in range(frame_skips):
                ok,frame = video.read()
                if not ok:
                    break
            if not ok:
                break
            cv2.imwrite(dir_+str(frame_count)+'.jpg',frame)
            frame_count += 1
            if frame_count == max_frames:
                break
        video.release()


def video_download(video_url,file_name,path):
    try:
        video = pytube.YouTube(video_url)
        stream = video.streams.get_highest_resolution()
        stream.download(output_path=path,filename=file_name)
        print(f"Downloaded {video.title}")
        return video.title

    except Exception as e:
        print(f"Error downloading {video_url}: {e}")
        return None

def download_from_txt(file):
    fil = open(file,'r')
    vals = []
    while True:
        try:
            os.mkdir('./videos' )
            break
        except:
            break
            pass
    vid_count = 1
    for line in fil.readlines():
        try:
            fil_name = 'video'+str(vid_count)+'.mp4'
            url = line.split()[0]
            start = int(line.split()[1].split('-')[0])
            end = int(line.split()[1].split('-')[1])
            file_title = video_download(url,fil_name,'./videos/')
            if file_title != None:
                vals.append(['./videos/',fil_name,start,end,file_title,url])
            vid_count += 1

        except Exception as e:
            print('error in file read',e,line)

    df = pd.DataFrame(vals, columns=['path','file_name', 'start','end','video_title','url'])
    df.to_csv('./videos/videos.csv')
    return df

def frames_from_csv(file=None,df=None):
    if file is not None:
        df = pd.read_csv(file)
    for index, row in df.iterrows():
        path = row['path']
        file_name = row['file_name']
        st = row['start']
        en = row['end']
        video_to_frames(file_name,path,st,en)


def video_to_frames(file,path,st,en):
    frame_skips = 1
    frame_count = 1
    video = cv2.VideoCapture(path+file)
    fps = video.get(cv2.CAP_PROP_FPS) if video.get(cv2.CAP_PROP_FPS) > 20.0 else 23.9
    start_frame = int(fps * st) - 10 if int(fps * st) - 10 > 0 else 0
    end_frame = int(fps * en) + 20

    try:
        os.mkdir(path + file[:-4])
    except Exception as e:
        print('error',e)


    dir_ = path + file[:-4] + '/frame'
    while video.isOpened():
        for _ in range(frame_skips):
            ok,frame = video.read()
            frame_count += 1
            if not ok:
                break
        if not ok:
            break
        if frame_count >= start_frame and frame_count < end_frame:
            cv2.imwrite(dir_+str(frame_count)+'.jpg',frame)
        # else:
        #     cv2.imwrite(dir_ + str(frame_count) + '_CROP.jpg', frame)
    video.release()

# to extract frames from the different classes and videos in directories

# classes = ['left_laying_leg_raise']
# for class_ in classes:
#     videos_to_frames(class_)

# to extract frames according to timestamp
# video_to_frames('How to do a Leg Curl_ Health e-University.mp4','./',50,55)

#to download videos and save timestamps in csv
df = download_from_txt('videos.txt')
print('Extracting frames')
frames_from_csv(df=df)

# to only extract frames from a previosly generated csv
# frames_from_csv(file='./videos/videos.csv')