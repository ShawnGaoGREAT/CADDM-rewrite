import cv2
import os
import numpy as np
from imutils import face_utils
import dlib
import json
from tqdm import tqdm
from glob import glob
import concurrent.futures
IMG_META_DICT = dict()

root = "/root/autodl-tmp/gh/FF++_c23/manipulated_sequences/"
save_root = "/root/autodl-tmp/gh/FF++_32frames_fast_2/"
PREDICTOR_PATH = "shape_predictor_81_face_landmarks.dat"



def preprocess_video(video_path, save_path, label ,face_detector, face_predictor):
    # save the video meta info here
    #video_dict = dict()
    #save_path就是没后缀的，保存该图片的地址
    #video_path带
    IMG_META_DICT_subset = dict()
    source_video_path = get_source_video(video_path)  #带mp4的源视频的地址

    source_save_path = get_source_img(save_path)    #不带后缀
    frames_num_same_with_source = True

    # prepare the save path
    os.makedirs(save_path, exist_ok=True)
    # read the video and prepare the sampled index
    cap_video = cv2.VideoCapture(video_path)
    

    frame_count_video = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = np.linspace(0, frame_count_video - 1, 10, endpoint=True, dtype=int)
    if label == 0:
        frames_num_same_with_source = True
        len_frame_idxs = len(frame_idxs)
        frame_idxs = np.linspace(0, frame_count_video - 1, len_frame_idxs * 4, endpoint=True, dtype=int)
        source_frame_idxs = frame_idxs
    
    else:
        cap_source_video = cv2.VideoCapture(source_video_path)
        frame_count_source_video = int(cap_source_video.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count_video == frame_count_source_video:
            frames_num_same_with_source = False
        else:
            frames_num_same_with_source = False
        source_frame_idxs_ = np.linspace(0, frame_count_source_video - 1, 10, endpoint=True, dtype=int)
        len_source_frame_idxs = len(source_frame_idxs_)
        source_frame_idxs = np.linspace(0, frame_count_source_video - 1, len_source_frame_idxs * 4, endpoint=True, dtype=int)
    #frame_idxs_more = np.linspace(0, frame_count_video - 1, len(frame_idxs) * 4, endpoint=True, dtype=int)
    #frame_idxs = np.arange(0 , frame_count_video - 1 , 15)
    

    # process each frame
    for cnt_frame in range(frame_count_video):
        ret, frame = cap_video.read()
        height, width = frame.shape[:-1]

        if not frames_num_same_with_source :
            if cnt_frame > frame_count_source_video:
                continue

        if not ret:
            tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(video_path)))
            continue

        if cnt_frame not in frame_idxs:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector(frame, 1)
        if len(faces) == 0:
            tqdm.write('No faces in {}:{}'.format(cnt_frame, os.path.basename(video_path)))
            continue
        landmarks = list()  # save the landmark
        size_list = list()  # save the size of the detected face
        for face_idx in range(len(faces)):
            landmark = face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_s = (x1 - x0) * (y1 - y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        # save the landmark with the biggest face
        landmarks = np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]
        # save the meta info of the video
        video_dict = dict()
        video_dict['landmark'] = landmarks.tolist()
        video_dict['label'] = label

        video_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}"

        if not frames_num_same_with_source:
            if cnt_frame not in source_frame_idxs:
                cap_source_video.set(cv2.CAP_PROP_POS_FRAMES, cnt_frame)
                ret_source, frame_source = cap_source_video.read()
                if not ret_source:
                    tqdm.write('source_Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(video_path)))
                    continue
                frame_source = cv2.cvtColor(frame_source, cv2.COLOR_RGB2BGR)
                source_image_need_save_path = f"{source_save_path}/frame_{cnt_frame}.png"
                os.makedirs(source_save_path, exist_ok=True)
                cv2.imwrite(source_image_need_save_path, frame_source)




        '''
        if frames_num_same_with_source:
            video_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}"
        else:
            source_cnt_frame = source_frame_idxs[np.where(frame_idxs == cnt_frame)[0][0]]
            video_dict['source_path'] = f"{source_save_path}/frame_{source_cnt_frame}"
        '''

        IMG_META_DICT_subset[f"{save_path}/frame_{cnt_frame}"] = video_dict
        '''
        print('-------debug↓---------------')
        print('当前文件')
        print(f"{save_path}/frame_{cnt_frame}")
        print('+++++++++++++++++++++++++++++')
        print('img_meta_dict的状态')
        print(IMG_META_DICT)
        print('-------debug↑---------------')
        '''

        # save one frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = f"{save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(image_path, frame)
    cap_video.release()
    return IMG_META_DICT_subset


def preprocess_video_old(video_path, save_path, label ,face_detector, face_predictor):
    # save the video meta info here
    #video_dict = dict()


    source_video_path = get_source_video(video_path)

    source_save_path = get_source_img(save_path)
    frames_num_same_with_source = True

    # prepare the save path
    os.makedirs(save_path, exist_ok=True)
    # read the video and prepare the sampled index
    cap_video = cv2.VideoCapture(video_path)
    

    frame_count_video = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))


    if label == 0:
        frames_num_same_with_source = True
    else:
        cap_source_video = cv2.VideoCapture(source_video_path)
        frame_count_source_video = int(cap_source_video.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count_video == frame_count_source_video:
            frames_num_same_with_source = True
        else:
            frames_num_same_with_source = False
        source_frame_idxs = np.linspace(0, frame_count_source_video - 1, 1, endpoint=True, dtype=int)

    frame_idxs = np.linspace(0, frame_count_video - 1, 1, endpoint=True, dtype=int)
    #frame_idxs = np.arange(0 , frame_count_video - 1 , 15)
    

    # process each frame
    for cnt_frame in range(frame_count_video):
        ret, frame = cap_video.read()
        height, width = frame.shape[:-1]
        if not ret:
            tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(video_path)))
            continue
        if cnt_frame not in frame_idxs:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector(frame, 1)
        if len(faces) == 0:
            tqdm.write('No faces in {}:{}'.format(cnt_frame, os.path.basename(video_path)))
            continue
        landmarks = list()  # save the landmark
        size_list = list()  # save the size of the detected face
        for face_idx in range(len(faces)):
            landmark = face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_s = (x1 - x0) * (y1 - y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        # save the landmark with the biggest face
        landmarks = np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]
        # save the meta info of the video
        video_dict = dict()
        video_dict['landmark'] = landmarks.tolist()
        video_dict['label'] = label

        if frames_num_same_with_source:
            video_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}"
        else:
            source_cnt_frame = source_frame_idxs[np.where(frame_idxs == cnt_frame)[0][0]]
            video_dict['source_path'] = f"{source_save_path}/frame_{source_cnt_frame}"


        IMG_META_DICT[f"{save_path}/frame_{cnt_frame}"] = video_dict
        '''
        print('-------debug↓---------------')
        print('当前文件')
        print(f"{save_path}/frame_{cnt_frame}")
        print('+++++++++++++++++++++++++++++')
        print('img_meta_dict的状态')
        print(IMG_META_DICT)
        print('-------debug↑---------------')
        '''

        # save one frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = f"{save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(image_path, frame)
    cap_video.release()
    return


def get_source_img(path):  #输入是不带.mp4的
    if 'Real' in path:
        return path
    else:
        manipulate_method = path.split('/')[-2]
        manipulate_video_name = path.split('/')[-1]
        source_video_name = manipulate_video_name.split('_')[0]

        source_path = path.replace(manipulate_method, 'Real')
        source_path = source_path.replace(manipulate_video_name, source_video_name)
        return source_path   #返回的不带.mp4
    


def get_source_video(path):
    if 'Real' in path:
        return path
    else:
        manipulate_method = path.split('/')[-4]
        path = path.replace(manipulate_method, 'Real')
        path = path.replace(".mp4", '')
        manipulate_video_name = path.split('/')[-1]
        source_video_name = manipulate_video_name.split('_')[0]
        path = path.replace(manipulate_video_name, source_video_name)

        path = path + '.mp4'
        return path





def get_two_source(path):  #输入视频的完整地址，返回两个原视频的地址
    if 'Real' in path:
        return path
    
    manipulate_method = path.split('/')[-4]
    path_ = path[:-4]
    source = path_.split('/')[-1]
    source_name_1, source_name_2 = source.split("_")

    source_path_1 = path_.replace(source,source_name_1) + '.mp4'
    source_path_2 = path_.replace(source,source_name_2) + '.mp4'
    source_path_1 = source_path_1.replace(manipulate_method, 'Real')
    source_path_2 = source_path_2.replace(manipulate_method, 'Real')
    return (source_path_1, source_path_2)

        

def count_frames():
    num = 0
    same_frames_sum = 0
    same1 = 0
    same2 = 0
    both_same = 0
    for r, _, filename in sorted(os.walk(root, followlinks=True)):
        if 'c23' in r and 'DeepFakeDetection' not in r and 'FaceShifter' not in r:
            
            for name in tqdm(sorted(filename), desc=f'Processing files in {r}', unit='file'):
            #for name in sorted(filename):
                video_path = os.path.join(r, name)
                if not isinstance(get_two_source(video_path), tuple):
                    continue

                #计算当前视频的总帧数
                cap_video = cv2.VideoCapture(video_path)
                frames_num = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
                
                
                if isinstance(get_two_source(video_path), tuple):
                    source_path_1, source_path_2 = get_two_source(video_path)
                    source1_cap_video = cv2.VideoCapture(source_path_1)
                    source1_frames_num = int(source1_cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
                    source2_cap_video = cv2.VideoCapture(source_path_2)
                    source2_frames_num = int(source2_cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
                    if source1_frames_num == frames_num or source2_frames_num == frames_num:
                        same_frames_sum += 1 
                    if source1_frames_num == frames_num:
                        same1 += 1
                    if source2_frames_num == frames_num:
                        print(video_path)
                        print(source_path_2)
                        print("-------------------------")
                        same2 += 1
                    if source1_frames_num == frames_num and source2_frames_num == frames_num:
                        both_same += 1


                num += 1
                

                #print(video_path)
                #print(source_path)
                

    print(f"总共有视频数量：{num}")
    print(f"帧数符合要求的：{same_frames_sum}")
    print(f"和第一个一样{same1},第二个一样{same2}，两个都一样{both_same}")



def get_save_path(root):
    manipulate_method = root.split('/')[-3]
    save_path = save_root + manipulate_method + '/'
    return save_path
    

def get_label(video_path):
    if 'Real'in video_path:
        return 0
    else:
        return 1


def main():
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    for r, _, filename in sorted(os.walk(root, followlinks=True)):
        if 'c23' in r and 'DeepFakeDetection' not in r:
            
            #for name in sorted(filename):
            for name in tqdm(sorted(filename), desc=f'Processing files in {r}', unit='file'):
                save_path = (get_save_path(r) + name)[:-4]    #不带后缀 
                video_path = os.path.join(r, name)

                preprocess_video(video_path, save_path, get_label(video_path) ,face_detector, face_predictor)
    
    with open(f"{save_root}ldm.json", 'w') as f:
        json.dump(IMG_META_DICT, f)


def new_main():
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
    video_files = []

    for r, _, filenames in sorted(os.walk(root, followlinks=True)):
        if 'c23' in r and 'DeepFakeDetection' not in r and 'videos' in r:
            for name in sorted(filenames):
                save_path = (get_save_path(r) + name)[:-4]  # 不带后缀
                video_path = os.path.join(r, name)
                video_files.append((video_path, save_path, get_label(video_path)))


    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = []
        for video_path, save_path, label in video_files:
            futures.append(
                executor.submit(preprocess_video, video_path, save_path, label, face_detector, face_predictor)
            )

        # 使用 tqdm 显示进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing videos'):
            aaa = future.result()
            IMG_META_DICT.update(aaa)


    # 保存最终的元数据字典
    with open(f"{save_root}ldm.json", 'w') as f:
        json.dump(IMG_META_DICT, f)


if __name__ == "__main__":
    new_main()