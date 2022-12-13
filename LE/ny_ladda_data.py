import random 
from imageio import v3 as iio
import torch
import pandas as pd
import torchvision


def ladda(typ, seed, validation_round, data_divided_into, data_pass):
    all_data_list = []
    if typ == 'test':
        video_mod = [0]
    elif typ == 'validering':
        video_mod = [validation_round]
    else:
        video_mod = [1,2,3,4]
        video_mod.remove(validation_round)
        
    gestures = [
        'letter_A',
        'letter_B',
        'letter_C',
        'letter_L',
        'letter_R',
        'letter_U'
    ]
    for gest in gestures:
        data = one_letter(gest, data_divided_into, data_pass, video_mod)
        all_data_list.extend(data)
    random.Random(seed).shuffle(all_data_list)
    return all_data_list

def one_letter(gest, data_divided_into, data_pass, video_mod):
    resize = torchvision.transforms.Resize((640,480))
    all_data = []
    path = f'../dataset_v0/ASL_{gest}'
    CSV = pd.read_csv(f'{path}/annotations.csv')
    num_of_videos = CSV['video_idx'].nunique()
    for video_i in range(num_of_videos):
        if video_i%5 in video_mod:
            needs_scaling = False
            video = iio.imread(f'{path}/videos/video_{video_i}.mp4')
            one_video = torch.tensor(iio.imread(f'{path}/videos/video_{video_i}.mp4'))
            one_video = torch.reshape(one_video, 
                            (one_video.shape[0],one_video.shape[3],
                            one_video.shape[2],one_video.shape[1]))
             
            if one_video.shape[3] != 480 or one_video.shape[2] != 640:  
                x_old = one_video.shape[2]
                y_old = one_video.shape[3]
                one_video = resize(one_video)
                needs_scaling = True    
            one_video_csv = CSV[CSV['video_idx'] == video_i]
            x_y_one_video = get_one_video(one_video_csv, one_video,data_divided_into,data_pass)
            if needs_scaling:
                for frame_i in range(len(x_y_one_video)):
                    for joint_i in range(len(x_y_one_video[0][1])):
                        x_y_one_video[frame_i][1][joint_i][0] = x_y_one_video[frame_i][1][joint_i][0]*(640/x_old)
                        x_y_one_video[frame_i][1][joint_i][1] = x_y_one_video[frame_i][1][joint_i][1]*(480/y_old)
            all_data.extend(x_y_one_video)
    return all_data



def get_one_video(one_video_csv, one_video, data_divided_into, data_pass):
    x_y_pairs = []
    num_of_frames = one_video_csv['frame'].nunique()
    for frame_index in range(num_of_frames-1):
        if frame_index%data_divided_into == data_pass:
            one_frame_csv = one_video_csv[one_video_csv['frame']
                                            == frame_index]
            one_frame_csv = one_frame_csv[one_frame_csv['joint']
                                            != 'hand_position']
            this_frame = one_video[frame_index].float()
            csv_one_frame = get_data_from_one_frame(one_frame_csv)
            x_y_pairs.append([this_frame, csv_one_frame])
    return x_y_pairs

def get_data_from_one_frame(one_frame_csv):
    cord_list = []
    for index, row in one_frame_csv.iterrows():
        cord_one_point = []
        cord_one_point.append(row['y'])
        cord_one_point.append(row['x'])
        cord_list.append(cord_one_point)    
    return torch.tensor(cord_list)



