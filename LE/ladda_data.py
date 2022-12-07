# nödvändiga importer
import pandas as pd
import torch
from imageio import v3 as iio
import random
import dsntnn


def prepare_all_data(precent_of_videos=1):
    gestures = [
        'letter_A',
        'letter_B',
        'letter_C',
        'letter_L',
        'letter_R',
        'letter_U'
    ]
    train_data = []
    test_data = []
    for gesture in gestures:
        train_for_gesture, test_for_gesture = get_one_letter(
            gesture, precent_of_videos)
        train_data.extend(train_for_gesture)
        test_data.extend(test_for_gesture)
    random.Random(2).shuffle(train_data)
    random.Random(2).shuffle(test_data)
    return train_data, test_data


def get_one_letter(letter, precent_of_videos):
    data_all_videos = []
    path = f'../dataset_v0/ASL_{letter}'
    CSV = pd.read_csv(f'{path}/annotations.csv')
    num_of_videos = CSV['video_idx'].nunique()
    num_of_videos_we_want = round(precent_of_videos*num_of_videos)
    for video_index in range(num_of_videos_we_want):
        one_video = iio.imread(f'{path}/videos/video_{video_index}.mp4')
        if one_video.shape[1] == 480 or one_video.shape[2] == 640:
            one_video = iio.imread(f'{path}/videos/video_{video_index}.mp4')
            one_video_csv = CSV[CSV['video_idx'] == video_index]
            pairs_from_one_video = get_one_video(one_video_csv, one_video)
            data_all_videos.append(pairs_from_one_video)
    return shuffle_and_split(data_all_videos)


def shuffle_and_split(letter_data):
    train_size = round(0.8 * len(letter_data))
    train_set = letter_data[0:train_size]
    test_set = letter_data[train_size:len(letter_data)]
    return train_set, test_set


def get_one_video(one_video_csv, one_video):
    x_y_pairs = []
    num_of_frames = one_video_csv['frame'].nunique()
    for frame_index in range(num_of_frames-1):
        one_frame_csv = one_video_csv[one_video_csv['frame'] == frame_index]
        one_frame_csv = one_frame_csv[one_frame_csv['joint']
                                      != 'hand_position']
        this_frame = one_video[frame_index]
        this_frame_tensor = torch.tensor(this_frame).float()
        this_frame_tensor = torch.reshape(this_frame_tensor, (3, 640, 480))
        csv_one_frame = get_data_from_one_frame(one_frame_csv)
        x_y_pairs.append([this_frame_tensor, csv_one_frame])
        random.Random(2).shuffle(x_y_pairs)
    return x_y_pairs


def get_data_from_one_frame(one_frame_csv):
    cord_list = []
    for index, row in one_frame_csv.iterrows():
        cord=dsntnn.pixel_to_normalized_coordinates(torch.tensor([row['x'],row['y']]),[480,640])
        cord_list.append(torch.Tensor.tolist(cord))
    cord_tensor = torch.tensor(cord_list)
    return cord_tensor
