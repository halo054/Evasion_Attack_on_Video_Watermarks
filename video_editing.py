# %% 
import json
import torchvision
import videoseal
import torch
from videoseal.evals.metrics import bit_accuracy
import numpy as np
import random
import matplotlib.pyplot as plt
import os 

def load_video(path):
    video = torchvision.io.read_video(path,pts_unit = "sec", output_format="TCHW")
    video = video[0] # Load video and normalize to [0, 1]
    print("Video shape:",video.shape)
    return video

def save_video(path,video,FPS = 30):
    video = video.movedim(1,-1)
    #video = video*255.0
    torchvision.io.write_video(path,video,fps = FPS,video_codec ="h264")

def inverse_message(msgs):
    #zero_tensor = torch.zeros_like(msgs)
    new_message = torch.where(msgs>0,0,1)
    return new_message

def random_invert_k_bit(ori_message, k):
    new_message = torch.clone(ori_message)
    index = random.sample(range(len(ori_message[0])),k)
    print(index)
    for i in index:
        new_message[0][i] = new_message[0][i] ^ 1
    return new_message
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model
model = videoseal.load("videoseal")
model.to(device)

dir_path = "/cmlscratch/xyu054/videoseal/dataset/clips/web_wmbedded/AAAAAA/toe/success_toe"
video_name_list = os.listdir(dir_path)

result = {}

for video_name in video_name_list:
    current_name = video_name[:-4]
    video_path = dir_path + "/" + video_name
    print(video_path)
    video = load_video(video_path)
    video =  video/ 255.0
    video.to(device)
    extracted_message = model.extract_message(video, aggregation="avg")
    extracted_message_print= extracted_message.tolist()[0]
    msg = ''.join(str(i) for i in extracted_message_print)

    result[current_name] = msg

print(result)

result_in_json = json.dumps(result)
f = open("/cmlscratch/xyu054/videoseal/dataset/clips/web_wmbedded/AAAAAA/toe/result.json","w")
f.write(result_in_json)
f.close()
'''
#video = video[2000:2300]
'''
os.makedirs("ckpts", exist_ok=True)
if not os.path.exists("ckpts/y_256b_img.jit"):
    os.system("wget https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit -P ckpts/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("ckpts/y_256b_img.jit").to(device).eval()
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model
model = videoseal.load("videoseal")
model.to(device)

dir_path = "/cmlscratch/xyu054/videoseal/dataset/clips/web_wmbedded/AAAAAA/saaa/success_saaa"
video_name_list = os.listdir(dir_path)

#result = {}

for video_name in video_name_list:
    #current_name = video_name[:-4]
    current_name = video_name[:-4]
    video_path = dir_path + "/" + video_name
    print(video_path)
    video = load_video(video_path)
    video =  video/ 255.0
    video.to(device)
    extracted_message = model.extract_message(video, aggregation="avg")
    extracted_message_print= extracted_message.tolist()[0]
    #msg = ''.join(str(i) for i in extracted_message_print)
    #print(preds)
    #msg = ''.join(str(int(i)) for i in preds)
    #result[current_name] = msg
    inverted_message = inverse_message(extracted_message)
    print(inverted_message.shape)
    outputs = model.embed(video, is_video=True,msgs = inverted_message) # this will embed a random msg
    video_w = outputs["imgs_w"] # the counter_watermarked video

    store_path = "/cmlscratch/xyu054/videoseal/dataset/clips/Overwriting_attacked/AAAAAA/saaa/inverse_" + video_name
    save_video(store_path,video_w*255.0)
    '''
    video_w1 = outputs1["imgs_w"]
    store_path = "/cmlscratch/xyu054/videoseal/dataset/clips/Overwriting_attacked/AAAAAA/saaa/" + video_name
    save_video("/cmlscratch/xyu054/videoseal/Visualizations/watermarked_video_2.mp4",video_w2*255.0)
    '''
#print(result)
'''
result_in_json = json.dumps(result)
f = open("/cmlscratch/xyu054/videoseal/dataset/clips/web_wmbedded/AAAAAA/saaa/result.json","w")
f.write(result_in_json)
f.close()
'''




