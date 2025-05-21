# %% 
import torchvision
import videoseal
import torch
from videoseal.evals.metrics import bit_accuracy
import numpy as np
import random
import matplotlib.pyplot as plt

def load_video(path):
    video = torchvision.io.read_video(path,pts_unit = "sec", output_format="TCHW")
    video = video[0] # Load video and normalize to [0, 1]
    print("Video shape:",video.shape)
    return video

def save_video(path,video):
    video = video.movedim(1,-1)
    video = torchvision.io.write_video(path,video,fps = 60,video_codec ="h264")



def compare_message(msg1,msg2):
    length = len(msg1)
    correct_count = 0
    for i in range(length):
        if msg1[i] ==  msg2[i]:
            correct_count+=1
    print(correct_count,"in",length,"bits are identical")
    return correct_count

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

# %% 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_path = "/cmlscratch/xyu054/videoseal/Gabee_10s_clip.mp4"
video = load_video(video_path)
#video = video[2000:2300]
video =  video/ 255.0
video.to(device)

# Load the model
model = videoseal.load("videoseal")
model.to(device)

original_message = model.extract_message(video, aggregation="avg")
print()
print("Original Message:")
original_message_print= original_message.tolist()[0]
msg = ''.join(str(i) for i in original_message_print)
print()
print()
print(msg)


original_message_flat = original_message.tolist()[0]
correct_count_list = []
for k in [1,2,4,8,16,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]:
    print("inverting", k, "bits:")
    message = random_invert_k_bit(original_message, k)
    
    outputs1 = model.embed(video, is_video=True,msgs = message) # this will embed a random msg
    video_w1 = outputs1["imgs_w"] # the counter_watermarked video

    msg_extracted1 = model.extract_message(video_w1, aggregation="avg")
    msg_extracted1 = msg_extracted1.tolist()[0]
    
    correct_count = compare_message(original_message_flat,msg_extracted1)
    correct_count_list.append(correct_count)
    #path = "/cmlscratch/xyu054/videoseal/counter_embedded/AAAAAA_saaa_clip_counter_embedded_"+str(k)+"bits.mp4"
    #save_video(path,video_w1*255.0)
#plt.plot([1,2,4,8,16,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95],correct_count_list)
#plt.savefig("bit_vs_accuracy")
'''


inverse_message = inverse_message(original_message)
outputs1 = model.embed(video, is_video=True,msgs = inverse_message) # this will embed a random msg
video_w1 = outputs1["imgs_w"] # the counter_watermarked video

msg_extracted1 = model.extract_message(video_w1, aggregation="avg")
print()
print("Message after counter embedding:")
msg_extracted1 = msg_extracted1.tolist()[0]
print("Extracted Message:",msg_extracted1)
original_message = original_message.tolist()[0]
compare_message(original_message,msg_extracted1)
save_video("/cmlscratch/xyu054/videoseal/AAAAAA_saaa_clip_counter_embedded.mp4",video_w1*255.0)


'''

'''
msg_extracted2 = model.extract_message(video, aggregation="avg")
print()
print("Message after second embedding:")
msg_extracted2 = msg_extracted2.tolist()[0]

print("Extracted Message:",msg_extracted2)

msg = ''.join(str(i) for i in msg_extracted2)
print(msg)
'''
# Video Watermarking
'''
msg1 = model.get_random_msg()
outputs1 = model.embed(video, is_video=True,msgs = msg1) # this will embed a random msg


video_w1 = outputs1["imgs_w"] # the watermarked video
'''


#msgs1 = outputs1["msgs"]  #the embedded message
'''
original_message = msg1.tolist()[0]
print("Original message:")
print(original_message)

'''
# Extract the watermark message

'''
msg_extracted1 = model.extract_message(video_w1, aggregation="avg")
print()
print("Message after first embedding:")
msg_extracted1 = msg_extracted1.tolist()[0]
print("Extracted Message:",msg_extracted1)

compare_message(original_message,msg_extracted1)
'''

"""
Second watermarking
"""

'''
msg2 = inverse_message(msg1)
outputs2 = model.embed(video_w1, is_video=True,msgs = msg2) # this will embed a random msg
video_w2 = outputs2["imgs_w"] # the watermarked video
'''
'''
print("video_w2 shape 1:",video_w2.shape)
watermark1 = video_w1 - video
watermark2 = video_w2 - video

video_w1 = video_w1.movedim(1,-1)
watermark1 = watermark1.movedim(1,-1)
print("video_w1 shape:",video_w1.shape)
save_video("/cmlscratch/xyu054/videoseal/Visualizations/watermarked_video_1.mp4",video_w1*255.0)
save_video("/cmlscratch/xyu054/videoseal/Visualizations/watermark_1.mp4",watermark1*255.0)

video_w2 = video_w2.movedim(1,-1)
watermark2 = watermark2.movedim(1,-1)
print("video_w2 shape 2:",video_w2.shape)
save_video("/cmlscratch/xyu054/videoseal/Visualizations/watermarked_video_2.mp4",video_w2*255.0)
save_video("/cmlscratch/xyu054/videoseal/Visualizations/watermark_2.mp4",watermark2*255.0)
'''
'''
original_message2 = msg2.tolist()[0]
print("Original message2:")
print(original_message2)
'''

# Extract the watermark message
'''
msg_extracted2 = model.extract_message(video_w2, aggregation="avg")
print()
print("Message after second embedding:")
msg_extracted2 = msg_extracted2.tolist()[0]

print("Extracted Message:",msg_extracted2)

compare_message(original_message,msg_extracted2)
'''