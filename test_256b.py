import os
from PIL import Image
import torch
import torchvision
from torchvision.transforms.functional import to_tensor, to_pil_image


def load_video(path):
    video = torchvision.io.read_video(path,pts_unit = "sec", output_format="TCHW")
    video = video[0] # Load video and normalize to [0, 1]
    print("Video shape:",video.shape)
    return video

def save_video(path,video):
    video = video.movedim(1,-1)
    video = torchvision.io.write_video(path,video,fps = 30,video_codec ="h264")

def compare_message(msg1,msg2):
    length = len(msg1)
    correct_count = 0
    for i in range(length):
        if msg1[i] ==  msg2[i]:
            correct_count+=1
    print(correct_count,"in",length,"bits are identical")

def inverse_message(msgs):
    #zero_tensor = torch.zeros_like(msgs)
    new_message = torch.where(msgs>0,0,1)
    return new_message

def random_invert_k_bit(ori_message, k):
    index = sample(range(len(ori_message)),k)
    for i in index:
        ori_message[i] = ori_message[i] ^ 1
'''

# Download the model and load it.
os.makedirs("ckpts", exist_ok=True)
if not os.path.exists("ckpts/y_256b_img.jit"):
    os.system("wget https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit -P ckpts/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("ckpts/y_256b_img.jit").to(device).eval()

video_path = "/cmlscratch/xyu054/videoseal/AAAAAA_saaa_clip.mp4"
video = load_video(video_path)
print(1)



video_watermarked = torchvision.io.read_video(video_path,pts_unit = "sec")[0].permute(0, 3, 1, 2)
video_watermarked = (video_watermarked.float() / 255.0).to(device)
preds = model.detect_video_and_aggregate(
        video_watermarked,
        aggregation="avg"  # Options: "avg", "squared_avg", "l1norm_avg", "l2norm_avg"
    )
preds = preds.detach().cpu().numpy()[0]
print()
print("Message extracted:")

print(preds)

# Video Watermarking
msg = ''.join(str(int(i)) for i in preds)
print(msg)
'''




'''
msg_extracted2 = model.extract_message(video, aggregation="avg")


msg_extracted2 = msg_extracted2.tolist()[0]
print("Extracted Message:",msg_extracted2)
print()
msg = ''.join(str(i) for i in msg_extracted2)
print(msg)
'''

os.makedirs("ckpts", exist_ok=True)
if not os.path.exists("ckpts/y_256b_img.jit"):
    os.system("wget https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit -P ckpts/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("ckpts/y_256b_img.jit").to(device).eval()

video_path = "/cmlscratch/xyu054/videoseal/dataset/clips/raw/toe/10.mp4"

video = torchvision.io.read_video(video_path,pts_unit = "sec")[0].permute(0, 3, 1, 2)  # TCHW format
video = (video.float() / 255.0).to(device)[:150]  # First 16 frames to avoid OOMs
msg = torch.randint(0, 2, (1, 256)).float().to(device)
video_watermarked = model.embed(video, msg, is_video=True)
watermark = video_watermarked*255 - video*255
#print(watermark[0])
watermark = torch.abs(watermark)
#print(watermark[0])
#watermark = torch.where(watermark== 0, 255,watermark  )
video_path = "/cmlscratch/xyu054/videoseal/video.mp4"
watermarked_video_path = "/cmlscratch/xyu054/videoseal/watermarked_video.mp4"
watermark_path = "/cmlscratch/xyu054/videoseal/watermark.mp4"
save_video(video_path,video*255)
save_video(watermarked_video_path,video_watermarked*255)
save_video(watermark_path,watermark*50)
