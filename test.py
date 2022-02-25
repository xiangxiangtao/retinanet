import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw


print('Loading model..')
net = RetinaNet()
net.load_state_dict(torch.load('./weights/ckpt_follow_gmy1_epoch78.pth'))
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
img = Image.open('/home/ecust/txx/dataset/smoke/IR/real/IR_smoke_research_group/valid/image/41_0.png')

# print(img.mode)
# img=img.convert('RGB')
# print(img.mode)

w = h = 600
img = img.resize((w,h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x)

print('Decoding..')
encoder = DataEncoder()
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()
