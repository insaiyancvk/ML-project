from torchvision import transforms

import torch.nn as nn
import torch, warnings, sys, os, requests, argparse

warnings.filterwarnings("ignore")
from PIL import Image


parser = argparse.ArgumentParser(description='args')
grp = parser.add_mutually_exclusive_group()
grp.add_argument('-p','--path', type=str, help='Path to image from your PC', default=None)
grp.add_argument('-u', '--url', type=str, help='URL of the image', default=None)

args = parser.parse_args()

img = None

if args.path is not None:

    if os.path.isfile(args.path):
        img = Image.open(args.path)
    else:
        print(f'{args.path} does not exist. Please check the path')
        sys.exit()

if args.url is not None:

    req = requests.get(args.url, stream=True)
    
    if req.status_code == 200:
        img = Image.open(req.raw)
        
    else:
        print(f'Error HTTP {req.status_code}')
        sys.exit()


labels = [
    'Cat',
    'Dog'
]

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=1280, out_features = 2)
)
model.load_state_dict(torch.load('./assets/MobileNetV2CatsVdogsTEST.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if img is not None:

    img_processed = transform(img)
    img_processed = img_processed.unsqueeze(0)
    
    preds = nn.functional.softmax(model(img_processed)[0], dim=0)
    
    print({labels[i]: float(preds[i]) for i in range(len(labels))})