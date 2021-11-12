import gradio as gr
from torchvision import transforms
import torch.nn as nn
import torch, warnings
warnings.filterwarnings("ignore")
from PIL import Image

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
model.load_state_dict(torch.load('./assets/MobileNetV2CatsVdogsTEST.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    preds = nn.functional.softmax(model(img)[0], dim=0)
    return {labels[i]: float(preds[i]) for i in range(len(labels))}

title = "Cats Vs Dogs"
description = "Demo for cat-dog classifier. To use it, simply upload A picture, or click one of the examples from below to load them."

inputs = gr.inputs.Image()
outputs = gr.outputs.Label(num_top_classes=2)
gr.Interface(
    fn = predict, 
    inputs = inputs,
    outputs = outputs,
    title = title, 
    description = description, 
    allow_flagging = False,
    layout = 'horizontal',
    theme = 'compact',
    thumbnail = '.\\assets\\dog1.jpg',
    examples = [
        ['./assets/cat0.jpeg'],
        ['./assets/dog0.jpg']
        ]
    ).launch()