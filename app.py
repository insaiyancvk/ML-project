import gradio as gr
from torchvision import transforms
from torchvision.models.resnet import resnet50
import torch.nn as nn
import torch, warnings
warnings.filterwarnings("ignore")
from PIL import Image

labels = [
    'Cat',
    'Dog'
]

resnet = resnet50(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, len(labels))
resnet.load_state_dict(torch.load('./assets/resnet50CatsVdogs.pth', map_location=torch.device('cpu')))
transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    preds = nn.functional.softmax(resnet(img)[0], dim=0)
    return {labels[i]: float(preds[i]) for i in range(len(labels))}

title = "Cats Vs Dogs"
description = "Demo for cat-dog classifier. To use it, simply upload the picture, or click one of the examples below to load them."

inputs = gr.inputs.Image()
outputs = gr.outputs.Label(num_top_classes=2)
gr.Interface(
    fn=predict, 
    inputs=inputs,
    outputs=outputs,
    title=title, 
    description=description, 
    allow_flagging = False,
    layout = 'horizontal',
    examples = [['./assets/cat1.png'],['./assets/dog1.jpg']]
    ).launch()