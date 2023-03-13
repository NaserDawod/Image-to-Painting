from flask import Flask, request, send_file, render_template, url_for
import io

import torch, os, numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from skimage.transform import resize
import torch.nn as nn


# Define a convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        # define convolutional block with reflection padding, instance normalization, and ReLU activation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


# Define a residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # define residual block with two convolutional blocks (with ReLU activation on the first block only)
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


# Define the generator network
class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        # define initial convolutional block
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1,
                      padding=3, padding_mode="reflect", ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True), )

        self.down_blocks = nn.ModuleList([
            ConvBlock(
                num_features, num_features * 2, kernel_size=3, stride=2,
                padding=1
            ),
            ConvBlock(
                num_features * 2, num_features * 4, kernel_size=3, stride=2,
                padding=1, ),
        ])

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4, num_features * 2, down=False, kernel_size=3,
                    stride=2, padding=1, output_padding=1,
                ),
                ConvBlock(
                    num_features * 2, num_features * 1, down=False, kernel_size=3,
                    stride=2, padding=1, output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1, img_channels, kernel_size=7, stride=1,
            padding=3, padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


def plot_images(gen, normal_img):
    # normal_img = np.array(Image.open(normal_path).convert("RGB"))
    # Apply augmentations
    augmentations = transforms(image=normal_img, image0=normal_img)
    normal = augmentations["image"]

    # Generate an image using the generator
    generated_sample2 = gen(normal)
    drawing = generated_sample2.detach() * 0.5 + 0.5

    drawing = drawing.permute(1, 2, 0)

    drawing = resize(drawing.numpy(), (256, 256))
    drawing = np.round(drawing * 255).astype(np.uint8)

    return drawing


transforms = A.Compose([
    A.Resize(width=320, height=320),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
    ToTensorV2(), ],
    additional_targets={"image0": "image"},
)

gen = torch.load("model.pth")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    # Get the uploaded image from the request object
    image = request.files["image"]

    # Open the image using Pillow
    with Image.open(io.BytesIO(image.read())) as img:
        # Process the image (e.g. resize, apply filters)
        # result_img = img.transpose(Image.FLIP_LEFT_RIGHT).convert("RGB")
        normal_img = np.array(img.convert("RGB"))
        drawing = plot_images(gen, normal_img)
        im = Image.fromarray(drawing)
        # Convert the result image to bytes
        result_bytes = io.BytesIO()
        im.save(result_bytes, format="JPEG")
        result_bytes.seek(0)

    # Return the result image to the client
    return send_file(result_bytes, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
