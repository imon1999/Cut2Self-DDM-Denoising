import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, models
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import trange
from torch.autograd import Variable
from partialconv2d import PartialConv2d
from model import self2self


def image_loader(image, device, p1, p2):
    """load image, returns cuda tensor"""
    # Fixed loader
    loader = T.Compose([T.RandomHorizontalFlip(p1),
                        T.RandomVerticalFlip(p2),
                        T.ToTensor()
                        ])
    image = Image.fromarray(image.astype(np.uint8))
    # image = loader(image).float()
    # image = torch.tensor(image)
    image = image.unsqueeze(0).to(device)
    return image


if __name__ == "__main__":
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)
    model = self2self(3)
    # Path to the directory
    path = "/"
    # Extract the list of filenames
    files = glob.glob(path + '*', recursive=False)
    folder_list = []
    # Loop to print the filenames
    for filename in files:
        folder_list.append(filename)
    print(len(folder_list))
    image_list = []
    image_folder = []
    for filepath in glob.iglob("/*.png"):
        image_list.append(filepath)
    print(len(image_list))
    z = 0

    for z in range(len(image_list)):
        img = np.array(Image.open(image_list[z]))
        # img = np.stack((img1,) * 3, axis=-1)
        print("Start new image running")
        print(img.shape)
        learning_rate = 1e-4
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        w, h, c = img.shape
        rate = 0.5
        NPred = 100
        x = torch.Tensor(img)
        slice_avg = torch.tensor([1, 3, w, h]).to(device)
        i = []
        l = []

        for itr in range(100000):
            stdev = torch.std(x, dim=(0, 1, 2), keepdim=True)
            probs = torch.clip(torch.normal(mean=(1 - rate), std=stdev), 0, 1)
            mask = torch.bernoulli(probs).to(device)

            img_input = img * mask.cpu().numpy()
            y = (1 - mask.cpu().numpy()) * img

            p1, p2 = np.random.rand(), np.random.rand()
            img_input_tensor = image_loader((img_input).astype(np.uint8), device, p1, p2)
            y = image_loader((y).astype(np.uint8), device, p1, p2)
            mask = np.expand_dims(np.transpose(convrt, [2, 0, 1]), 0)
            mask = torch.tensor(mask).to(device, dtype=torch.float32)

            model.train()
            img_input_tensor = img_input_tensor * mask
            output = model(img_input_tensor, mask)

            if itr == 0:
                slice_avg = loader(output)
            else:
                slice_avg = slice_avg * 0.99 + loader(output) * 0.01

            loss = torch.sum(abs(output - y) * (1 - mask)) / torch.sum(1 - mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Iteration {itr + 1}, loss = {loss.item() * 100:.4f}")
            i.append(itr + 1)
            l.append(loss.item() * 100)

            if (itr + 1) % 1000 == 0:
                model.eval()
                img_array = []
                sum_preds = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                for j in range(NPred):
                    stdev = torch.std(x, dim=(0, 1, 2), keepdim=True)
                    probs = torch.clip(torch.normal(mean=(1 - rate), std=stdev), 0, 1)
                    mask = torch.bernoulli(probs).to(device)

                    img_input = img * mask.cpu().numpy()
                    img_input_tensor = image_loader((img_input).astype(np.uint8), device, 0.1, 0.1)
                    mask = np.expand_dims(np.transpose(convrt, [2, 0, 1]), 0)
                    mask = torch.tensor(mask).to(device, dtype=torch.float32)

                    output_test = model(img_input_tensor, mask)
                    sum_preds[:, :, :] += np.transpose(output_test.detach().cpu().numpy(), [2, 3, 1, 0])[:, :, :, 0]
                    img_array.append(np.transpose(output_test.detach().cpu().numpy(), [2, 3, 1, 0])[:, :, :, 0])
                if z == z:
                    k = z
                    print("k= " + str(k) + " image saving done")
                    # calculate avg
                    average = np.squeeze(np.uint8(np.clip(np.average(img_array, axis=0), 0, 1) * 255))
                    base_filename = os.path.basename(image_list[z])
                    input_name = os.path.splitext(base_filename)[0]
                    avg_path = os.path.join(folder_list[z], f"{input_name}_avg_{itr + 1}.png")
                    write_img = Image.fromarray(average)
                    write_img.save(avg_path)

                    # calculate median
                    median = np.squeeze(np.uint8(np.clip(np.median(img_array, axis=0), 0, 1) * 255))
                    med_path = os.path.join(folder_list[z], f"{input_name}_med_{itr + 1}.png")
                    write_img2 = Image.fromarray(median)
                    write_img2.save(med_path)

                k = k + z
    z + 1

