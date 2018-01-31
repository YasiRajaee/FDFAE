import torch
import os
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import Generator
from model import Discriminator
import argparse
from PIL import Image


def build_model(config):
    # Define a generator and a discriminator
    # Model hyper-parameters
    c_dim = config.c_dim
    image_size = config.image_size
    g_conv_dim = config.g_conv_dim
    d_conv_dim = config.d_conv_dim
    g_repeat_num = config.g_repeat_num
    d_repeat_num = config.d_repeat_num

    G = Generator(g_conv_dim, c_dim, g_repeat_num)
    D = Discriminator(image_size, d_conv_dim, c_dim, d_repeat_num)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
    return G

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--c_dim', type=int, default=5)
    parser.add_argument('--celebA_crop_size', type=int, default=178)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)

    # Training settings
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)

    # Test settings
    parser.add_argument('--test_model', type=str, default='20_1000')

    # Path
    parser.add_argument('--model_save_path', type=str, default='./models') #model path, needed
    parser.add_argument('--result_path', type=str, default='./')


    config = parser.parse_args()

    # Model hyper-parameters
    image_size = config.image_size

    # Test settings
    test_model = config.test_model

    # Path
    model_save_path = config.model_save_path
    result_path = config.result_path

    # Build tensorboard if use
    G = build_model(config)

    """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
    # Load trained parameters
    G_path = os.path.join(model_save_path, '{}_G.pth'.format(test_model))
    G.load_state_dict(torch.load(G_path))
    G.eval()

    transform = transforms.Compose([
            transforms.CenterCrop(config.celebA_crop_size),
            transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    #image = transform(Image.open("./data/CelebA_nocrop/images/000001.jpg"))
    image = transform(Image.open('/home/yasamanrajaee/ssd_detector/FDFAE/data/input/celebA/img_align_celeba/000001.jpg'))
    image = image.unsqueeze(0)
    label_src = torch.FloatTensor([[0,0,1,1,0]]) #Black, blonde, brown, man, young
    label_target = torch.FloatTensor([[0,1,0,1,1]]) #Black, blonde, brown, man, young

    data_loader = [(image, label_src)]

    for i, (real_x, org_c) in enumerate(data_loader): #celebA_loader
        print(real_x.type)
        real_x = to_var(real_x, volatile=True)
        print(org_c.type)
        #target_c_list = make_celeb_labels(org_c)
        target_c_list = [to_var(label_target, volatile=True)]

        # Start translations
        fake_image_list = [real_x]
        for target_c in target_c_list:
            fake_image_list.append(G(real_x, target_c))
        fake_images = torch.cat(fake_image_list, dim=3)
        save_path = os.path.join(result_path, '{}_fake1.png'.format(i+1))
        save_image(denorm(fake_images.data), save_path, nrow=1, padding=0)
        print('Translated test images and saved into "{}"..!'.format(save_path))
        break
