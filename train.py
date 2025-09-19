from models.dit import MFDiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlow
from accelerate import Accelerator
import time
import os
from PIL import Image
from pathlib import Path

import wandb

if __name__ == '__main__':
    WORK_DIR = Path(__file__).resolve().parent / 'exps' / f'exp_{time.strftime("%Y%m%d-%H%M%S")}'
    
    n_steps = 10000
    # n_steps = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 48
    os.makedirs(WORK_DIR / 'images', exist_ok=True)
    os.makedirs(WORK_DIR / 'checkpoints', exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')
    
    print(f'rank {accelerator.process_index} is running', flush=True)

    # dataset = torchvision.datasets.CIFAR10(
    #     root="cifar",
    #     train=True,
    #     download=True,
    #     transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
    # )
    
    # this is tang
    if accelerator.is_main_process:
        dataset = torchvision.datasets.MNIST(
            root="mnist",
            train=True,
            download=True,
            transform=T.Compose([T.Resize((32, 32)), T.ToTensor(),]),
        )
    accelerator.wait_for_everyone()
    dataset = torchvision.datasets.MNIST(
        root="mnist",
        train=True,
        download=False,
        transform=T.Compose([T.Resize((32, 32)), T.ToTensor(),]),
    )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8
    )
    train_dataloader = cycle(train_dataloader)

    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=1,
        # in_channels=int('cifar' in dataset.__class__.__name__.lower())*2+1,
        dim=256,
        # dim=384,
        depth=8,
        # depth=12,
        num_heads=8,
        # num_heads=6,
        num_classes=10,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    meanflow = MeanFlow(channels=1,
                        image_size=32,
                        num_classes=10,
                        flow_ratio=0.50,
                        time_dist=['lognorm', -0.4, 1.0],
                        cfg_ratio=0.10,
                        cfg_scale=2.0,
                        # experimental
                        cfg_uncond='u')
    if accelerator.is_main_process:
        print('num params: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)), flush=True)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # global_step = 0
    global_step = -1 # debug
    losses = 0.0
    mse_losses = 0.0

    log_step = 500
    sample_step = 500
    
    if accelerator.is_main_process:
        wandb.init(project="zhh_mf_revive", entity='evazhu-massachusetts-institute-of-technology')
        wandb.config.update({
            'work_dir': str(WORK_DIR),
        })

    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()
        for step in pbar:
            data = next(train_dataloader)
            x = data[0].to(accelerator.device)
            c = data[1].to(accelerator.device)

            loss, mse_val = meanflow.loss(model, x, c)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()
            mse_losses += mse_val.item()

            if accelerator.is_main_process:
                if global_step % log_step == 0:
                    current_time = time.asctime(time.localtime(time.time()))
                    batch_info = f'Global Step: {global_step}'
                    loss_info = f'Loss: {losses / log_step:.6f}    MSE_Loss: {mse_losses / log_step:.6f}'

                    # Extract the learning rate from the optimizer
                    lr = optimizer.param_groups[0]['lr']
                    lr_info = f'Learning Rate: {lr:.6f}'

                    log_message = f'{current_time}\n{batch_info}    {loss_info}    {lr_info}\n'
                    print(log_message, flush=True)

                    # with open('log.txt', mode='a') as n:
                        # n.write(log_message)
                    wandb.log({
                        "loss": losses / log_step,
                        "mse_loss": mse_losses / log_step,
                        "learning_rate": lr,
                        # "step": global_step
                    }, step=global_step)

                    losses = 0.0
                    mse_losses = 0.0

            if global_step % sample_step == 0:
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, 'module') else model
                    z = meanflow.sample_each_class(model_module, 10)
                    log_img = make_grid(z, nrow=10)
                    img_save_path = WORK_DIR / f"images/step_{global_step}.png"
                    save_image(log_img, img_save_path)
                    
                    save_img = log_img.clip(0,1).permute(1, 2, 0).cpu().numpy()
                    save_img = (255*save_img).astype('uint8')
                    wandb.log({
                        "sample": wandb.Image(Image.fromarray(save_img))
                    }, step=global_step)
                accelerator.wait_for_everyone()
                model.train()
                
    if accelerator.is_main_process:
        ckpt_path = WORK_DIR / f"checkpoints/step_{global_step}.pt"
        accelerator.save(model_module.state_dict(), ckpt_path)