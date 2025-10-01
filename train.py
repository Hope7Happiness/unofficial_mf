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
from functools import partial

import wandb

class Avger(list):
    def avg(self):
        return sum(self) / len(self)
    def __str__(self):
        return f'{self.avg():.6f}' if len(self) > 0 else 'None'
def _log_config(d, key=None, is_main=False):
    if not is_main:
        return d
    if key is None:
        wandb.config.update(d)
    else:
        wandb.config[key] = d
    return d

if __name__ == '__main__':
    WORK_DIR = Path(__file__).resolve().parent / 'exps' / f'exp_{time.strftime("%Y%m%d-%H%M%S")}'
    
    n_steps = 10000
    # n_steps = 200000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 48
    os.makedirs(WORK_DIR / 'images', exist_ok=True)
    os.makedirs(WORK_DIR / 'checkpoints', exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')
    if accelerator.is_main_process:
        wandb.init(project="zhh_mf_revive", entity='evazhu-massachusetts-institute-of-technology')
        wandb.config.update({
            'work_dir': str(WORK_DIR),
        })
    log_config = partial(_log_config, is_main=accelerator.is_main_process)

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
    # **log_config(dict(new=False), key='dit')
    **log_config(dict(new=True), key='dit')
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0, betas=(0.9,0.95))

    meanflow = MeanFlow(**log_config(dict(channels=1,
                        image_size=32,
                        num_classes=10,
                        flow_ratio=0.50,
                        time_dist=['lognorm', -0.4, 1.0],
                        # cfg_ratio=0.10,
                        cfg_ratio=None, # no cfg
                        # cfg_scale=2.0,
                        # experimental
                        # cfg_uncond='u'
                        
                        loss_fn='new',
                        )))
    if accelerator.is_main_process:
        print('num params: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)), flush=True)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0
    # global_step = -1 # debug
    losses = Avger()
    mse_losses, fm_losses, mf_losses = Avger(), Avger(), Avger()

    log_step = 500
    sample_step = 500

    with tqdm(range(n_steps), disable=not accelerator.is_main_process) as pbar:
        # pbar.set_description("Training")
        model.train()
        for step in pbar:
            data = next(train_dataloader)
            x = data[0].to(accelerator.device)
            c = data[1].to(accelerator.device)
            
            ### WE TRAIN UNCOND MODEL ###

            loss, mse_val, fm_val, mf_val = meanflow.loss(model, x, c=c)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("WARNING: nan or inf loss, abort training.")

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses.append(loss.item())
            mse_losses.append(mse_val.item()), fm_losses.append(fm_val.item()), mf_losses.append(mf_val.item())

            if accelerator.is_main_process:
                loss_info = f'Loss: {losses}    MSE_Loss: {mse_losses}    FM_Loss: {fm_losses}    MF_Loss: {mf_losses}'

                # Extract the learning rate from the optimizer
                lr = optimizer.param_groups[0]['lr']
                lr_info = f'Learning Rate: {lr:.6f}'

                # log_message = f'{current_time}\n{batch_info}    {loss_info}    {lr_info}\n'
                # print(log_message, flush=True)
                pbar.set_description(f'{loss_info} | {lr_info}')

                    # with open('log.txt', mode='a') as n:
                        # n.write(log_message)
                if global_step % log_step == 0:
                    wandb.log({
                        "loss": losses.avg(),
                        "mse_loss": mse_losses.avg(),
                        "fm_loss": fm_losses.avg(),
                        "mf_loss": mf_losses.avg(),
                        "learning_rate": lr,
                        # "step": global_step
                    }, step=global_step)

                    losses = Avger()
                    mse_losses = Avger()

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
