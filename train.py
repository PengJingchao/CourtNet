import os, time, cv2
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CyclicLR, LambdaLR
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from Backbones.vit import VisionTransformer
from Backbones.models_denseformer import DenseFormer
from Backbones.discriminator import discriminator

from utils import *
from config import config
from dataset.MFIRST import G1G2Dataset
# from dataset.SIRST import G1G2Dataset



def get_customized_schedule_with_warmup(optimizer, num_warmup_steps, d_model=1.0, last_epoch=-1):
    def lr_lambda(current_step):
        current_step += 1

        arg1 = current_step ** -0.5
        arg2 = current_step * (num_warmup_steps ** -1.5)
        return (d_model ** -0.5) * min(arg1, arg2)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(g1_path_checkpoint=None, g2_path_checkpoint=None, dis_path_checkpoint=None, RESUME=False):
    assert RESUME is False or g1_path_checkpoint is not None, 'if RESUME, checkpoint must be specified!'
    # output_dir
    root_result_dir = os.path.join('pytorch_outputs')
    os.makedirs(root_result_dir, exist_ok=True)
    model_result_dir = os.path.join(root_result_dir, 'models')
    os.makedirs(model_result_dir, exist_ok=True)
    images_dir = os.path.join(root_result_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # log
    log_dir = os.path.join(root_result_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir)

    # dataset
    trainsplit = G1G2Dataset(mode='train')
    trainset = DataLoader(trainsplit, batch_size=config.mini_batch_size, pin_memory=True,
                          num_workers=4, shuffle=True, drop_last=True)
    testsplit = G1G2Dataset(mode='test.py')
    testset = DataLoader(testsplit, batch_size=1, pin_memory=True,
                         num_workers=4, shuffle=False, drop_last=True)

    # Model
    g1 = DenseFormer()
    g2 = VisionTransformer()
    dis = discriminator()
    if g1_path_checkpoint is not None:
        g1_checkpoint = torch.load(g1_path_checkpoint)
        g1.load_state_dict(g1_checkpoint['model_state'])
        g2_checkpoint = torch.load(g2_path_checkpoint)
        g2.load_state_dict(g2_checkpoint['model_state'])
        dis_checkpoint = torch.load(dis_path_checkpoint)
        dis.load_state_dict(dis_checkpoint['model_state'])
        print('Load .pth successfully....')
    g1.cuda()
    g2.cuda()
    dis.cuda()

    # optimizer

    optimizer_g1 = torch.optim.Adam(g1.parameters(), lr=1e-2)
    optimizer_g2 = torch.optim.Adam(g2.parameters(), lr=1e-2)
    optimizer_dis = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=5e-4)
    if RESUME:
        optimizer_g1.load_state_dict(g1_checkpoint['optimizer_state'])
        optimizer_g2.load_state_dict(g2_checkpoint['optimizer_state'])
        optimizer_dis.load_state_dict(dis_checkpoint['optimizer_state'])
        print('Resume last training....')

    # scheduler_g1 = StepLR(optimizer_g1, step_size=config.step_size, gamma=config.gamma)
    # scheduler_g2 = StepLR(optimizer_g2, step_size=config.step_size, gamma=config.gamma)
    # scheduler_dis = StepLR(optimizer_dis, step_size=config.step_size, gamma=config.gamma)
    scheduler_g1 = get_customized_schedule_with_warmup(optimizer_g1, num_warmup_steps=200, d_model=728)
    scheduler_g2 = get_customized_schedule_with_warmup(optimizer_g2, num_warmup_steps=200, d_model=728)
    scheduler_dis = StepLR(optimizer_dis, step_size=config.step_size, gamma=config.gamma)

    # loss
    loss1 = nn.BCEWithLogitsLoss()

    it = 0
    start_epoch = 0
    if RESUME:
        start_epoch = g1_checkpoint['epoch']
        it = g1_checkpoint['it']

    for epoch in range(start_epoch + 0, start_epoch + config.max_epoch_num):
        # epoch
        total_it_per_epoch = len(trainset)
        for bt_idx, data in enumerate(tqdm(trainset)):
            # batch
            torch.cuda.empty_cache()
            it = it + 1
            summary_writer.add_scalar('lr/g1', float(optimizer_g1.param_groups[0]['lr']), it)
            summary_writer.add_scalar('lr/g2', float(optimizer_g2.param_groups[0]['lr']), it)
            summary_writer.add_scalar('lr/dis', float(optimizer_dis.param_groups[0]['lr']), it)

            # dis
            dis.train()
            g1.eval()
            g2.eval()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_dis.zero_grad()

            # cuda
            input_images, output_images = data['input_images'], data['output_images']  # [B, 1, 128, 128]
            input_images = input_images.cuda(non_blocking=True).float()
            output_images = output_images.cuda(non_blocking=True).float()

            with torch.no_grad():
                g1_out = g1(input_images)  # [B, 1, 128, 128]
                g1_out = torch.clamp(g1_out, 0.0, 1.0)

                g2_out = g2(input_images)  # [B, 1, 128, 128]
                g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1)  # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1)  # [B, 2, 128, 128]
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1)  # [B, 2, 128, 128]
            disc_input = torch.cat([pos1, neg1, neg2], dim=0)  # [3*B, 2, 128, 128]

            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input)  # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(config.mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(config.mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)

            ES0 = torch.mean(loss1(logits_real, gen_gt))
            ES1 = torch.mean(loss1(logits_fake1, gen_gt1))
            ES2 = torch.mean(loss1(logits_fake2, gen_gt2))

            disc_loss = ES0 + ES1 + ES2
            summary_writer.add_scalar('loss/dis', disc_loss, it)
            # logger.info(" discriminator loss is {}".format(disc_loss))
            disc_loss.backward()
            optimizer_dis.step()

            # g1
            dis.eval()
            g1.train()
            g2.eval()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_dis.zero_grad()

            g1_out = g1(input_images)  # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)
            # MD1 = torch.mean(torch.mul(torch.pow(g1_out - output_images, 2), output_images))
            # FA1 = torch.mean(torch.mul(torch.pow(g1_out - output_images, 2), 1 - output_images))
            # MF_loss1 = config.lambda1 * MD1 + FA1
            axes = tuple(range(2, len(output_images.shape)))
            precision = torch.mean(torch.sum(g1_out * output_images, axes) / (torch.sum(g1_out, axes) + 1e-6))
            recall = torch.mean(torch.sum(g1_out * output_images, axes) / (torch.sum(output_images, axes) + 1e-6))
            MF_loss1 = -50 * torch.pow((1 - precision), 3) * precision.log() - torch.pow((1 - recall), 3) * recall.log()
            # F1 = 2 * recall * precision / (recall + precision + 1e-6)
            # MF_loss1 = -10 * F1.log()

            with torch.no_grad():
                g2_out = g2(input_images)  # [B, 1, 128, 128]
                g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1)  # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1)  # [B, 2, 128, 128]
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1)  # [B, 2, 128, 128]
            disc_input = torch.cat([pos1, neg1, neg2], dim=0)  # [3*B, 2, 128, 128]

            with torch.no_grad():
                logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input)  # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(config.mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(config.mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)

            gen_adv_loss1 = torch.mean(loss1(logits_fake1, gen_gt))
            gen_loss1 = 100 * MF_loss1 + 10 * gen_adv_loss1 + 1 * Lgc
            summary_writer.add_scalar('loss/g1', gen_loss1, it)
            # logger.info(" g1 loss is {}".format(gen_loss1))

            gen_loss1.backward()
            optimizer_g1.step()
            scheduler_g1.step()

            # g2
            dis.eval()
            g1.eval()
            g2.train()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_dis.zero_grad()

            with torch.no_grad():
                g1_out = g1(input_images)  # [B, 1, 128, 128]
                g1_out = torch.clamp(g1_out, 0.0, 1.0)

            g2_out = g2(input_images)  # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)
            axes = tuple(range(2, len(output_images.shape)))
            precision = torch.mean(torch.sum(g2_out * output_images, axes) / (torch.sum(g2_out, axes) + 1e-6))
            recall = torch.mean(torch.sum(g2_out * output_images, axes) / (torch.sum(output_images, axes) + 1e-6))
            MF_loss2 = -50 * torch.pow((1 - precision), 3) * precision.log() - torch.pow((1 - recall), 3) * recall.log()

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1)  # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1)  # [B, 2, 128, 128]
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1)  # [B, 2, 128, 128]
            disc_input = torch.cat([pos1, neg1, neg2], dim=0)  # [3*B, 2, 128, 128]

            with torch.no_grad():
                logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input)  # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(config.mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(config.mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)

            gen_adv_loss2 = torch.mean(loss1(logits_fake2, gen_gt))
            gen_loss2 = 100 * MF_loss2 + 10 * gen_adv_loss2 + 1 * Lgc
            summary_writer.add_scalar('loss/g2', gen_loss2, it)

            gen_loss2.backward()
            optimizer_g2.step()
            scheduler_g2.step()

        # test
        sum_val_loss_g1 = 0
        sum_val_false_ratio_g1 = 0
        sum_val_detect_ratio_g1 = 0
        sum_val_F1_g1 = 0

        sum_val_loss_g2 = 0
        sum_val_false_ratio_g2 = 0
        sum_val_detect_ratio_g2 = 0
        sum_val_F1_g2 = 0

        sum_val_loss_g3 = 0
        sum_val_false_ratio_g3 = 0
        sum_val_detect_ratio_g3 = 0
        sum_val_F1_g3 = 0

        for bt_idx_test, data in enumerate(tqdm(testset)):
            g1.eval()
            g2.eval()
            dis.eval()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_dis.zero_grad()

            with torch.no_grad():
                input_images, output_images = data['input_images'], data['output_images']  # [B, 1, 128, 128]
                input_images = input_images.cuda(non_blocking=True).float()
                output_images = output_images.cuda(non_blocking=True).float()

                g1_out = g1(input_images)  # [B, 1, 128, 128]
                g1_out = torch.clamp(g1_out, 0.0, 1.0)

                g2_out = g2(input_images)  # [B, 1, 128, 128]
                g2_out = torch.clamp(g2_out, 0.0, 1.0)

                pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1)  # [B, 2, 128, 128]
                neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1)  # [B, 2, 128, 128]
                neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1)  # [B, 2, 128, 128]
                disc_input = torch.cat([pos1, neg1, neg2], dim=0)  # [3*B, 2, 128, 128]
                _, logits_fake1, logits_fake2, _ = dis(disc_input)

                g3_out = (g1_out * (logits_fake1[:, 0] / (logits_fake1[:, 0] + logits_fake2[:, 0])) + g2_out * (
                            logits_fake2[:, 0] / (logits_fake1[:, 0] + logits_fake2[:, 0])))  # jiaquan的方式进行融合

                output_images = output_images.cpu().numpy()
                g1_out = g1_out.detach().cpu().numpy()
                g2_out = g2_out.detach().cpu().numpy()
                g3_out = g3_out.detach().cpu().numpy()
                # g1
                val_loss_g1 = np.mean(np.square(g1_out - output_images))
                sum_val_loss_g1 += val_loss_g1
                val_false_ratio_g1 = np.mean(np.maximum(0, g1_out - output_images))
                sum_val_false_ratio_g1 += val_false_ratio_g1
                val_detect_ratio_g1 = np.sum(g1_out * output_images) / np.maximum(np.sum(output_images), 1)
                sum_val_detect_ratio_g1 += val_detect_ratio_g1
                val_F1_g1 = calculateF1Measure(g1_out, output_images, 0.5)
                sum_val_F1_g1 += val_F1_g1

                # g2
                val_loss_g2 = np.mean(np.square(g2_out - output_images))
                sum_val_loss_g2 += val_loss_g2
                val_false_ratio_g2 = np.mean(np.maximum(0, g2_out - output_images))
                sum_val_false_ratio_g2 += val_false_ratio_g2
                val_detect_ratio_g2 = np.sum(g2_out * output_images) / np.maximum(np.sum(output_images), 1)
                sum_val_detect_ratio_g2 += val_detect_ratio_g2
                val_F1_g2 = calculateF1Measure(g2_out, output_images, 0.5)
                sum_val_F1_g2 += val_F1_g2

                # g3
                val_loss_g3 = np.mean(np.square(g3_out - output_images))
                sum_val_loss_g3 += val_loss_g3
                val_false_ratio_g3 = np.mean(np.maximum(0, g3_out - output_images))
                sum_val_false_ratio_g3 += val_false_ratio_g3
                val_detect_ratio_g3 = np.sum(g3_out * output_images) / np.maximum(np.sum(output_images), 1)
                sum_val_detect_ratio_g3 += val_detect_ratio_g3
                val_F1_g3 = calculateF1Measure(g3_out, output_images, 0.5)
                sum_val_F1_g3 += val_F1_g3

                # save pic
                output_image1 = np.squeeze(g1_out * 255.0)  # /np.maximum(output_image1.max(),0.0001))
                output_image2 = np.squeeze(g2_out * 255.0)  # /np.maximum(output_image2.max(),0.0001))
                output_image3 = np.squeeze(g3_out * 255.0)  # /np.maximum(output_image3.max(),0.0001))
                cv2.imwrite(os.path.join(images_dir, '%05d_G1.png' % (bt_idx_test)), np.uint8(output_image1))
                cv2.imwrite(os.path.join(images_dir, '%05d_G2.png' % (bt_idx_test)), np.uint8(output_image2))
                cv2.imwrite(os.path.join(images_dir, '%05d_Res.png' % (bt_idx_test)), np.uint8(output_image3))

        # logger.info("======================== g1 results ============================")
        avg_val_loss_g1 = sum_val_loss_g1 / len(testsplit)
        avg_val_false_ratio_g1 = sum_val_false_ratio_g1 / len(testsplit)
        avg_val_detect_ratio_g1 = sum_val_detect_ratio_g1 / len(testsplit)
        avg_val_F1_g1 = sum_val_F1_g1 / len(testsplit)

        summary_writer.add_scalar('valloss/g1', avg_val_loss_g1, epoch + 1)
        summary_writer.add_scalar('false_alarm_rate/g1', avg_val_false_ratio_g1, epoch + 1)
        summary_writer.add_scalar('detection_rate/g1', avg_val_detect_ratio_g1, epoch + 1)
        summary_writer.add_scalar('F1_measure/g1', avg_val_F1_g1, epoch + 1)

        # logger.info("======================== g2 results ============================")
        avg_val_loss_g2 = sum_val_loss_g2 / len(testsplit)
        avg_val_false_ratio_g2 = sum_val_false_ratio_g2 / len(testsplit)
        avg_val_detect_ratio_g2 = sum_val_detect_ratio_g2 / len(testsplit)
        avg_val_F1_g2 = sum_val_F1_g2 / len(testsplit)

        summary_writer.add_scalar('valloss/g2', avg_val_loss_g2, epoch + 1)
        summary_writer.add_scalar('false_alarm_rate/g2', avg_val_false_ratio_g2, epoch + 1)
        summary_writer.add_scalar('detection_rate/g2', avg_val_detect_ratio_g2, epoch + 1)
        summary_writer.add_scalar('F1_measure/g2', avg_val_F1_g2, epoch + 1)

        # logger.info("======================== g3 results ============================")
        avg_val_loss_g3 = sum_val_loss_g3 / len(testsplit)
        avg_val_false_ratio_g3 = sum_val_false_ratio_g3 / len(testsplit)
        avg_val_detect_ratio_g3 = sum_val_detect_ratio_g3 / len(testsplit)
        avg_val_F1_g3 = sum_val_F1_g3 / len(testsplit)

        summary_writer.add_scalar('valloss/g3', avg_val_loss_g3, epoch + 1)
        summary_writer.add_scalar('false_alarm_rate/g3', avg_val_false_ratio_g3, epoch + 1)
        summary_writer.add_scalar('detection_rate/g3', avg_val_detect_ratio_g3, epoch + 1)
        summary_writer.add_scalar('F1_measure/g3', avg_val_F1_g3, epoch + 1)

        print('current epoch {}/{}, total iteration: {}, g1 F1: {}, g2 F1: {}, g3 F1: {}'.format(
            epoch + 1, config.max_epoch_num, it, avg_val_F1_g1, avg_val_F1_g2, avg_val_F1_g3))

        ############# save model
        ckpt_name1 = os.path.join(model_result_dir, 'g1_epoch_{}_batch_{}'.format(epoch + 1, bt_idx + 1))
        ckpt_name2 = os.path.join(model_result_dir, 'g2_epoch_{}_batch_{}'.format(epoch + 1, bt_idx + 1))
        ckpt_name3 = os.path.join(model_result_dir, 'dis_epoch_{}_batch_{}'.format(epoch + 1, bt_idx + 1))
        save_checkpoint(checkpoint_state(g1, optimizer_g1, epoch + 1, it), filename=ckpt_name1)
        save_checkpoint(checkpoint_state(g2, optimizer_g2, epoch + 1, it), filename=ckpt_name2)
        save_checkpoint(checkpoint_state(dis, optimizer_dis, epoch + 1, it), filename=ckpt_name3)

        # scheduler_g1.step()
        # scheduler_g2.step()
        scheduler_dis.step()


if __name__ == '__main__':
    train()
    tensorboard --logdir=/home/pjc/MyProgram/MDvsFA_cGAN-master/pytorch_outputs/logs

    
