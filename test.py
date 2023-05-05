import os, time, cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from Backbones.vit import VisionTransformer
from Backbones.models_denseformer import DenseFormer
from Backbones.discriminator import discriminator

from utils import *
from dataset.MFIRST import G1G2Dataset
# from dataset.SIRST import G1G2Dataset


def test(g1_path_checkpoint, g2_path_checkpoint, dis_path_checkpoint, save_pic=False):
    root_result_dir = os.path.join('pytorch_outputs')
    os.makedirs(root_result_dir, exist_ok=True)
    image_result_dir = os.path.join(root_result_dir, 'images')
    os.makedirs(image_result_dir, exist_ok=True)

    # dataset
    testsplit = G1G2Dataset(mode='test.py')
    testset = DataLoader(testsplit, batch_size=1, pin_memory=True,
                         num_workers=1, shuffle=False, drop_last=True)

    # Model
    g1 = DenseFormer()
    g2 = VisionTransformer()
    dis = discriminator()
    g1_checkpoint = torch.load(g1_path_checkpoint)
    g1.load_state_dict(g1_checkpoint['model_state'])
    g2_checkpoint = torch.load(g2_path_checkpoint)
    g2.load_state_dict(g2_checkpoint['model_state'])
    dis_checkpoint = torch.load(dis_path_checkpoint)
    dis.load_state_dict(dis_checkpoint['model_state'])
    print('Load checkpoint successfully....')
    g1.cuda()
    g2.cuda()
    dis.cuda()

    # 
    sum_val_loss_g1 = 0
    sum_val_false_ratio_g1 = 0
    sum_val_detect_ratio_g1 = 0
    sum_val_Precision_g1 = 0
    sum_val_Recall_g1 = 0
    sum_val_F1_g1 = 0
    g1_time = 0

    sum_val_loss_g2 = 0
    sum_val_false_ratio_g2 = 0
    sum_val_detect_ratio_g2 = 0
    sum_valPrecision_g2 = 0
    sum_valRecall_g2 = 0
    sum_val_F1_g2 = 0
    g2_time = 0

    sum_val_loss_g3 = 0
    sum_val_false_ratio_g3 = 0
    sum_val_detect_ratio_g3 = 0
    sum_valPrecision_g3 = 0
    sum_valRecall_g3 = 0
    sum_val_F1_g3 = 0
    g3_time = 0

    for bt_idx_test, data in enumerate(tqdm(testset)):
        g1.eval()
        g2.eval()
        dis.eval()

        # cuda
        input_images, output_images = data['input_images'], data['output_images']  # [B, 1, 128, 128]
        input_images = input_images.cuda(non_blocking=True).float()
        output_images = output_images.cuda(non_blocking=True).float()

        with torch.no_grad():
            stime1 = time.time()
            g1_out = g1(input_images)  # [B, 1, 128, 128]
            etime1 = time.time()
            g1_time += etime1 - stime1
            g1_out = torch.clamp(g1_out, 0.0, 1.0)

            stime2 = time.time()
            g2_out = g2(input_images)  # [B, 1, 128, 128]
            etime2 = time.time()
            g2_time += etime2 - stime2
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([input_images, input_images], dim=1)  # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1)  # [B, 2, 128, 128]
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1)  # [B, 2, 128, 128]
            disc_input = torch.cat([pos1, neg1, neg2], dim=0)  # [3*B, 2, 128, 128]
            _, logits_fake1, logits_fake2, _ = dis(disc_input)

            g3_out = (g1_out * (logits_fake1[:, 0] / (logits_fake1[:, 0] + logits_fake2[:, 0])) + g2_out * (
                    logits_fake2[:, 0] / (logits_fake1[:, 0] + logits_fake2[:, 0])))  #
            etime3 = time.time()
            g3_time += etime3 - stime1

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
        val_Precision_g1, val_Recall_g1, val_F1_g1 = calculatePreRecF1Measure(g1_out, output_images, 0.5)
        sum_val_Precision_g1 += val_Precision_g1
        sum_val_Recall_g1 += val_Recall_g1
        sum_val_F1_g1 += val_F1_g1

        # g2
        val_loss_g2 = np.mean(np.square(g2_out - output_images))
        sum_val_loss_g2 += val_loss_g2
        val_false_ratio_g2 = np.mean(np.maximum(0, g2_out - output_images))
        sum_val_false_ratio_g2 += val_false_ratio_g2
        val_detect_ratio_g2 = np.sum(g2_out * output_images) / np.maximum(np.sum(output_images), 1)
        sum_val_detect_ratio_g2 += val_detect_ratio_g2
        val_Precision_g2, val_Recall_g2, val_F1_g2 = calculatePreRecF1Measure(g2_out, output_images, 0.5)
        sum_valPrecision_g2 += val_Precision_g2
        sum_valRecall_g2 += val_Recall_g2
        sum_val_F1_g2 += val_F1_g2

        # g3
        val_loss_g3 = np.mean(np.square(g3_out - output_images))
        sum_val_loss_g3 += val_loss_g3
        val_false_ratio_g3 = np.mean(np.maximum(0, g3_out - output_images))
        sum_val_false_ratio_g3 += val_false_ratio_g3
        val_detect_ratio_g3 = np.sum(g3_out * output_images) / np.maximum(np.sum(output_images), 1)
        sum_val_detect_ratio_g3 += val_detect_ratio_g3
        val_Precision_g3, val_Recall_g3, val_F1_g3 = calculatePreRecF1Measure(g3_out, output_images, 0.5)
        sum_valPrecision_g3 += val_Precision_g3
        sum_valRecall_g3 += val_Recall_g3
        sum_val_F1_g3 += val_F1_g3

        # save_pic
        if save_pic:
            output_image1 = np.squeeze(g1_out * 255.0)
            output_image2 = np.squeeze(g2_out * 255.0)
            output_image3 = np.squeeze(g3_out * 255.0)
            cv2.imwrite(os.path.join(image_result_dir, "%05d_G1.png" % bt_idx_test), np.uint8(output_image1))
            cv2.imwrite(os.path.join(image_result_dir, "%05d_G2.png" % bt_idx_test), np.uint8(output_image2))
            cv2.imwrite(os.path.join(image_result_dir, "%05d_Res.png" % bt_idx_test), np.uint8(output_image3))

    print("======================== g1 results ============================")
    avg_val_loss_g1 = sum_val_loss_g1 / len(testsplit)
    avg_val_false_ratio_g1 = sum_val_false_ratio_g1 / len(testsplit)
    avg_val_detect_ratio_g1 = sum_val_detect_ratio_g1 / len(testsplit)
    avg_val_Presicion_g1 = sum_val_Precision_g1 / len(testsplit)
    avg_val_Recall_g1 = sum_val_Recall_g1 / len(testsplit)
    avg_val_F1_g1 = sum_val_F1_g1 / len(testsplit)

    print("================val_L2_loss is %f" % avg_val_loss_g1)
    print("================falseAlarm_rate is %f" % avg_val_false_ratio_g1)
    print("================detection_rate is %f" % avg_val_detect_ratio_g1)
    print("================Presicion measure is %f" % avg_val_Presicion_g1)
    print("================Recall measure is %f" % avg_val_Recall_g1)
    print("================F1 measure is %f" % avg_val_F1_g1)
    print("g1 time is {}".format(g1_time))

    print("======================== g2 results ============================")
    avg_val_loss_g2 = sum_val_loss_g2 / len(testsplit)
    avg_val_false_ratio_g2 = sum_val_false_ratio_g2 / len(testsplit)
    avg_val_detect_ratio_g2 = sum_val_detect_ratio_g2 / len(testsplit)
    avg_val_Presicion_g2 = sum_valPrecision_g2 / len(testsplit)
    avg_val_Recall_g2 = sum_valRecall_g2 / len(testsplit)
    avg_val_F1_g2 = sum_val_F1_g2 / len(testsplit)

    print("================val_L2_loss is %f" % avg_val_loss_g2)
    print("================falseAlarm_rate is %f" % avg_val_false_ratio_g2)
    print("================detection_rate is %f" % avg_val_detect_ratio_g2)
    print("================Presicion measure is %f" % avg_val_Presicion_g2)
    print("================Recall measure is %f" % avg_val_Recall_g2)
    print("================F1 measure is %f" % avg_val_F1_g2)
    print("g2 time is {}".format(g2_time))

    print("======================== g3 results ============================")
    avg_val_loss_g3 = sum_val_loss_g3 / len(testsplit)
    avg_val_false_ratio_g3 = sum_val_false_ratio_g3 / len(testsplit)
    avg_val_detect_ratio_g3 = sum_val_detect_ratio_g3 / len(testsplit)
    avg_val_Presicion_g3 = sum_valPrecision_g3 / len(testsplit)
    avg_val_Recall_g3 = sum_valRecall_g3 / len(testsplit)
    avg_val_F1_g3 = sum_val_F1_g3 / len(testsplit)

    print("================val_L2_loss is %f" % avg_val_loss_g3)
    print("================falseAlarm_rate is %f" % avg_val_false_ratio_g3)
    print("================detection_rate is %f" % avg_val_detect_ratio_g3)
    print("================Presicion measure is %f" % avg_val_Presicion_g3)
    print("================Recall measure is %f" % avg_val_Recall_g3)
    print("================F1 measure is %f" % avg_val_F1_g3)
    print("total time is {}".format(g3_time))
    return avg_val_F1_g3


if __name__ == '__main__':
    g1_path_checkpoint = '<change to your g1.pth>'
    g2_path_checkpoint = '<change to your g2.pth>'
    dis_path_checkpoint = '<change to your dis.pth>'

    test(g1_path_checkpoint, g2_path_checkpoint, dis_path_checkpoint, save_pic=False)

    
