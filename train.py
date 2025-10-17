# coding: utf-8

from test import validation
from model import Model
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch
import argparse
import string
import random
import time
import sys
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(
        root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        # 'True' to check training progress with validation function.
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(
                opt.saved_model,  map_location=device), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(
            device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters,
                               lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(
            filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while (True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(
            labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(
                preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        # To see training progress, we also conduct validation when 'iteration == 0'
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(
                        model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(),
                               f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True,
                        help='path to training dataset')
    parser.add_argument('--valid_data', required=True,
                        help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int,
                        default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int,
                        default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000,
                        help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000,
                        help='Interval between each validation')
    parser.add_argument('--saved_model', default='',
                        help="path to model to continue training")
    parser.add_argument('--FT', action='store_true',
                        help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true',
                        help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1,
                        help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95,
                        help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5,
                        help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true',
                        help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off',
                        action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,
                        required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str,
                        required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str,
                        required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    opt.character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ一丁七丈三上下不丑且丕世丘丙丞丟並丫中丰串丸丹主乃久么之乍乎乏乒乓乖乘乙九乞也乳乾亂了予事二于云互五井些亞亟亡亢交亥亦亨享京亭亮亳亶人什仁仃仄仆仇今介仍仔仕他仗付仙仞仡代令以仰仲件价任份仿企伉伊伍伎伏伐休伕伙伯估伴伶伸伺似伽佃但佈位低住佐佑佔何佗佘余佚佛作佝佞佟你佣佤佩佬佯佰佳併佻佼使侃侄來侈例侍侏侑侖侗供依侮侯侵侶侷便係促俄俊俏俐俑俗俘俚保俞俟俠信俬修俯俱俳俵俸俺俾倀倆倉個倍們倒倔倖倘候倚倜借倡倣倦倨倩倪倫倬倭值偃假偈偉偌偏偕做停健側偵偶偷偽傀傅傌傍傑傘備傚傢傣催傭傲傳債傷傻傾僂僅僉像僑僕僖僚僥僧僭僮僰僱僳僵價僻儀儂億儆儉儋儐儒儘償儡優儲儸儻儼儿兀允元兄充兆兇先光克兌免兒兔兕兗兜入內全兩八公六兮共兵其具典兼冀冉冊再冑冒冕冗冠冢冤冥冬冰冶冷冼准凋凌凍凜凝几凡凰凱凳凶凸凹出函刀刁刃分切刈刊刎刑划列初判別刨利刪刮到制刷券刺刻剁剃則削剌前剎剔剖剛剝剡剩剪副割創剷剽剿劃劇劈劉劍劑力功加劣助努劫劭劾勁勃勇勉勒動勖勗勘務勛勝勞募勢勤勦勰勳勵勸勺勻勾勿包匆匈匍匐匕化北匙匝匠匡匣匪匯匱匹匾匿區十千卅升午卉半卑卒卓協南博卜卞占卡卦卯印危即卵卷卸卹卻卿厄厘厚厝原厥厭厲去參又叉及友反叔取受叛叟叡叢口古句另叨叩只叫召叭叮可台叱史右司叻叼吁吃各合吉吊吋同名后吏吐向吒君吝吞吟吠否吧吩含听吳吵吶吸吹吻吼吾呀呂呃呆呈告呎呢呤周味呵呸呻呼命咀咁咄咆和咎咐咒咕咖咚咧咨咩咪咫咬咭咯咱咳咸咽哀品哄哆哇哈哉員哥哦哨哩哪哭哮哲哺哼哿唁唄唆唇唉唐唑唔唧唬售唯唱唸唾啃啄商啊問啕啞啟啡啣啤啥啦啶啼啾喀喂喃善喇喉喊喋喔喘喙喚喜喝喟喧喪喫喬單喱喻嗅嗎嗓嗔嗜嗟嗡嗣嗦嗩嗯嗲嗶嗽嘆嘈嘉嘌嘎嘔嘖嘗嘛嘧嘩嘮嘯嘲嘴嘸嘻嘿噁噎器噩噪噬噴噶噸噹嚇嚎嚏嚐嚥嚨嚮嚴嚼囂囉囊囍囑囚四回因囤困囹固圃圄圈國圍園圓圖團圜土在圩圭圮地圳圻圾址均坊坌坍坎坏坐坑坡坤坦坨坪坭坳坷坻垂垃型垓垚垛垠垢垣垮埂埃埋城埏埒埔埕域埠埤埭執埸培基埼堀堂堅堆堇堉堊堍堠堡堤堪堯堰報場堵塊塌塑塔塗塘塚塞塢塤填塵塾墀境墅墉墊墓墘墜增墟墨墩墮墳墾壁壅壇壑壓壕壘壙壞壟壢壤壩士壬壯壹壺壼壽夏夔夕外夙多夜夠夢夥大天太夫夭央夯失夷夸夾奄奇奈奉奎奏奐契奔奕套奘奚奠奢奧奪奭奮女奴奶奸她好如妃妄妊妍妒妓妖妙妝妞妣妤妥妨妮妲妳妹妻妾姆姊始姐姑姒姓委姚姜姝姣姥姦姨姪姬姮姻姿威娃娑娘娛娜娟娠娣娥娩娶娼婁婆婉婕婚婢婦婧婪婭婷婺婿媒媚媛媧媲媳媽嫁嫂嫉嫌嫖嫚嫡嫣嫦嫩嫻嬉嬋嬌嬗嬛嬤嬪嬬嬰嬴嬸嬿孀孃孌子孔孕孖字存孚孛孜孝孟孢季孤孩孫孱孵學孺孽孿宁它宅宇守安宋完宏宓宕宗官宙定宛宜客宣室宥宦宮宰害宴宵家宸容宿寂寄寅密寇富寐寒寓寞察寡寢寥實寧寨審寫寬寮寰寵寶寸寺封射將專尉尊尋對導小少尕尖尚尤尪尬就尷尸尹尺尻尼尾尿局屁居屆屈屋屍屎屏屑展屜屠屢層履屬屯山岌岐岑岔岡岩岫岬岱岳岷岸峇峋峒峙峨峪峭峰島峻峽崁崆崇崎崑崔崖崗崙崚崛崞崢崤崧崩崮崴崽嵊嵌嵎嵐嵩嵬嵯嶄嶇嶙嶲嶷嶺嶼嶽巍巒巔巖川州巡巢工左巧巨巫差巰己已巳巴巷巽巾市布帆希帑帕帖帘帚帛帝帥師席帳帶帷常帽幄幅幌幕幗幘幟幡幢幣幫干平年并幸幹幻幼幽幾庄庇床庋序底庖店庚府庠度座庫庭庵庶康庸庹庾廁廂廈廉廊廒廓廕廖廚廝廟廠廡廢廣廩廬廳延廷建廿弁弄弈弊弋式弒弓弔引弗弘弛弟弢弦弧弩弱張強弼彆彈彊彌彎彗彘彙形彤彥彧彩彪彬彭彰影彷役彼彿往征待徇很徊律後徐徑徒得徘徙從徠御徨復循徬徭微徵德徹徽心必忌忍忐忑忒忖志忘忙忠忡忤快忭忱念忸忻忽忿怎怒怕怖思怠怡急性怨怩怪怯恁恂恃恆恉恍恐恕恙恢恣恤恥恨恩恪恫恬恭息恰恿悄悅悉悌悍悔悖悚悝悟悠患您悱悲悵悶悸悼悽情惆惇惊惋惑惕惚惜惟惠惡惦惰惱想惶惹惺惻愁愆愈愉愍意愕愚愛感愧愫愷愿慄慈態慌慎慕慘慚慟慢慣慧慨慫慮慰慵慶慷慾憂憊憎憐憑憚憤憧憨憩憫憬憲憶憾懂懃懇懈應懊懋懣懦懲懶懷懸懺懼懾懿戀戇戈戊戌戍戎成我戒戕或戚戛戟戡截戮戰戲戳戴戶戾房所扁扇扈手才扎扑扒打扔托扛扣扦扭扮扯扳扶批扼找承技抃抄抉把抑抒抓投抖抗折抨披抬抱抵抹押抽拂拄拆拇拈拉拋拌拍拎拐拒拓拔拖拗拘拙拚招拜括拭拮拯拱拳拴拷拼拽拾拿持挂指按挑挖挨挪挫振挹挺挽挾捂捅捆捉捍捎捏捐捕捧捨捩捫据捱捲捶捷捺捻掀掃掄授掉掌掎掏掐排掖掘掙掛掟掠採探掣接控推掩措掰掾揀揆揉揍描提插揖揚換握揣揩揪揭揮援揹損搏搐搓搖搗搜搞搡搥搪搬搭搶摒摔摘摧摩摭摯摶摸摹摺摻摽撂撇撈撐撒撓撕撞撤撥撩撫撬播撮撰撲撻撼撾撿擁擂擄擅擇擊擋操擎擒擔擘據擠擢擦擬擭擱擲擴擺擾攀攄攏攔攘攙攜攝攣攤攪攫攬支收攸改攻放政故效敏救敔敕敖敗敘教敝敞敢散敦敬敲整敵敷數斂斃文斌斐斑斗料斛斜斟斡斤斥斧斫斬斯新斷方於施旁旃旄旅旋旌族旒旗既日旦旨早旬旭旱旺旻昀昂昃昆昇昊昌明昏易昔昕昝星映昤春昧昨昭是昱昴昵昶時晃晉晏晒晙晚晝晞晟晤晦晨普景晰晴晶晸智晾暄暇暈暉暌暐暑暖暗暝暠暢暨暫暮暱暴暹曄曆曉曖曙曜曝曠曦曩曬曰曲曳更書曹曼曾替最會月有朋服朔朕朗望朝期木未末本札朱朴朵朽杆杉李杏材村杖杜杞束杠杭杯杰東杲杳杵杷杻杼松板枇枉枋析枕林枚果枝枯架枷枸柁柄柏某柑染柔柘柚柜柝柞柢查柩柬柯柰柱柳柴柵柿栓栖栗栝校栩株栲核根格栽桀桂桃桅框案桉桌桎桐桑桓桴桶桿梁梅梆梏梓梗條梟梢梧梨梭梯械梳梵棄棉棋棍棐棒棕棗棘棚棟棠棣棧棨棪棫森棱棲棵棹棺棻棼椅椋植椎椏椒椰椴椿楊楓楔楙楚楞楠楢楣楨楫業極楷楹概榆榎榔榕榛榜榨榭榮榴榷榻榿槃槊構槍槎槐槓槙槤槭槳槻槽槿樁樂樊樑樓樗標樞樟模樣樵樸樹樺樽橄橇橈橋橘橙機橡橢橫橿檀檄檎檐檔檗檜檢檬檯檳檸檻櫃櫓櫚櫛櫟櫥櫧櫨櫬櫸櫻欄權欒欖欠次欣欲欹欺欽款歃歆歇歉歌歎歐歙歜歟歡止正此步武歧歪歲歷歸歹死殂殃殆殉殊殖殘殫殭殮殯殲段殷殺殼殿毀毅毆毋母每毒毓比毖毗毘毛毫毬毯毽氏氐民氓气氘氙氚氛氟氣氦氧氨氫氬氮氯氰水永氾汀汁求汊汎汐汕汗汙汛汜汝汞江池污汨汪汰汲汴汶決汽汾沁沂沃沅沈沉沌沐沒沓沔沖沙沚沛沫沭沮沱河沸油治沼沽沾沿況泄泅泉泊泌泓法泖泗泚泛泡波泣泥注泫泮泰泳泵洄洋洌洒洗洙洛洞津洩洪洮洱洲洵洶洸洹活洽派流浙浚浦浩浪浮浴海浸涂涅涇消涉涌涕涪涮涯液涵涼涿淀淄淅淆淇淋淌淑淒淖淘淙淚淞淡淤淦淨淪淫淮淯深淳淵混淹淺添淼清渚減渝渠渡渣渤渥渦測渭港渲渴游渺渾湃湄湊湍湖湘湛湜湟湣湧湮湯溉源準溘溜溝溟溢溥溧溪溫溯溲溴溶溺溼溽滁滂滄滅滇滋滌滎滑滓滔滕滘滬滯滲滴滷滸滾滿漁漂漆漏漓演漕漠漢漣漩漪漫漬漱漲漳漶漸漾漿潁潑潔潘潛潞潟潢潤潭潮潯潰潸潼澀澂澄澆澈澍澎澔澗澡澤澧澮澱澳澹激濁濂濃濉濕濘濛濟濠濡濤濫濬濮濰濱濺濾瀆瀉瀋瀏瀑瀕瀘瀚瀛瀝瀟瀦瀧瀨瀾灌灑灘灝灞灣火灰灶灸灼災炅炊炎炒炕炖炙炤炫炬炭炮炯炳炸為烜烤烯烴烷烹烽焉焊焙焚焜無焦焯焰然煃煇煉煌煎煒煙煜煞煤煥照煩煬煮煲煽熄熊熏熒熔熙熟熨熬熱熵熹熾燁燃燄燈燉燊燎燒燕燙營燥燦燧燬燭燮燹燼燾爆爇爍爐爛爪爬爭爰爵父爸爹爺爽爾牂牆片版牌牒牘牙牛牟牠牡牢牧物牲特牽犀犁犍犒犛犢犧犬犯狀狂狄狌狐狗狙狠狡狩狸狹狼狽猁猖猗猛猜猝猞猥猩猴猶猷猾猿獄獅獎獗獠獨獲獵獷獸獻獼獾玄率玉王玕玖玟玠玥玦玩玫玲玳玷玻珀珂珈珊珍珙珞珠珩珪班珮珽現球琅理琇琉琊琍琚琛琢琥琦琨琪琮琯琰琳琴琵琶瑁瑋瑕瑙瑚瑛瑜瑞瑟瑣瑤瑩瑪瑭瑯瑰瑾璀璃璆璇璉璋璐璘璜璞璟璦璧璨璫環璽璿瓊瓏瓚瓜瓢瓣瓦瓮瓶瓷甄甕甘甚甜生產甥甦用甩甫甬甯田由甲申男甸町甾畀畈畋界畏畔留畜畝畢畤略畦番畫異當畸畿疆疇疊疏疑疙疚疝疣疤疥疫疲疳疵疹疼疾病症痊痍痔痕痘痙痛痞痢痣痰痲痳痴痹痺痿瘀瘁瘉瘋瘍瘓瘟瘠瘡瘤瘦瘧瘩瘳瘴療癆癌癒癘癟癡癢癬癮癱癸登發白百皂的皆皇皈皋皎皓皖皚皮皰皺皿盂盃盅盆盈益盎盒盔盛盜盞盟盡監盤盥盧盪目盯盱盲直相盼盾省眈眉看眙真眠眨眩眭眯眶眷眸眺眼眾睇睛睜睞睡睢督睦睪睫睬睹睽睾睿瞄瞇瞋瞎瞑瞞瞢瞧瞪瞬瞭瞰瞳瞻瞿矗矚矛矜矢矣知矩短矮矯石矻矽砂砆砌砍研砝砟砢砦砧砭砲破砵砸硃硅硒硝硤硨硫硬确硯硼硿碇碌碎碑碓碗碘碚碟碣碧碩碭碰碳碴確碻碼碾磁磅磊磋磐磔磚磡磧磨磬磯磲磷磺礁礎礑礙礦礪礫示礽社祀祁祂祇祈祉祊祐祕祖祗祚祛祜祝神祟祠祥祧票祭祺祿禁禍禎福禕禦禧禪禮禱禳禹禺禽禾禿秀私秉秋科秒秕秘租秣秤秦秧秩秫秸移稀稃稅程稍稔稗稙稚稜稞稟稠種稱稷稻稼稽稿穀穆穌積穎穗穢穩穫穴究穹空穿突窄窆窈窒窕窖窗窘窟窠窨窩窪窮窯窺窿竄竅竇竊立竑站竟章竣童竭端競竹竺竽竿笈笏笑笘笙笛笞笠符笨第笭筅筆等筊筋筍筏筐筑筒答策筠筩筮筱筲筵筷箇箋箍箏箔箕算管箬箭箱箴箸節範篆篇築篙篚篠篡篤篦篩篷篾簇簋簑簡簧簪簷簸簽簾簿籀籃籌籍籙籠籤籬籲米籽粉粑粒粕粗粘粟粥粱粲粵粹粽精粿糊糌糕糖糙糜糞糟糠糧糯糰糴糸系糾紀約紅紆紉紊紋納紐紓純紗紘紙級紛紜素紡索紫紮累細紳紹紺絀終絃組結絕絛絜絞絡絢給絨絮統絲絳絹綁綏綑經綜綠綢綦綬維綱網綴綵綸綺綻綽綾綿緊緋緒線緝緞締緣編緩緬緯練緻縉縊縑縛縝縞縣縫縮縯縱縷總績繁繃繆織繕繖繞繡繩繪繫繭繳繹繼纂纇纈續纏纓纖纘纛纜缶缸缺缽罄罌罐罔罕罘罟罩罪置罰署罵罷罹羅羆羈羊羌美羔羚羞群羥羧羨義羯羰羲羸羹羽羿翁翅翊翌翎習翔翟翠翡翥翦翩翮翰翱老考耄者耆耋而耍耐耒耕耖耗耘耙耜耦耳耶耽耿聆聊聒聖聘聚聞聯聰聲聳聶職聽聾聿肄肅肆肇肉肋肌肖肘肚肛肜肝股肢肥肩肪肫肯肱育肺胂胃胄背胎胖胚胛胞胡胤胥胭胯胰胱胸胺能脂脅脆脈脊脖脛脣脩脫脹脾腆腊腋腌腎腐腑腓腔腕腥腦腧腫腰腱腳腴腸腹腺腿膀膂膈膊膏膚膛膜膝膠膨膩膳膺膽膾膿臀臂臃臆臉臊臘臚臞臟臣臥臧臨自臬臭至致臺臻臼臾舀舂舅與興舉舊舌舍舒舔舖舜舞舟航般舵舶舷舸船舺艇艮良艱色艷艾芊芋芍芎芒芙芝芡芥芩芫芬芭芮芯花芳芷芸芹芽芾苑苒苓苔苕苗苛苜苞苟苡苣若苦苧苯英苳苹苻苾茁茂范茄茅茆茉茌茗茛茜茨茫茯茱茲茴茵茶茸茹荀荃荅草荊荏荐荒荔荖荷荸荻荼荽莆莉莊莎莒莓莖莘莞莢莪莫莽菀菁菅菇菊菌菏菑菘菜菠菡菩華菱菲菴菸萁萃萄萇萊萌萍萎萩萬萱萸萼落葆葉著葛葡董葦葫葬葳葵葶葺蒂蒐蒔蒙蒜蒞蒡蒨蒯蒲蒴蒸蒺蒼蒿蓀蓄蓆蓉蓋蓑蓓蓬蓮蓼蔑蔓蔗蔚蔡蔣蔥蔬蔭蔻蔽蕃蕈蕉蕊蕙蕞蕨蕩蕪蕭蕾薄薇薈薊薑薙薛薜薦薨薩薪薯薰薹薺藉藍藏藐藕藜藝藤藥藨藩藪藻蘄蘅蘆蘇蘊蘋蘑蘚蘧蘭蘸蘼蘿虎虐虔處虖虛虜虞號虧虫虱虹蚊蚌蚓蚕蚜蚣蚤蚨蚩蚪蚵蚺蛄蛆蛇蛉蛋蛐蛔蛙蛛蛟蛤蛩蛭蛹蛺蛻蛾蜀蜂蜆蜈蜊蜑蜒蜓蜘蜚蜜蜡蜥蜱蜴蜷蜻蜿蝌蝕蝗蝘蝙蝠蝦蝮蝴蝶蝸螂螃螈融螞螟螢螭螯螳螺螽蟀蟄蟆蟈蟋蟑蟠蟬蟲蟹蟻蟾蠃蠅蠆蠍蠑蠓蠔蠕蠟蠡蠢蠣蠱蠲蠶蠻血行衍術街衙衛衝衡衢衣表衫衰衷衹袁袋袍袒袖袛袞袪被袱裁裂裔裕裘裙補裝裡裨裱裲裳裴裸裹製裾褂複褐褒褚褡褥褪褫褲褶褸褻襄襖襟襠襤襦襪襬襯襲西要覃覆見規覓視覘親覲覺覽觀角觔觚觝解觴觸觿言訂訃訇計訊訌討訐訓訕訖託記訛訝訟訢訣訥訪設許訴訶診註証詐詒詔評詗詛詞詠詢試詩詫詬詭詮詰話該詳詹詼誅誇誌認誓誕誘語誠誡誣誤誥誦誨說誰課誼調諂談請諍諏諒論諛諜諡諤諦諧諫諭諮諱諳諷諸諺諾謀謁謂謄謇謊謎謗謙謚講謝謠謨謫謬謳謹謾譁證譏識譙譚譜警譬譯議譴護譽讀變讓讖讙讚讞谷谿豁豆豈豉豊豌豎豐豔豚象豢豹豺貂貉貊貌貍貓貝貞負財貢貧貨販貪貫責貴貶買貸費貼貽貿賀賁賂賃賄資賈賊賑賓賚賜賞賠賡賢賣賤賦質賬賭賴賸賺購賽贇贈贊贏贓贖贛赤赦赫赭走赴赶起趁超越趕趙趟趣趨足趴趵趾跆跋跌跑跖跗跛距跟跡跣跤跨跪路跳踊踏踐踝踞踢踩踰踴踵踹蹂蹄蹇蹈蹊蹋蹔蹟蹠蹤蹦蹬蹭蹯蹲蹴蹶蹼躁身躬躲躺軀車軋軌軍軒軔軛軟軫軸軾較輅載輒輓輔輕輛輜輝輟輩輪輯輸輻輾輿轂轄轅轉轍轎轟辛辜辟辣辦辨辭辯辰辱農迂迄迅迎近返迢迤迥迦迨迪迫迭述迴迷迺追退送适逃逄逅逆逋逍透逐逑途逕逗這通逛逝逞速造逢連逮逯週進逵逶逸逼逾遁遂遇遊運遍過遏遐遑遒道達違遘遙遜遞遠遣遨適遭遮遲遴遵遶遷選遺遼遽避邀邁邂還邇邈邊邏邑邕邛邠邢那邦邪邯邰邱邳邴邵邸邽邾郁郃郅郇郊郎郛郜郝郡郢部郭郯郴郵都郾鄂鄉鄒鄔鄙鄞鄧鄭鄯鄰鄱鄴鄺酆酉酊酋酌配酐酒酗酚酣酥酩酪酬酮酯酴酵酷酸醅醇醉醋醍醐醒醚醛醜醞醣醫醬醮醯醴釀釁采釉釋里重野量釐金釗釘釜針釣釦釧釵釷鈉鈍鈔鈕鈞鈣鈴鈷鈸鈺鈾鉀鉅鉉鉑鉗鉚鉛鉞鉤鉬鉸鉻銀銃銅銓銖銘銜銥銨銫銳銷銼鋁鋅鋆鋒鋤鋪鋰鋸鋼錄錐錒錕錘錚錠錡錢錦錨錫錮錯錳錶鍊鍋鍍鍛鍥鍬鍰鍱鍵鍶鍼鍾鎂鎊鎌鎖鎗鎚鎡鎧鎩鎬鎮鎳鏈鏖鏗鏘鏞鏟鏡鏢鏤鏽鐘鐙鐮鐵鐸鑄鑊鑑鑒鑣鑫鑰鑱鑲鑷鑼鑽鑾鑿長門閂閃閉開閏閑閒間閔閘閡閣閤閥閨閩閫閭閱閹閻閼閾闆闇闈闊闌闓闔闕闖關闞闡闢阜阡阪阮阱防阻阽阿陀陂附陋陌降限陛陝陞陟陡院陣除陪陰陲陳陵陶陷陸陽隄隅隆隈隊隋隍階隔隕隗隘隙際障隧隨險隱隴隸隹隻隼雀雁雄雅集雇雉雋雌雍雎雒雕雖雘雙雛雜雞離難雨雪雯雰雲零雷雹電需霄霆震霉霍霎霏霑霓霖霙霜霞霧霨霰露霸霹霽霾靂靄靈青靖靚靛靜非靠靡面靨革靳靴靶靼鞅鞋鞍鞏鞘鞠鞣鞭韁韃韋韌韓韜韞韭音韶韻響頁頂頃項順須頊頌頏預頑頒頓頗領頜頡頤頦頭頰頷頸頹頻顆題額顎顏顒顓願顙顛類顥顧顫顯顱風颯颱颶飄飆飛食飢飩飪飭飯飲飴飼飽飾餅餉養餌餐餒餓餘餚餛餞餡館餬餵餽餾饅饈饋饌饑饒饕饗首馗香馥馨馬馭馮馳馴駁駐駒駕駙駛駝駭駱駿騁騎騏騑騙騧騫騰騷騾驁驃驄驅驊驍驕驗驚驛驟驢驤驩驪骨骰骷骸骼髀髏髒髓體髖高髦髮髯髻鬃鬆鬍鬘鬚鬟鬣鬥鬧鬩鬯鬱鬲鬻鬼魁魂魄魅魋魏魔魚魨魯魴魷鮆鮑鮚鮠鮨鮪鮫鮭鮮鮸鯁鯉鯊鯔鯖鯛鯡鯧鯨鯰鯽鰈鰍鰓鰨鰭鰲鰻鱄鱈鱔鱗鱘鱠鱧鱨鱷鱸鱺鳥鳩鳳鳴鳶鴉鴒鴛鴞鴦鴨鴻鴿鵝鵠鵡鵪鵬鵰鵲鶉鶩鶯鶴鶿鷗鷳鷸鷹鷺鸕鸚鸛鸞鹵鹹鹼鹽鹿麂麇麋麒麓麗麝麟麥麩麴麵麻麼麾黃黎黏黑黔默黛黜點黠黧黨黯黴黷黼鼎鼐鼓鼠鼬鼻齊齋齒齡齣齦齧龍龐龔龕龜龢'

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # same with ASTER setting (use 94 char).
        opt.character = string.printable[:-6]

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
