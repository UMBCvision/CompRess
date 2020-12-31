import torch
import torch.nn as nn
import pdb

class Resnet50_X4(nn.Module) :

    def __init__(self):
        super(Resnet50_X4, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.conv2d = nn.Conv2d(3, 256, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.batch_normalization = norm_layer(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # layer 1

        # b1
        self.conv2d_1 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_1 = norm_layer(1024)

        self.conv2d_2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_2 = norm_layer(256)

        self.conv2d_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_3 = norm_layer(256)

        self.conv2d_4 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_4 = norm_layer(1024)

        # b2
        self.conv2d_5 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_5 = norm_layer(256)

        self.conv2d_6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_6 = norm_layer(256)

        self.conv2d_7 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_7 = norm_layer(1024)

        # b3
        self.conv2d_8 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_8 = norm_layer(256)

        self.conv2d_9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_9 = norm_layer(256)

        self.conv2d_10 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_10 = norm_layer(1024)

        # layer 2

        # b1
        self.conv2d_11 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False)
        self.batch_normalization_11 = norm_layer(2048)

        self.conv2d_12 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_12 = norm_layer(512)

        self.conv2d_13 = nn.Conv2d(512, 512, kernel_size=3, stride=2, bias=False, padding=1)
        self.batch_normalization_13 = norm_layer(512)

        self.conv2d_14 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_14 = norm_layer(2048)

        # b2
        self.conv2d_15 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_15 = norm_layer(512)

        self.conv2d_16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_16 = norm_layer(512)

        self.conv2d_17 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_17 = norm_layer(2048)

        # b3
        self.conv2d_18 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_18 = norm_layer(512)

        self.conv2d_19 = nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_19 = norm_layer(512)

        self.conv2d_20 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_20 = norm_layer(2048)

        # b4
        self.conv2d_21 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_21 = norm_layer(512)

        self.conv2d_22 = nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_22 = norm_layer(512)

        self.conv2d_23 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_23 = norm_layer(2048)

        # layer 3

        # b1
        self.conv2d_24 = nn.Conv2d(2048, 4096, kernel_size=1, stride=2, bias=False)
        self.batch_normalization_24 = norm_layer(4096)

        self.conv2d_25 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_25 = norm_layer(1024)

        self.conv2d_26 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, bias=False, padding=1)
        self.batch_normalization_26 = norm_layer(1024)

        self.conv2d_27 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_27 = norm_layer(4096)

        # b2
        self.conv2d_28 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_28 = norm_layer(1024)

        self.conv2d_29 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False , padding=1)
        self.batch_normalization_29 = norm_layer(1024)

        self.conv2d_30 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_30 = norm_layer(4096)

        # b3
        self.conv2d_31 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_31 = norm_layer(1024)

        self.conv2d_32 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_32 = norm_layer(1024)

        self.conv2d_33 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_33 = norm_layer(4096)

        # b4
        self.conv2d_34 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_34 = norm_layer(1024)

        self.conv2d_35 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_35 = norm_layer(1024)

        self.conv2d_36 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_36 = norm_layer(4096)

        # b5
        self.conv2d_37 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_37 = norm_layer(1024)

        self.conv2d_38 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_38 = norm_layer(1024)

        self.conv2d_39 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_39 = norm_layer(4096)

        # b6
        self.conv2d_40 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_40 = norm_layer(1024)

        self.conv2d_41 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_41 = norm_layer(1024)

        self.conv2d_42 = nn.Conv2d(1024, 4096, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_42 = norm_layer(4096)

        # layer 4

        # b1
        self.conv2d_43 = nn.Conv2d(4096, 8192, kernel_size=1, stride=2, bias=False)
        self.batch_normalization_43 = norm_layer(8192)

        self.conv2d_44 = nn.Conv2d(4096, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_44 = norm_layer(2048)

        self.conv2d_45 = nn.Conv2d(2048, 2048, kernel_size=3, stride=2, bias=False, padding=1)
        self.batch_normalization_45 = norm_layer(2048)

        self.conv2d_46 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_46 = norm_layer(8192)

        # b2
        self.conv2d_47 = nn.Conv2d(8192, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_47 = norm_layer(2048)

        self.conv2d_48 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_48 = norm_layer(2048)

        self.conv2d_49 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_49 = norm_layer(8192)

        # b2
        self.conv2d_50 = nn.Conv2d(8192, 2048, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_50 = norm_layer(2048)

        self.conv2d_51 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, bias=False, padding=1)
        self.batch_normalization_51 = norm_layer(2048)

        self.conv2d_52 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1, bias=False)
        self.batch_normalization_52 = norm_layer(8192)

        self.fc = nn.Linear(8192 , 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight , 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 0.2)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(self.batch_normalization(x))
        x = self.maxpool(x)

        # layer1
        # b1
        shortcut = self.batch_normalization_1(self.conv2d_1(x))
        x = self.relu(self.batch_normalization_2(self.conv2d_2(x)))
        x = self.relu(self.batch_normalization_3(self.conv2d_3(x)))
        x = self.batch_normalization_4(self.conv2d_4(x))
        x = self.relu(x + shortcut)

        # b2
        shortcut = x
        x = self.relu(self.batch_normalization_5(self.conv2d_5(x)))
        x = self.relu(self.batch_normalization_6(self.conv2d_6(x)))
        x = self.batch_normalization_7(self.conv2d_7(x))
        x = self.relu(x + shortcut)

        # b3
        shortcut = x
        x = self.relu(self.batch_normalization_8(self.conv2d_8(x)))
        x = self.relu(self.batch_normalization_9(self.conv2d_9(x)))
        x = self.batch_normalization_10(self.conv2d_10(x))
        x = self.relu(x + shortcut)

        # layer2
        # b1
        shortcut = self.batch_normalization_11(self.conv2d_11(x))
        x = self.relu(self.batch_normalization_12(self.conv2d_12(x)))
        x = self.relu(self.batch_normalization_13(self.conv2d_13(x)))
        x = self.batch_normalization_14(self.conv2d_14(x))
        x = self.relu(x + shortcut)

        # b2
        shortcut = x
        x = self.relu(self.batch_normalization_15(self.conv2d_15(x)))
        x = self.relu(self.batch_normalization_16(self.conv2d_16(x)))
        x = self.batch_normalization_17(self.conv2d_17(x))
        x = self.relu(x + shortcut)

        # b3
        shortcut = x
        x = self.relu(self.batch_normalization_18(self.conv2d_18(x)))
        x = self.relu(self.batch_normalization_19(self.conv2d_19(x)))
        x = self.batch_normalization_20(self.conv2d_20(x))
        x = self.relu(x + shortcut)

        # b4
        shortcut = x
        x = self.relu(self.batch_normalization_21(self.conv2d_21(x)))
        x = self.relu(self.batch_normalization_22(self.conv2d_22(x)))
        x = self.batch_normalization_23(self.conv2d_23(x))
        x = self.relu(x + shortcut)

        # layer3
        # b1
        shortcut = self.batch_normalization_24(self.conv2d_24(x))
        x = self.relu(self.batch_normalization_25(self.conv2d_25(x)))
        x = self.relu(self.batch_normalization_26(self.conv2d_26(x)))
        x = self.batch_normalization_27(self.conv2d_27(x))
        x = self.relu(x + shortcut)

        # b2
        shortcut = x
        x = self.relu(self.batch_normalization_28(self.conv2d_28(x)))
        x = self.relu(self.batch_normalization_29(self.conv2d_29(x)))
        x = self.batch_normalization_30(self.conv2d_30(x))
        x = self.relu(x + shortcut)

        # b3
        shortcut = x
        x = self.relu(self.batch_normalization_31(self.conv2d_31(x)))
        x = self.relu(self.batch_normalization_32(self.conv2d_32(x)))
        x = self.batch_normalization_33(self.conv2d_33(x))
        x = self.relu(x + shortcut)

        # b4
        shortcut = x
        x = self.relu(self.batch_normalization_34(self.conv2d_34(x)))
        x = self.relu(self.batch_normalization_35(self.conv2d_35(x)))
        x = self.batch_normalization_36(self.conv2d_36(x))
        x = self.relu(x + shortcut)

        # b5
        shortcut = x
        x = self.relu(self.batch_normalization_37(self.conv2d_37(x)))
        x = self.relu(self.batch_normalization_38(self.conv2d_38(x)))
        x = self.batch_normalization_39(self.conv2d_39(x))
        x = self.relu(x + shortcut)

        # b6
        shortcut = x
        x = self.relu(self.batch_normalization_40(self.conv2d_40(x)))
        x = self.relu(self.batch_normalization_41(self.conv2d_41(x)))
        x = self.batch_normalization_42(self.conv2d_42(x))
        x = self.relu(x + shortcut)


        # layer4
        # b1
        shortcut = self.batch_normalization_43(self.conv2d_43(x))
        x = self.relu(self.batch_normalization_44(self.conv2d_44(x)))
        x = self.relu(self.batch_normalization_45(self.conv2d_45(x)))
        x = self.batch_normalization_46(self.conv2d_46(x))
        x = self.relu(x + shortcut)

        # b2
        shortcut = x
        x = self.relu(self.batch_normalization_47(self.conv2d_47(x)))
        x = self.relu(self.batch_normalization_48(self.conv2d_48(x)))
        x = self.batch_normalization_49(self.conv2d_49(x))
        x = self.relu(x + shortcut)

        # b3
        shortcut = x
        x = self.relu(self.batch_normalization_50(self.conv2d_50(x)))
        x = self.relu(self.batch_normalization_51(self.conv2d_51(x)))
        x = self.batch_normalization_52(self.conv2d_52(x))
        x = self.relu(x + shortcut)

        x = self.avgpool(x)
        x = torch.flatten(x , 1)

        x = self.fc(x)

        return x

