from lib import *

def create_vgg():
    layers = []
    in_channels = 3

    cfgs = [64, 64, 'M', 128, 128, 'M',
            256, 256, 256, 'MC', 512, 512, 512, 'M',
            512, 512, 512]

    for cfg in cfgs:
        if cfg == 'M': #floor
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == 'MC': #ceiling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1)

            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg
        
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]


    return nn.ModuleList(layers)


def create_extras():
    layers = []
    in_channels = 1024

    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfgs[0], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3)]
    layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3)]

    return nn.ModuleList(layers)

def create_loc_conf(num_classes=21, bbox_ratio_num=[4, 6, 6, 6, 4, 4]):
    loc_layers = []
    conf_layers = []

    # source1
    # loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[0]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[0]*num_classes, kernel_size=3, padding=1)]

    #source2
    #loc
    loc_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*4, kernel_size=3, padding=1)]
    #conf
    conf_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*21, kernel_size=3, padding=1)]

    #source3
    #loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[2]*4, kernel_size=3, padding=1)]
    #conf 
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[2]*21, kernel_size=3, padding=1)]

    #source4
    #loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[3]*4, kernel_size=3, padding=1)]
    #conf 
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[3]*21, kernel_size=3, padding=1)]

    #source5
    #loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[4]*4, kernel_size=3, padding=1)]
    #conf 
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[4]*21, kernel_size=3, padding=1)]

    #source6
    #loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[5]*4, kernel_size=3, padding=1)]
    #conf 
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[5]*21, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)



if __name__ == "__main__":
    vgg = create_vgg()
    # print(vgg)
    extras = create_extras()
    # print(extras)
    loc, conf = create_loc_conf()
    print(loc)
    print(conf)