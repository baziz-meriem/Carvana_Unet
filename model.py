import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),#no need to set the bias cause using batch norm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):#out channel 1 (binary segm)
        
        super(UNET, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        #pool layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))#add a layer to module list
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        #the last down doubleConv
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)#only changes the number of channels (1 in this case)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x =  down(x)
            skip_connections.append(x) # store the resulting feature map for later concatenation
            x = self.pool(x) # adding the maxPool layer after each doubleConv

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse for later use by decoder

        for idx in range(0, len(self.ups),2):

            x = self.ups[idx](x)# feed the x to the transposeConv layer

            skip_connection = skip_connections[idx//2]# get the corresponding index

            if x.shape != skip_connection.shape: # in case the shapes are different
               x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x),dim=1) #concatenate the output of the transposeConv with the skip connection's feature map along the channels dim

            x = self.ups[idx+1](concat_skip) # passing the concatenated version to the doubleConv layer (idx+1)

        return self.final_conv(x) # passing the result through the last layer
    
def test():
    x = torch.randn((3,1,160,160))# batchsize of 3 ,1 channel,160*160 input
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape #check if the input and output have the same shape
    
if __name__ == "__main__":
    test()