import all_root
import torch.nn as nn
import torch

from netvgg import vgg

class Attention_block(nn.Module):
    def __init__(self,F_g,F_int,flag=1):
        F_l = F_g
        self.flag = flag
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1-x1)+self.relu(x1-g1)
        psi = self.psi(psi)
        # if self.flag:
        #     return psi*x
        # else:
        #     return psi*g
        return psi
    
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(ch_in,ch_in,kernel_size=3,stride=1,padding=1,bias=True),
		    # nn.BatchNorm2d(chi),
			nn.ReLU(),
            nn.ConvTranspose2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    # nn.BatchNorm2d(ch_out),
			nn.ReLU(),
            nn.ConvTranspose2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU()
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
a = torch.randn(1,12,360,240)    

class all_net(nn.Module):
    def __init__(self):
        super(all_net,self).__init__()
        self.backbone = vgg()
        # self.ground_1 = Attention_block(64)
        # self.ground_2 = Attention_block(128)
        # self.ground_3 = Attention_block(256)
        self.diff_1 = Attention_block(64,32)
        self.diff_2 = Attention_block(128,64)
        self.diff_3 = Attention_block(256,128)
        
        self.diff = [self.diff_1, self.diff_2, self.diff_3]
        self.up = nn.Sequential(
            # nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1,bias=True),
			# nn.ReLU(inplace=True),
            # nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1,bias=True),
			nn.ReLU(),
            nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1,bias=True),
            nn.Conv2d(16,1,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(1),
			# nn.ReLU(),
            nn.Sigmoid()
        )
        self.up_1 = up_conv(256,128)
        self.up_2 = up_conv(256,64)
        self.up_3 = up_conv(128,32)
        
    def forward(self, x):
        # 共12个通道，分别为-2，-1，当前帧，背景
        t = []
        for i in range(0,12,3):
            t.append(self.backbone(x[:,i:i+3,:,:]))
            # 第一维度三图片，第二维度是深度
        alp = []
        for i in range(3):
            x = self.diff[i](t[0][i],t[1][i])*0.2
            x += self.diff[i](t[2][i],t[1][i])*0.3
            x += self.diff[i](t[2][i],t[3][i])*0.5
            x = x*t[2][i]
            alp.append(x)
            
        # return alp
        
        out = self.up_1(alp[2])
        out = torch.cat((out,alp[1]),dim=1)
        out = self.up_2(out)
        out = torch.cat((out,alp[0]),dim=1)
        out = self.up_3(out)
        out = self.up(out)
        
        return out
                
# model = all_net()

# b = model(a)

class BinaryDiceLoss(nn.Module):
	def __init__(self):
		super(BinaryDiceLoss, self).__init__()
	
	def forward(self, input, targets):
		# 获取每个批次的大小 N
		N = targets.size()[0]
		# 平滑变量
		smooth = 1
		# 将宽高 reshape 到同一纬度
		input_flat = input.view(N, -1)
		targets_flat = targets.view(N, -1)
	
		# 计算交集
		intersection = input_flat * targets_flat 
		dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
		# 计算一个批次中平均每张图的损失
		loss = 1 - dice_eff.sum() / N
		return loss

    