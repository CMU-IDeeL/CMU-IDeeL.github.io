import torch
import torch.nn as nn
import math


class InvertedResidualBlock(nn.Module):
    """
    Intuitively, layers in MobileNet can be split into "feature mixing" 
    and "spatial mixing" layers. You can think of feature mixing as each pixel
    "thinking on its own" about its own featuers, and you can think of spatial
    mixing as pixels "talking with each other". Alternating these two builds
    up a CNN.

    In a bit more detail:

    - The purpose of the "feature mixing" layers is what you've already seen in 
    hw1p2. Remember, in hw1p2, we went from some low-level audio input to
    semantically rich representations of phonemes. Featuring mixing is simply a 
    linear layer (a weight matrix) that transforms simpler features into 
    something more advanced.

    - The purpose of the "spatial mixing" layers is to mix features from different
    spatial locations. You can't figure out a face by looking at each pixel on
    its own, right? So we need 3x3 convolutions to mix features from neighboring
    pixels to build up spatially larger features.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio):
        super().__init__() # Just have to do this for all nn.Module classes

        # Can only do identity residual connection if input & output are the
        # same channel & spatial shape.
        if stride == 1 and in_channels == out_channels:
            self.do_identity = True
        else:
            self.do_identity = False
        
        # Expand Ratio is like 6, so hidden_dim >> in_channels
        hidden_dim = in_channels * expand_ratio

        """
        What is this doing? It's a 1x1 convolutional layer that drastically
        increases the # of channels (feature dimension). 1x1 means each pixel
        is thinking on its own, and increasing # of channels means the network
        is seeing if it can "see" more clearly in a higher dimensional space.

        Some patterns are just more obvious/separable in higher dimensions.

        Also, note that bias = False since BatchNorm2d has a bias term built-in.

        As you go, note the relationship between kernel_size and padding. As you
        covered in class, padding = kernel_size // 2 (kernel_size being odd) to
        make sure input & output spatial resolution is the same.
        """
        self.feature_mixing = nn.Sequential(
            # TODO: Fill this in!
        )

        """
        What is this doing? Let's break it down.
        - kernel_size = 3 means neighboring pixels are talking with each other.
          This is different from feature mixing, where kernel_size = 1.

        - stride. Remember that we sometimes want to downsample spatially. 
          Downsampling is done to reduce # of pixels (less computation to do), 
          and also to increase receptive field (if a face was 32x32, and now
          it's 16x16, a 3x3 convolution covers more of the face, right?). It
          makes sense to put the downsampling in the spatial mixing portion
          since this layer is "in charge" of messing around spatially anyway.

          Note that most of the time, stride is 1. It's just the first block of
          every "stage" (layer \subsetof block \subsetof stage) that we have
          stride = 2.

        - groups = hidden_dim. Remember depthwise separable convolutions in 
          class? If not, it's fine. Usually, when we go from hidden_dim channels
          to hidden_dim channels, they're densely connected (like a linear 
          layer). So you can think of every pixel/grid in an input
          3 x 3 x hidden_dim block being connected to every single pixel/grid 
          in the output 3 x 3 x hidden_dim block.
          What groups = hidden_dim does is remove a lot of these connections.

          Now, each input 3 x 3 block/region is densely connected to the
          corresponding output 3 x 3 block/region. This happens for each of the
          hidden_dim input/output channel pairs independently.
          So we're not even mixing different channels together - we're only 
          mixing spatial neighborhoods. 
          
          Try to draw this out, or come to my (Jinhyung Park)'s OH if you want 
          a more in-depth explanation.
          https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
        """
        self.spatial_mixing = nn.Sequential(
            # TODO: Fill this in!
        )

        """
        What's this? Remember that hidden_dim is quite large - six times the 
        in_channels. So it was nice to do the above operations in this high-dim
        space, where some patterns might be more clear. But we still want to 
        bring it back down-to-earth.

        Intuitively, you can takeaway two reasons for doing this:
        - Reduces computational cost by a lot. 6x in & out channels means 36x
          larger weights, which is crazy. We're okay with just one of input or 
          output of a convolutional layer being large when mixing channels, but 
          not both.
        
        - We also want a residual connection from the input to the output. To 
          do that without introducing another convolutional layer, we want to
          condense the # of channels back to be the same as the in_channels.
          (out_channels and in_channels are usually the same).
        """
        self.bottleneck_channels = nn.Sequential(
            # TODO: Fill this in!
        )

    def forward(self, x):
        out = self.feature_mixing(x)
        out = self.spatial_mixing(out)
        out = self.bottleneck_channels(out)

        if self.do_identity:
            return x + out
        else:
            return out

class MobileNetV2(nn.Module):
    """
    The heavy lifting is already done in InvertedBottleneck.

    Why MobileNetV2 and not V3? V2 is the foundation for V3, which uses "neural
    architecture search" to find better configurations of V2. If you understand
    V2 well, you can totally implement V3!
    """
    def __init__(self, num_classes= 7000):
        super().__init__()

        self.num_classes = num_classes

        """
        First couple of layers are special, just do them here.
        This is called the "stem". Usually, methods use it to downsample or twice.
        """
        self.stem = nn.Sequential(
            # TODO: Fill this in!
        )

        """
        Since we're just repeating InvertedResidualBlocks again and again, we
        want to specify their parameters like this.
        The four numbers in each row (a stage) are shown below.
        - Expand ratio: We talked about this in InvertedResidualBlock
        - Channels: This specifies the channel size before expansion
        - # blocks: Each stage has many blocks, how many?
        - Stride of first block: For some stages, we want to downsample. In a
          downsampling stage, we set the first block in that stage to have
          stride = 2, and the rest just have stride = 1.

        Again, note that almost every stage here is downsampling! By the time
        we get to the last stage, what is the image resolution? Can it still
        be called an image for our dataset? Think about this, and make changes
        as you want.
        """
        self.stage_cfgs = [
            # expand_ratio, channels, # blocks, stride of first block
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Remember that our stem left us off at 16 channels. We're going to 
        # keep updating this in_channels variable as we go
        in_channels = 16

        # Let's make the layers
        layers = []
        for curr_stage in self.stage_cfgs:
            expand_ratio, num_channels, num_blocks, stride = curr_stage
            
            for block_idx in range(num_blocks):
                out_channels = num_channels
                layers.append(InvertedResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    # only have non-trivial stride if first block
                    stride=stride if block_idx == 0 else 1, 
                    expand_ratio=expand_ratio
                ))
                # In channels of the next block is the out_channels of the current one
                in_channels = out_channels 
            
        self.layers = nn.Sequential(*layers) # Done, save them to the class

        # Some final feature mixing
        self.final_block = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        )

        # Now, we need to build the final classification layer.
        self.cls_layer = nn.Sequential(
            # TODO: Fill this in!
            # Pool over & collapse the spatial dimensions to (1, 1)
            # Collapse the trivial (1, 1) dimensions
            # Project to our # of classes
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Usually, I like to use default pytorch initialization for stuff, but
        MobileNetV2 made a point of putting in some custom ones, so let's just
        use them.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.stem(x)
        out = self.layers(out)
        out = self.final_block(out)
        out = self.cls_layer(out)

        return out