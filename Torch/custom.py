import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    '''
    Parameters
    ----------
    1. img_size : int 
       Size of the image (it is a square).
    
    2. patch_size : int
       Size of tje patch (it is a square).
      
    3. in_chans : int
       Number of input channels.
    
    4. embed_dim : int
       the embedding dimension.
       
    Attributes
    ----------
    
    1. n_patched : int
       Number of patched inside of our image.
    
    2. proj : nn.Conv2d
       Convolutoinal layer that does both the splitting into patches and their embedding.
    '''
    
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=786):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,)
        
    def forward(self, x):
        '''
        Run forward pass.
        
        Parameters
        -----------
        x : torch.Tensor
            shape '(n_samples, in_chans, img_size, img_size)'.
        
        Returns
        --------
        torch.Tensor
              shape '(n_samples, n_patches, embed_dim)'.

        '''
        
        x = self.proj(x)   # (n_samples, embed_dim, n_patches ** 0.5, n_patched ** 0.5) 
        
        x = x.flatten(2)   # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (n_samples, n_patches, embed_dim)
        
        return x
    

class Attention(nn.Module):
    '''
    Attention mechanism.
    
    Parameters
    ----------
    
    1. dim : int
       the input and out dimension of per token features.
    
    2. n_heads : int
       Number of attention heads 
       
    3. qkv_bias : bool
       If True then we include bias to the query, key and value projections
       
    4. attn_p : float
       Dropout probility applied to the query, key and value tensors.
       
    5. proj_p : float
       Dropout probability applied to the output tensor.
       
    Attributes
    -----------
    
    scale : float
       Normalizing consant for the dot product.
       
    qkv : nn.linear
       Linear projection for the query, key and value
       
    proj : nn.Linear
       Linear mappin that takes in the concatenated output of all attention heads and maps it into a new space.
       
    attn_drop, proj_drop : nn.Dropout
       Dropout layers.
    '''
    
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    
    def forward(self, x):
        '''
        Run forward pass.
        
        Parameters
        -----------
        x : torch.Tensor
            shape '(n_samples, n_patches + 1, dim)'.
        
        Returns
        --------
        torch.Tensor
              shape '(n_samples, n_patches + 1, dim)'.

        '''
        n_samples, n_tokens, dim = x.shape
        x = self.proj(x)   # (n_samples, embed_dim, n_patches ** 0.5, n_patched ** 0.5) 
        
        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) # (n_samples. n_patches +1, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim) # (n_samples, n_patches +1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches +1, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)   # (n_samples, n_heads, head_dim, n_patches+1)
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches+1, n_patches+1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches+1, n_patches+1)
        
        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)   # (n_samples, n_patches+1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches+1, dim)
        
        x = self.proj(weighted_avg) # (n_samples, n_patches+1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches+1, dim)
        
        return x
        
class MLP(nn.Module):
    '''
    Multilayer perceptron
    
    Parameters
    ----------
    
    1. in_features : int
       Number of input features.
       
    2. hidden_features : int
       Number of nodes in the hidden layer
       
    3. out_features : int
       Number of output features.
    
    4. p : float
       Dropout probability.
       
    Attributes
    ---------------
    1. fc : nn.Linear
       the first linear layer.
       
    2. act : nn.GELU
       GELU activation fuction.
       
    3. fc2 : nn.linear
       the second linear layer.
    
    4. drop : nn.Dropout
       Dropout layer.      
    '''
    
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        '''
        Run forward pass.
        
        Parameters
        ------------
        x : torch.Tensor
            Shape '(n_samples, n_patches +1, in_features)'
        
        Returns
        ---------
        torch.Tensor
            Shape '(n_samples, n_patches+1, out_features)'
        '''
        x = self.fc1(x) # (n_samples, n_patches+1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches+1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches+1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches+1, hidden_features)
        x - self.drop(x)  # (n_samples, n_patches+1, hidden_features)
        
        return x
class Block(nn.Module):
    '''
    Transformer block
    
    Parameters
    -------------
    1. dim : int
       Embedding dimension
    
    2. n_heads : int
       Number of attention heads.
       
    3. mlp_ratio : float
       Determines the hidden dimension size of the 'MLP' module with respect to 'dim'.
       
    4. qkv_bias : bool
       If True then we include bias to the query, key and value projections.
       
    5. p, attn_p : float 
       Dropout probability
       
    Attributes
    1. norm1, norm2 : LayerNorm
       Layer normalization
       
    2. attn : Attention
       Attention module.
       
    3. mlp : MLP
       MLP module
    '''
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, 
                              n_heads=n_heads,
                              qkv_bias=qkv_bias, 
                              attn_p=attn_p, 
                              proj_p=p)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, 
                       hidden_features=hidden_features,
                       out_features=dim,)
    def forward(self, x):
        '''
        Run forward pass.
        
        Parameters
        ------------
        x : torch.Tensor
           Shape '(n_samples, n_patches+1, dim)'.
           
        Returns
        --------
        torch.Tensor
           Shape '(n_samples, n_patches+1, dim)'.
        '''
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x 

class VisionTransformer(nn.Module):
    '''
    Simplified implementation of the Vision transformer.
    
    Parameters
    -----------
    1. img_size : int
       both height and the width of the image (it is a sqaure)
    
    2. patch_size : int 
       Both height and the width of the patch (it is a square).
       
    3. in_chans : int
       Number of input channels
       
    4. n_classes : int
       Number of classes.
       
    5. embed_dim : int
       Dimensionality of the token/patch embeddings.
       
    6. depth : int 
       Number of blocks
       
    7. n_heads : int
       Number of attention heads.
       
    8. mlp_ratio : float
       Determines the hidden dimension of the 'MLP' module.
       
    9. qkv_bias : bool
       If True then we include bias to the query, key and value projections.
       
    10. p, attn_p : float
       Dropout probability 
       
    Attributes
    -----------
    patch_embed : PatchEmbed 
       Instance of 'PatchEmbed' layer.
       
    cls_token : nn.Parameter
       Learnable parameter that will represent the first token in the sequence.
       
    pos_emb : nn.Parameter
       Positional embedding of the cls token + all the patches. 
       It has '(n_patches + 1) * embed_dim' elements.
    
    pos_drop : nn.Dropout
       Dropout layer.
       
    blocks : nn.ModuleList
       List of 'Block' modules.
       
    norm : nn.LayerNorm
       Layer normalization.      
    '''
    def __init__(self, img_size=384, 
                 patch_size=16,
                 in_chans=3,
                 n_classes=1000,
                 embed_dim=768,
                 depth=12,
                 n_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 p=0,
                 attn_p=0.,):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim,)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pod_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList([Block(dim=embed_dim,
                                           n_heads=n_heads,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           p=p,
                                           attn_p=attn_p,)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear*(embed_dim, n_classes)
    
    def forward(self, x):
        '''
        Run the forward pass
        
        Parameters 
        -------------
        x : torch.Tensor
           Shape '(n_samples, in_chans, img_size, img_size)'.
        
        Returns
        ----------
        logits : torch.Tensor
           Logits over all the classes - '(n_samples, n_classes)'.
        '''
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        cls_token_final = x[:,0]  # just the CLS token
        x = self.head(cls_token_final)
        
        return x