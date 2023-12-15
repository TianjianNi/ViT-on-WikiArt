import math

import torch
import torch.nn as nn


class PatchEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embedding_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_patches = int(image_size // patch_size) * int(image_size // patch_size)
        self.projection = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embedding_dim,
                                    kernel_size=self.patch_size, stride=patch_size)

    def forward(self, x):  # (batch_size, in_channels, image_size, image_size)
        x = self.projection(x)  # (batch_size, embedding_dim, image_size // patch_size, image_size // patch_size)
        x = torch.flatten(x, start_dim=2)  # (batch_size, embedding_dim, num_patches)
        x = torch.transpose(x, dim0=1, dim1=2)  # (batch_size, num_patches, embedding_dim)
        return x


'''
Test for PatchEmbedding

input_image = torch.rand(128, 3, 27, 27)
model = PatchEmbedding(27, 9, 3, 100)
output_image = model(input_image)
print(output_image.shape)

torch.Size([128, 9, 100])
'''


class Embeddings(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embedding_dim):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(image_size, patch_size, in_channels, embedding_dim)
        self.num_patches = int(image_size // patch_size) * int(image_size // patch_size)
        self.class_token = nn.Parameter(data=torch.zeros(1, 1, embedding_dim))
        self.position_embeddings = nn.Parameter(data=torch.zeros(1, 1 + self.num_patches, embedding_dim))

    def forward(self, x):  # (batch_size, in_channels, image_size, image_size)
        x = self.patch_embeddings(x)  # (batch_size, num_patches, embedding_dim)
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)  # (batch_size, 1, embedding_dim)
        x = torch.cat((class_token, x), dim=1)  # (batch_size, num_patches + 1, embedding_dim)
        x = x + self.position_embeddings  # (batch_size, num_patches + 1, embedding_dim)
        return x


'''
Test for Embedding

input_image = torch.rand(128, 3, 27, 27)
model = Embeddings(27, 9, 3, 100)
output_image = model(input_image)
print(output_image.shape)

torch.Size([128, 10, 100])
'''


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = int(self.embedding_dim // self.num_attention_heads)
        self.qkv = nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 3)
        self.projection = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, x):  # (batch_size, num_patches + 1, embedding_dim)
        qkv = self.qkv(x)  # (batch_size, num_patches + 1, 3 * dim)
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.num_attention_heads, self.head_dim)
        qkv = torch.permute(input=qkv, dims=(2, 0, 3, 1, 4))  # (3, batch_size, num_attention_heads, num_patches + 1, head_dim)
        query, key, value = qkv[0], qkv[1], qkv[2]
        key_transpose = torch.transpose(input=key, dim0=-1, dim1=-2)  # (batch_size, num_attention_heads, head_dim, num_patches + 1)
        dot_product = torch.matmul(input=query, other=key_transpose) * (self.head_dim ** -0.5)  # (batch_size, num_attention_heads, num_patches + 1, num_patches + 1)
        softmax = torch.softmax(input=dot_product, dim=-1)  # (batch_size, num_attention_heads, num_patches + 1, num_patches + 1)
        attention_output = torch.matmul(softmax, value)  # (batch_size, num_attention_heads, num_patches + 1, head_dim)
        attention_output = torch.transpose(input=attention_output, dim0=1, dim1=2)  # (batch_size, num_patches + 1, num_attention_heads, head_dim)
        attention_output = attention_output.flatten(2)  # (batch_size, num_patches + 1, embedding_dim)
        x = self.projection(attention_output)  # (batch_size, num_patches + 1, embedding_dim)
        return x


'''
Test for MultiHeadAttention

input_image = torch.rand(128, 10, 100)
model = MultiHeadAttention(embedding_dim=100, num_attention_heads=5)
output_image = model(input_image)
print(output_image.shape)

torch.Size([128, 10, 100])
'''


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)  # (batch_size, num_patches + 1, hidden_features)
        x = self.act(x)  # (batch_size, num_patches + 1, hidden_features)
        x = self.fc2(x)  # (batch_size, num_patches + 1, out_features)
        return x


class Block(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, mlp_ratio=4.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_features = int(self.embedding_dim * mlp_ratio)
        self.num_attention_heads = num_attention_heads
        self.multi_head_attention = MultiHeadAttention(embedding_dim=embedding_dim, num_attention_heads=num_attention_heads)
        self.mlp = MLP(in_features=self.embedding_dim, hidden_features=self.hidden_features, out_features=self.embedding_dim)
        self.norm1 = nn.LayerNorm(self.embedding_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.embedding_dim, eps=1e-6)

    def forward(self, x):  # (batch_size, num_patches + 1, embedding_dim)
        x = x + self.multi_head_attention(self.norm1(x))  # (batch_size, num_patches + 1, embedding_dim)
        x = x + self.mlp(self.norm2(x))  # (batch_size, num_patches + 1, embedding_dim)
        return x


'''
Test for Block

input_image = torch.rand(128, 10, 100)
model = Block(embedding_dim=100, num_attention_heads=5, mlp_ratio=5)
output_image = model(input_image)
print(output_image.shape)

torch.Size([128, 10, 100])
'''


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=384,
                 patch_size=16,
                 in_channels=3,
                 n_classes=27,
                 embedding_dim=768,
                 num_blocks=12,
                 num_attention_heads=12,
                 mlp_ratio=4.0):
        super().__init__()
        self.embeddings = Embeddings(image_size=image_size, patch_size=patch_size,
                                     in_channels=in_channels, embedding_dim=embedding_dim)
        self.blocks = nn.ModuleList(
            [Block(embedding_dim=embedding_dim, num_attention_heads=num_attention_heads, mlp_ratio=mlp_ratio)
             for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.head = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):  # (batch_size, in_channels, num_patches + 1, embedding_dim)
        x = self.embeddings(x)  # (batch_size, num_patches + 1, embedding_dim)
        for block in self.blocks:
            x = block(x)  # (batch_size, num_patches + 1, embedding_dim)
        x = self.norm(x)  # (batch_size, num_patches + 1, embedding_dim)
        class_token = x[:, 0]  # (batch_size, embedding_dim)
        x = self.head(class_token)  # (batch_size, n_classes)
        return x


'''
Test for VisionTransformer
input_image = torch.rand(128, 3, 100, 100)
model = VisionTransformer(image_size=100,
                          patch_size=20,
                          in_channels=3,
                          n_classes=27,
                          embedding_dim=1000,
                          num_blocks=5,
                          num_attention_heads=20,
                          mlp_ratio=4.0)
output_image = model(input_image)
print(output_image.shape)

torch.Size([128, 27])
'''
