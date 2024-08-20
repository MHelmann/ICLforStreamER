from torch import nn
from torch import cat
from transformers import AutoModel

from config.config import get_deep_learning_config

lm_mp = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased'
}


class EntityPairClassifier(nn.Module):
    def __init__(self):
        super(EntityPairClassifier, self).__init__()
        dl_config = get_deep_learning_config()
        # print(lm_mp[dl_config["lm"]])
        self.bert = AutoModel.from_pretrained(lm_mp[dl_config["lm"]], return_dict=False)
        hidden_size = self.bert.config.hidden_size
        # projector as proposed in SimCLR
        proj_out_size = 768
        self.projector = nn.Linear(hidden_size, proj_out_size)
        self.bn = nn.BatchNorm1d(proj_out_size, affine=False)
        # a fully connected layer for fine tuning
        self.fc = nn.Linear(proj_out_size * 2, 2)

    def forward(self, e_a_tensor, e_b_tensor, e_ab_tensors):
        # left+right
        enc_pair = self.projector(self.bert(e_ab_tensors)[0][:, 0, :]) # (batch_size, emb_size)
        #enc_pair = self.bert(x12)[0][:, 0, :] # (batch_size, emb_size)
        batch_size = len(e_a_tensor)
        # left and right
        enc = self.projector(self.bert(cat((e_a_tensor, e_b_tensor)))[0][:, 0, :])
        #enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
        enc1 = enc[:batch_size] # (batch_size, emb_size)
        enc2 = enc[batch_size:] # (batch_size, emb_size)
        output = self.fc(cat((enc_pair, (enc1 - enc2).abs()), dim=1))
        return output

