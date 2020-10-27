import torch.nn as nn
"""
Inheriting from model_m002_crnn5_so.py
"""
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN_RF(nn.Module):

    def __init__(self, nh, nclass):
        super(CRNN_RF, self).__init__()

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, x):
        """
        Recognition network based on BiLSTM
        :param x[b,c,w]: coz h = 1 had to be squeezed, w ~ number words, b ~ number line, c ~ features
        :return: label decoded by BLSTM
        """
        out = x.permute(2, 0, 1)  # [width, batch, channel]

        # print("after squeeze: ", out.size())

        output = self.rnn(out)

        # print("after rnn: ",out.size())

        return output