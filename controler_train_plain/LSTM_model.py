from torch import nn
from torch.autograd import Variable


class LSTM(nn.Module):
	def __init__(self, input_size=34, hidden_size=256, output_size=12, num_layer=2):
		super(LSTM, self).__init__()
		self.layer1 = nn.LSTM(input_size, hidden_size, num_layer,batch_first=True)
		self.layer2 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x, _ = self.layer1(x)
		s, b, h = x.size()
		x = x.view(s * b, h)
		x = self.layer2(x)
		x = x.view(s, b, -1)
		return x


model = LSTM(34, 256, 12, 2)