import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd


def create_acc_loss_graph(model_name):
	style.use('ggplot')
	contents = pd.read_csv(f"./logs/{model_name}.csv", index_col='epoch')

	ax1 = plt.subplot2grid((2, 1), (0, 0))
	ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

	ax1.plot(contents.index, contents.cla_acc.rolling(10).mean(), label="Cla Acc")
	ax1.plot(contents.index, contents.dis_acc.rolling(10).mean(), label="Dis Acc")
	ax1.legend(loc=2)

	ax2.plot(contents.index, contents.enc_loss.rolling(10).mean(), label="Enc Loss")
	ax2.plot(contents.index, contents.dec_loss.rolling(10).mean(), label="Dec Loss")
	ax2.plot(contents.index, contents.cla_loss.rolling(10).mean(), label="Cla Loss")
	ax2.plot(contents.index, contents.dis_loss.rolling(10).mean(), label="Dis Loss")
	ax2.legend(loc=2)
	plt.savefig(f"./figures/{model_name}.png", dpi=300)
